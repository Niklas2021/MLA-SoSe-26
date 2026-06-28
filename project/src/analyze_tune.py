"""Cross-Shape-Auswertung der Multi-Shape-Studie - laeuft lokal ohne GPU.

Liest fuer jede Shape aus problems.py die results/tune_<name>.csv und stellt die
Messung der L2/Bandbreiten-Prognose gegenueber. Pro Shape:
  - Tuner-Gewinn (beste gemessene Config vs. feste Default-Config)
  - Modell-Guete: Spearman, Rang der Mess-Siegerin, "Modell-#1 erreicht X% vom
    Optimum" (Regret) - fuer das reine bw-Modell und die register-bereinigte v2

Damit beantworten wir: lohnt sich Tuning ueberhaupt, und WO (in welchem Regime)
traegt die analytische Prognose?

Aufruf:  python analyze_tune.py   (aus project/src/)
"""

import csv
import os

from autotuner.search import (enumerate_candidates, prune, rank,
                              estimate_acc_registers, _classify_einsum)
from autotuner.device_props import GB10
from autotuner.stats import spearman
from problems import PROBLEMS, DEFAULT_CONFIG

REG_FRACTION_V2 = 0.4
DEFAULT_SIG = tuple(DEFAULT_CONFIG[k] for k in
                    ("variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2"))


def sig(c):
    return (c.variant, c.m_prim, c.n_prim, c.k_prim, c.m_l2, c.n_l2)


def batch_of(einsum, shapes):
    _, _, _, _, batch_chars = _classify_einsum(einsum)
    size = {}
    lhs = einsum.replace(" ", "").split("->")[0]
    for tstr, shp in zip(lhs.split(","), shapes):
        for c, s in zip(tstr, shp):
            size[c] = s
    b = 1
    for c in batch_chars:
        b *= size[c]
    return b


def load_csv(name):
    path = os.path.join(os.path.dirname(__file__), "..", "results", f"tune_{name}.csv")
    if not os.path.exists(path):
        return None
    meas = {}
    for r in csv.DictReader(open(path)):
        key = (r["variant"], int(r["m_prim"]), int(r["n_prim"]),
               int(r["k_prim"]), int(r["m_l2"]), int(r["n_l2"]))
        meas[key] = {"ms": float(r["ms"]), "tflops": float(r["tflops"]),
                     "ok": int(r["ok"])}
    return meas


def model_eval(kept, meas, batch, model, reg_clean=False):
    """Wertet ein Modell gegen die Messung aus. Liefert dict mit Spearman, Rang
    der Mess-Siegerin und 'Modell-#1 erreicht X% vom Optimum'."""
    cands = kept
    if reg_clean:
        cands = [c for c in kept if estimate_acc_registers(c) <= REG_FRACTION_V2 * GB10.regs_per_block]
    # nur Configs mit gueltiger Messung
    cands = [c for c in cands if sig(c) in meas and meas[sig(c)]["ok"]]
    if len(cands) < 2:
        return None

    ranked = rank(cands, GB10, batch=batch, model=model)
    order = [sig(c) for c, _ in ranked]
    key = "est_ms_occ" if model == "bw_occ" else "est_ms"

    est = [m[key] for _, m in ranked]
    ms = [meas[s]["ms"] for s in order]
    sp = spearman(est, ms)

    best_sig = min(order, key=lambda s: meas[s]["ms"])
    best_tflops = meas[best_sig]["tflops"]
    model_pick = order[0]
    opt_frac = meas[model_pick]["tflops"] / best_tflops
    winner_rank = order.index(best_sig) + 1
    return {"spearman": sp, "winner_rank": winner_rank,
            "opt_frac": opt_frac, "n": len(cands)}


def main():
    header = (f"{'Shape':12s} {'Regime':22s} {'best':>6s} {'deflt':>6s} "
              f"{'Gewinn':>7s} | {'bw:sp':>6s} {'bw:rk':>5s} {'bw:opt%':>7s} | "
              f"{'v2:sp':>6s} {'v2:opt%':>7s}")
    print(header)
    print("-" * len(header))

    sp_bw, sp_v2 = [], []
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if meas is None:
            print(f"{p['name']:12s} (keine CSV - noch nicht gemessen)")
            continue

        batch = batch_of(p["einsum"], p["shapes"])
        cands, _ = enumerate_candidates(p["einsum"], p["shapes"])
        kept, _ = prune(cands, GB10)

        ok = [r for r in meas.values() if r["ok"]]
        if not ok:
            print(f"{p['name']:12s} {p['regime'][:22]:22s}  (alle Messungen fehlgeschlagen)")
            continue
        best = max(ok, key=lambda r: r["tflops"])["tflops"]
        deflt = meas.get(DEFAULT_SIG)
        deflt_t = deflt["tflops"] if deflt and deflt["ok"] else float("nan")
        gain = best / deflt_t if deflt_t == deflt_t else float("nan")

        bw = model_eval(kept, meas, batch, "bw")
        v2 = model_eval(kept, meas, batch, "bw", reg_clean=True)
        if bw:
            sp_bw.append(bw["spearman"])
        if v2:
            sp_v2.append(v2["spearman"])

        print(f"{p['name']:12s} {p['regime'][:22]:22s} {best:6.1f} {deflt_t:6.1f} "
              f"{gain:6.3f}x | {bw['spearman']:+6.2f} {bw['winner_rank']:5d} "
              f"{100*bw['opt_frac']:6.1f}% | {v2['spearman']:+6.2f} "
              f"{100*v2['opt_frac']:6.1f}%")

    print()
    print("Legende: best/deflt = beste gemessene vs. Default-Config (TFLOPS); "
          "Gewinn = best/default")
    print("         bw:sp = Spearman(Modell, Messung); bw:rk = Modellrang der "
          "Mess-Siegerin; bw:opt% = wie viel % vom Optimum man bekaeme, wenn man")
    print("         dem Modell-#1 vertraut. v2 = bw mit Register-Filter (acc<=0.4*reg).")
    if sp_bw:
        print(f"\nMittlerer Spearman: bw {sum(sp_bw)/len(sp_bw):+.2f}, "
              f"v2 {sum(sp_v2)/len(sp_v2):+.2f}")

    top7_vs_hand(K=7)


def top7_vs_hand(K=7):
    """Praktischer Tuner-Modus: nur die Modell-Top-k messen, die schnellste davon
    nehmen, und gegen den Handkernel (Default-Config) sowie das absolute Optimum
    halten."""
    print(f"\n=== Tuner-Modus: Modell-Top-{K} messen, beste nehmen ===")
    hdr = (f"{'Shape':10s} {'Hand':>6s} {'top'+str(K):>6s} {'absBest':>8s} | "
           f"{'topK/Hand':>9s} {'topK/Best':>9s}")
    print(hdr)
    print("-" * len(hdr))
    gains, fracs = [], []
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if not meas:
            continue
        ok = {s: m["tflops"] for s, m in meas.items() if m["ok"]}
        if not ok:
            continue
        absbest = max(ok.values())
        hand = meas.get(DEFAULT_SIG)
        hand = hand["tflops"] if hand and hand["ok"] else float("nan")

        batch = batch_of(p["einsum"], p["shapes"])
        cands, _ = enumerate_candidates(p["einsum"], p["shapes"])
        kept, _ = prune(cands, GB10)
        pool = [c for c in kept
                if estimate_acc_registers(c) <= REG_FRACTION_V2 * GB10.regs_per_block
                and sig(c) in ok]
        top = [sig(c) for c, _ in rank(pool, GB10, batch=batch, model="bw")][:K]
        pick = max(ok[s] for s in top)
        gains.append(pick / hand)
        fracs.append(pick / absbest)
        print(f"{p['name']:10s} {hand:6.1f} {pick:6.1f} {absbest:8.1f} | "
              f"{pick/hand:8.3f}x {100*pick/absbest:8.1f}%")
    if gains:
        print(f"\nSchnitt: top-{K} vs Hand {sum(gains)/len(gains):.3f}x, "
              f"erreicht {100*sum(fracs)/len(fracs):.1f}% des Optimums")


if __name__ == "__main__":
    main()
