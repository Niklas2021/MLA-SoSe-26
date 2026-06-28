# Auswertung des Multi-Shape-Sweeps, laeuft lokal aus den CSVs (ohne GPU).
# Modell-Prognose vs. Messung pro Shape + Tuner-Modus (nur Top-k messen).
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
        meas[key] = {"ms": float(r["ms"]), "tflops": float(r["tflops"]), "ok": int(r["ok"])}
    return meas


def model_eval(kept, meas, batch, model, reg_clean=False):
    cands = kept
    if reg_clean:
        cands = [c for c in kept if estimate_acc_registers(c) <= REG_FRACTION_V2 * GB10.regs_per_block]
    cands = [c for c in cands if sig(c) in meas and meas[sig(c)]["ok"]]
    if len(cands) < 2:
        return None
    ranked = rank(cands, GB10, batch=batch, model=model)
    order = [sig(c) for c, _ in ranked]
    key = "est_ms_occ" if model == "bw_occ" else "est_ms"
    est = [m[key] for _, m in ranked]
    ms = [meas[s]["ms"] for s in order]
    best_sig = min(order, key=lambda s: meas[s]["ms"])
    return {"spearman": spearman(est, ms),
            "winner_rank": order.index(best_sig) + 1,
            "opt_frac": meas[order[0]]["tflops"] / meas[best_sig]["tflops"],
            "n": len(cands)}


def main():
    header = (f"{'Shape':12s} {'Regime':22s} {'best':>6s} {'deflt':>6s} {'Gewinn':>7s} | "
              f"{'bw:sp':>6s} {'bw:rk':>5s} {'bw:opt%':>7s} | {'v2:sp':>6s} {'v2:opt%':>7s}")
    print(header)
    print("-" * len(header))

    sp_bw, sp_v2 = [], []
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if meas is None:
            print(f"{p['name']:12s} (keine CSV)")
            continue
        ok = [r for r in meas.values() if r["ok"]]
        if not ok:
            print(f"{p['name']:12s} {p['regime'][:22]:22s}  (alle fehlgeschlagen)")
            continue
        best = max(ok, key=lambda r: r["tflops"])["tflops"]
        deflt = meas.get(DEFAULT_SIG)
        deflt_t = deflt["tflops"] if deflt and deflt["ok"] else float("nan")
        gain = best / deflt_t if deflt_t == deflt_t else float("nan")

        batch = batch_of(p["einsum"], p["shapes"])
        cands, _ = enumerate_candidates(p["einsum"], p["shapes"])
        kept, _ = prune(cands, GB10)
        bw = model_eval(kept, meas, batch, "bw")
        v2 = model_eval(kept, meas, batch, "bw", reg_clean=True)
        sp_bw.append(bw["spearman"])
        sp_v2.append(v2["spearman"])

        print(f"{p['name']:12s} {p['regime'][:22]:22s} {best:6.1f} {deflt_t:6.1f} {gain:6.3f}x | "
              f"{bw['spearman']:+6.2f} {bw['winner_rank']:5d} {100*bw['opt_frac']:6.1f}% | "
              f"{v2['spearman']:+6.2f} {100*v2['opt_frac']:6.1f}%")

    if sp_bw:
        print(f"\nMittlerer Spearman: bw {sum(sp_bw)/len(sp_bw):+.2f}, v2 {sum(sp_v2)/len(sp_v2):+.2f}")
    top7_vs_hand(K=7)


def top7_vs_hand(K=7):
    # nur die Modell-Top-K messen, schnellste nehmen, gegen Hand + abs. Optimum
    print(f"\n=== Tuner-Modus: Modell-Top-{K} messen, beste nehmen ===")
    hdr = f"{'Shape':10s} {'Hand':>6s} {'top'+str(K):>6s} {'absBest':>8s} | {'topK/Hand':>9s} {'topK/Best':>9s}"
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
                if estimate_acc_registers(c) <= REG_FRACTION_V2 * GB10.regs_per_block and sig(c) in ok]
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
