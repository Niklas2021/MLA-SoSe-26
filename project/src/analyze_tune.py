# Auswertung des Multi-Shape-Sweeps, laeuft lokal aus den CSVs (ohne GPU).
# Modell-Prognose vs. Messung pro Shape (bw / v2 / roofline) + Tuner-Modus (Top-k).
import csv
import os

from autotuner.search import (enumerate_candidates, prune, rank,
                              estimate_acc_registers)
from autotuner.einsum_parser import parse_einsum
from autotuner.device_props import GB10
from autotuner.stats import spearman
from problems import PROBLEMS, DEFAULT_CONFIG

REG_FRACTION_V2 = 0.4
DEFAULT_SIG = tuple(DEFAULT_CONFIG[k] for k in
                    ("variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2", "order"))


def sig(c):
    return (c.variant, c.m_prim, c.n_prim, c.k_prim, c.m_l2, c.n_l2,
            getattr(c, "order", 0))


def batch_of(einsum, shapes):
    e = parse_einsum(einsum, shapes)
    b = 1
    for c in e.batch_chars:
        b *= e.size_of[c]
    return b


RESULTS_DIR = os.environ.get("RESULTS_DIR",
                             os.path.join(os.path.dirname(__file__), "..", "results"))


def load_csv(name):
    path = os.path.join(RESULTS_DIR, f"tune_{name}.csv")
    if not os.path.exists(path):
        return None
    meas = {}
    for r in csv.DictReader(open(path)):
        key = (r["variant"], int(r["m_prim"]), int(r["n_prim"]),
               int(r["k_prim"]), int(r["m_l2"]), int(r["n_l2"]),
               int(r.get("order") or 0))
        meas[key] = {"ms": float(r["ms"]), "tflops": float(r["tflops"]), "ok": int(r["ok"])}
    return meas


def _pool(kept, meas, reg_clean):
    cands = kept
    if reg_clean:
        cands = [c for c in kept if estimate_acc_registers(c) <= REG_FRACTION_V2 * GB10.regs_per_block]
    return [c for c in cands if sig(c) in meas and meas[sig(c)]["ok"]]


def model_eval(kept, meas, batch, model, reg_clean=False, K=7):
    cands = _pool(kept, meas, reg_clean)
    if len(cands) < 2:
        return None
    ranked = rank(cands, GB10, batch=batch, model=model)
    order = [sig(c) for c, _ in ranked]
    key = {"bw_occ": "est_ms_occ", "roofline": "roof_ms"}.get(model, "est_ms")
    est = [m[key] for _, m in ranked]
    ms = [meas[s]["ms"] for s in order]
    best_sig = min(order, key=lambda s: meas[s]["ms"])
    absbest = max(meas[s]["tflops"] for s in order)
    return {"spearman": spearman(est, ms),
            "winner_rank": order.index(best_sig) + 1,
            "topk_frac": max(meas[s]["tflops"] for s in order[:K]) / absbest,
            "bound": ranked[0][1].get("bound", "-"),
            "n": len(cands)}


def main():
    header = (f"{'Shape':12s} {'Regime':20s} {'best':>6s} {'deflt':>6s} {'Gew':>6s} | "
              f"{'bw:sp':>6s} {'v2:sp':>6s} {'roof:sp':>7s} {'roof:rk':>7s} {'bound':>7s}")
    print(header)
    print("-" * len(header))

    sp = {"bw": [], "v2": [], "roof": []}
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if meas is None:
            print(f"{p['name']:12s} (keine CSV)")
            continue
        ok = [r for r in meas.values() if r["ok"]]
        if not ok:
            print(f"{p['name']:12s} {p['regime'][:20]:20s}  (alle fehlgeschlagen)")
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
        rf = model_eval(kept, meas, batch, "roofline")
        sp["bw"].append(bw["spearman"])
        sp["v2"].append(v2["spearman"])
        sp["roof"].append(rf["spearman"])

        print(f"{p['name']:12s} {p['regime'][:20]:20s} {best:6.1f} {deflt_t:6.1f} {gain:5.2f}x | "
              f"{bw['spearman']:+6.2f} {v2['spearman']:+6.2f} {rf['spearman']:+7.2f} "
              f"{rf['winner_rank']:7d} {rf['bound']:>7s}")

    if sp["bw"]:
        print(f"\nMittlerer Spearman: bw {sum(sp['bw'])/len(sp['bw']):+.2f}, "
              f"v2 {sum(sp['v2'])/len(sp['v2']):+.2f}, roofline {sum(sp['roof'])/len(sp['roof']):+.2f}")
    top_k_compare(K=7)


def top_k_compare(K=7):
    # Praxis-Metrik: nur Modell-Top-K messen, schnellste nehmen. v2 (Vorfilter) vs
    # roofline (globaler Ranker). Zeigt: bessere Korrelation != besserer Top-k-Filter.
    print(f"\n=== Modell-Top-{K} messen, schnellste nehmen: v2 vs roofline ===")
    hdr = (f"{'Shape':12s} {'absBest':>8s} | {'v2-top':>7s} {'v2/Best':>8s} | "
           f"{'roof-top':>8s} {'roof/Best':>9s}")
    print(hdr)
    print("-" * len(hdr))
    fr = {"v2": [], "roof": []}
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if not meas:
            continue
        ok = {s: m["tflops"] for s, m in meas.items() if m["ok"]}
        if not ok:
            continue
        absbest = max(ok.values())
        batch = batch_of(p["einsum"], p["shapes"])
        cands, _ = enumerate_candidates(p["einsum"], p["shapes"])
        kept, _ = prune(cands, GB10)
        v2pool = _pool(kept, meas, reg_clean=True)
        rfpool = _pool(kept, meas, reg_clean=False)
        v2top = [sig(c) for c, _ in rank(v2pool, GB10, batch=batch, model="bw")][:K]
        rftop = [sig(c) for c, _ in rank(rfpool, GB10, batch=batch, model="roofline")][:K]
        v2p = max(ok[s] for s in v2top)
        rfp = max(ok[s] for s in rftop)
        fr["v2"].append(v2p / absbest)
        fr["roof"].append(rfp / absbest)
        print(f"{p['name']:12s} {absbest:8.1f} | {v2p:7.1f} {100*v2p/absbest:7.1f}% | "
              f"{rfp:8.1f} {100*rfp/absbest:8.1f}%")
    if fr["v2"]:
        print(f"\nSchnitt Optimum-Ausbeute: v2 {100*sum(fr['v2'])/len(fr['v2']):.1f}%, "
              f"roofline {100*sum(fr['roof'])/len(fr['roof']):.1f}%")


if __name__ == "__main__":
    main()
