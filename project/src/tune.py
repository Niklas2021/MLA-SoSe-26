"""M3 - Vollmessung + Modell-Evaluation.

Das ist das Herzstueck der Auswertung. Ablauf:
  1. Kandidaten enumerieren + prunen (KEIN dedup -> wir messen fair alles).
  2. jeden Kandidaten kompilieren, auf Korrektheit pruefen, mit do_bench messen.
     Compile-Fehler werden abgefangen und als "failed" protokolliert.
  3. gemessene Rangliste = Ground Truth.
  4. die zwei Modelle (bw / bw_occ) dagegen halten: auf welchem Modell-Rang
     steht die gemessen beste Config? recall@k? Spearman-Korrelation?

Ergebnis: results/tune_<name>.csv (alle Configs) + results/tune_<name>.log.

Auf der GB10 ausfuehren:  python tune.py   (aus project/src/)
"""

import datetime
import os

import torch
import triton.testing

from autotuner.search import enumerate_candidates, prune, rank, _classify_einsum
from autotuner.kernels import run_candidate

try:
    from autotuner.device_props import get_device_properties
    DEV = get_device_properties()
except Exception:
    from autotuner.device_props import GB10 as DEV


# ---- Problem (default: A05 batched matmul) --------------------------------
NAME = "a05"
EINSUM = "cmk, ckn -> cmn"
SHAPES = [(4, 4096, 4096), (4, 4096, 4096)]

# do_bench-Settings fuer den Voll-Sweep: bewusst moderat, das reicht fuer eine
# stabile Rangfolge und haelt den Lauf bei ~10-15 min statt > 1 h.
WARMUP = 50
REP = 300


def signature(cand):
    return (cand.variant, cand.m_prim, cand.n_prim, cand.k_prim, cand.m_l2, cand.n_l2)


def problem_flops_and_batch(einsum, shapes):
    all_dims, m, n, k, batch_chars = _classify_einsum(einsum)
    size_of = {}
    lhs = einsum.replace(" ", "").split("->")[0]
    for tstr, shp in zip(lhs.split(","), shapes):
        for c, s in zip(tstr, shp):
            size_of[c] = s
    batch = 1
    for c in batch_chars:
        batch *= size_of[c]
    flops = 2.0 * batch * size_of[m] * size_of[n] * size_of[k]
    return flops, batch


# ---- kleine Statistik-Helfer (kein scipy) ---------------------------------

def rankdata(values):
    """Durchschnittsraenge (1-basiert), Ties gemittelt."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def spearman(xs, ys):
    return pearson(rankdata(xs), rankdata(ys))


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"=== M3 Tuning + Evaluation ({NAME}) ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if not torch.cuda.is_available():
        log("FEHLER: keine CUDA-GPU.")
        _write(lines, NAME)
        return
    log(f"GPU: {DEV.gpu_name}   Einsum: {EINSUM}   Shapes: {SHAPES}")

    flops, batch = problem_flops_and_batch(EINSUM, SHAPES)

    cands, skipped = enumerate_candidates(EINSUM, SHAPES)
    kept, rejected = prune(cands, DEV)
    log(f"enumeriert {len(cands)}, geprunt -> {len(kept)} zu messen "
        f"({len(rejected)} vorab verworfen)")
    log(f"do_bench: warmup={WARMUP}, rep={REP}")
    log("")

    # Referenz fuer Korrektheit
    torch.manual_seed(0)
    A = torch.randn(SHAPES[0], dtype=torch.float16, device="cuda")
    B = torch.randn(SHAPES[1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(EINSUM.replace(" ", ""), A.float(), B.float()).half()

    # --- alle Kandidaten messen ---
    results = {}   # signature -> dict
    for idx, cand in enumerate(kept):
        sig = signature(cand)
        row = {"cand": cand, "ok": False, "ms": float("inf"),
               "tflops": 0.0, "note": ""}
        try:
            out = run_candidate(cand, A, B)
            torch.cuda.synchronize()
            ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-1)
            if not ok:
                row["note"] = "incorrect"
            else:
                ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                             warmup=WARMUP, rep=REP)
                row.update(ok=True, ms=ms, tflops=flops / (ms * 1e-3) / 1e12)
        except Exception as e:
            row["note"] = f"failed: {type(e).__name__}"
        results[sig] = row

        if (idx + 1) % 25 == 0:
            log(f"  ... {idx+1}/{len(kept)} gemessen")

    # --- Ground Truth: nach TFLOPS sortiert ---
    good = [r for r in results.values() if r["ok"]]
    good.sort(key=lambda r: r["ms"])
    n_fail = len(kept) - len(good)
    log("")
    log(f"gemessen: {len(good)} ok, {n_fail} fehlgeschlagen/inkorrekt")
    if not good:
        _write(lines, NAME)
        return

    log("")
    log("--- gemessene Top-10 (Ground Truth) ---")
    for i, r in enumerate(good[:10]):
        log(f"  #{i+1:2d}  {r['ms']:7.3f} ms  {r['tflops']:6.2f} TFLOPS  | {r['cand'].label()}")
    best_sig = signature(good[0]["cand"])
    log(f"  (A05 Hand-L2 Referenz: 66.10 TFLOPS)")

    # --- Modelle dagegen halten ---
    for model in ("bw", "bw_occ"):
        ranked = rank(kept, DEV, batch=batch, model=model)
        model_order = [signature(c) for c, _ in ranked]
        pos = model_order.index(best_sig) + 1
        log("")
        log(f"--- Modell '{model}' ---")
        log(f"gemessen beste Config steht im Modell auf Rang #{pos} von {len(model_order)}")
        for k in (1, 5, 10, 20):
            topk_model = set(model_order[:k])
            topk_meas = {signature(r["cand"]) for r in good[:k]}
            hit = best_sig in topk_model
            recall = len(topk_model & topk_meas) / k
            log(f"  k={k:2d}:  beste∈Top-k? {'ja ' if hit else 'nein'}   "
                f"recall@k (Schnittmenge) = {recall:.2f}")

        # Spearman ueber alle korrekt gemessenen
        paired = [(m, results[s]["ms"]) for s, m in
                  ((signature(c), mm) for c, mm in ranked) if results[s]["ok"]]
        est = [m["est_ms_occ" if model == "bw_occ" else "est_ms"] for m, _ in paired]
        meas = [ms for _, ms in paired]
        log(f"  Spearman(Modell, Messung) = {spearman(est, meas):+.3f}")

    _write_csv(results, NAME, flops)
    _write(lines, NAME)


def _results_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d


def _write_csv(results, name, flops):
    path = os.path.abspath(os.path.join(_results_dir(), f"tune_{name}.csv"))
    with open(path, "w") as f:
        f.write("variant,m_prim,n_prim,k_prim,m_l2,n_l2,ok,ms,tflops,note\n")
        for r in results.values():
            c = r["cand"]
            f.write(f"{c.variant},{c.m_prim},{c.n_prim},{c.k_prim},{c.m_l2},"
                    f"{c.n_l2},{int(r['ok'])},{r['ms']:.5f},{r['tflops']:.3f},"
                    f"{r['note']}\n")
    print(f"[CSV: {path}]")


def _write(lines, name):
    path = os.path.abspath(os.path.join(_results_dir(), f"tune_{name}.log"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Log: {path}]")


if __name__ == "__main__":
    main()
