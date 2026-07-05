# Misst fuer jede Shape in problems.py alle geprunten Configs und schreibt
# results/tune_<name>.csv. Auswertung dann mit analyze_tune.py (ohne GPU).
# Auf der GB10:  python tune.py            (alle Shapes)
#                python tune.py a06        (nur Namen, die mit a06 anfangen)
import datetime
import os
import sys
import time

import torch
import triton.testing

from autotuner.search import enumerate_candidates, prune
from autotuner.einsum_parser import parse_einsum
from autotuner.kernels import run_candidate
from problems import PROBLEMS, DEFAULT_CONFIG

try:
    from autotuner.device_props import get_device_properties
    DEV = get_device_properties()
except Exception:
    from autotuner.device_props import GB10 as DEV

WARMUP = 50    # ms-Budget fuer do_bench
REP = 200


def sig(c):
    return (c.variant, c.m_prim, c.n_prim, c.k_prim, c.m_l2, c.n_l2)


def flops_and_batch(einsum, shapes):
    # 2 * (Produkt aller Output-Dims) * (Produkt aller K-Dims). Gilt auch fuer
    # A06 mit mehreren M/N/K -- die alte batch*m*n*k-Formel unterschlaegt a,c,b,s.
    e = parse_einsum(einsum, shapes)
    out_vol = 1
    for c in e.out:
        out_vol *= e.size_of[c]
    k_vol = 1
    for c in e.k_chars:
        k_vol *= e.size_of[c]
    batch = 1
    for c in e.batch_chars:
        batch *= e.size_of[c]
    return 2.0 * out_vol * k_vol, batch


def sweep_problem(problem, log):
    einsum = problem["einsum"]
    shapes = problem["shapes"]
    flops, batch = flops_and_batch(einsum, shapes)

    cands, _ = enumerate_candidates(einsum, shapes)
    kept, rejected = prune(cands, DEV)
    log(f"--- {problem['name']}  ({problem['regime']}) ---")
    log(f"einsum {einsum}  shapes {shapes}")
    log(f"enumeriert {len(cands)} -> {len(kept)} zu messen ({len(rejected)} verworfen)")

    torch.manual_seed(0)
    A = torch.randn(shapes[0], dtype=torch.float16, device="cuda")
    B = torch.randn(shapes[1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(einsum.replace(" ", ""), A.float(), B.float()).half()

    results = {}
    t_start = time.perf_counter()
    for cand in kept:
        row = {"cand": cand, "ok": False, "ms": float("inf"), "tflops": 0.0, "note": ""}
        try:
            out = run_candidate(cand, A, B)
            torch.cuda.synchronize()
            if not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                row["note"] = "incorrect"
            else:
                ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                             warmup=WARMUP, rep=REP)
                row.update(ok=True, ms=ms, tflops=flops / (ms * 1e-3) / 1e12)
        except Exception as e:
            row["note"] = f"failed: {type(e).__name__}"
        results[sig(cand)] = row
    wall = time.perf_counter() - t_start

    _write_csv(results, problem["name"])

    good = sorted((r for r in results.values() if r["ok"]), key=lambda r: r["ms"])
    n_fail = len(kept) - len(good)
    if good:
        best = good[0]
        default = results.get(tuple(DEFAULT_CONFIG[k] for k in
                              ("variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2")))
        gain = ""
        if default and default["ok"]:
            gain = f", Default {default['tflops']:.1f} -> Gewinn {best['tflops']/default['tflops']:.3f}x"
        log(f"best {best['tflops']:.2f} TFLOPS | {best['cand'].label()}{gain}")
    log(f"{len(good)} ok, {n_fail} fehlgeschlagen/inkorrekt   Sweep-Wall-Clock {wall:.1f} s")
    log("")
    return wall


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=== M3 Multi-Shape-Sweep ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if not torch.cuda.is_available():
        log("FEHLER: keine CUDA-GPU.")
        _write_log(lines)
        return
    log(f"GPU: {DEV.gpu_name}   do_bench warmup={WARMUP} rep={REP}")

    # optionale Namensfilter als Argumente (Praefix), sonst alle. Bei Filter eine
    # eigene Log (study_<keys>.log), damit ein Teil-Lauf die volle Studie nicht ueberschreibt.
    selected = PROBLEMS
    log_name = "study.log"
    if len(sys.argv) > 1:
        keys = sys.argv[1:]
        selected = [p for p in PROBLEMS if any(p["name"].startswith(k) for k in keys)]
        log_name = "study_" + "_".join(keys) + ".log"
        log(f"Filter {keys} -> {[p['name'] for p in selected]}")
    log("")

    total = 0.0
    for problem in selected:
        try:
            total += sweep_problem(problem, log)
        except Exception as e:
            log(f"--- {problem['name']}: ABGEBROCHEN ({type(e).__name__}: {e}) ---")
            log("")
    log(f"Gesamt-Wall-Clock aller Sweeps: {total/60:.1f} min")
    _write_log(lines, log_name)


def _results_dir():
    d = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(d, exist_ok=True)
    return d


def _write_csv(results, name):
    path = os.path.abspath(os.path.join(_results_dir(), f"tune_{name}.csv"))
    with open(path, "w") as f:
        f.write("variant,m_prim,n_prim,k_prim,m_l2,n_l2,ok,ms,tflops,note\n")
        for r in results.values():
            c = r["cand"]
            f.write(f"{c.variant},{c.m_prim},{c.n_prim},{c.k_prim},{c.m_l2},"
                    f"{c.n_l2},{int(r['ok'])},{r['ms']:.5f},{r['tflops']:.3f},{r['note']}\n")
    print(f"[CSV: {path}]")


def _write_log(lines, name="study.log"):
    path = os.path.abspath(os.path.join(_results_dir(), name))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Log: {path}]")


if __name__ == "__main__":
    main()
