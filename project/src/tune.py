# Misst fuer jede Shape in problems.py alle geprunten Configs und schreibt
# results/tune_<name>.csv. Auswertung dann mit analyze_tune.py (ohne GPU).
# Auf der GB10:  python tune.py            (alle Shapes)
#                python tune.py a06        (nur Namen, die mit a06 anfangen)
import datetime
import sys
import time

import torch
import triton.testing

import results_io
from autotuner.search import enumerate_candidates, prune
from autotuner.einsum_parser import parse_einsum
from autotuner.kernels import run_candidate
from problems import PROBLEMS, DEFAULT_CONFIG, baseline_for

from autotuner.device_props import resolve_device_properties

DEV = resolve_device_properties()

WARMUP = 50    # ms-Budget fuer do_bench
REP = 200


def sig(c):
    return (c.variant, c.m_prim, c.n_prim, c.k_prim, c.m_l2, c.n_l2, c.order)


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
    es = einsum.replace(" ", "")
    ref = torch.einsum(es, A.float(), B.float()).half()

    # externe Referenz: torch.einsum in fp16 ("einfach die Library nehmen")
    torch_ms = triton.testing.do_bench(lambda: torch.einsum(es, A, B), warmup=WARMUP, rep=REP)
    torch_tflops = flops / (torch_ms * 1e-3) / 1e12

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
        keys = ("variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2", "order")
        # zwei Baselines: die faire (feste Config fuer DIESE GPU) und die naiv
        # uebernommene A05-Wahl. Der Unterschied ist gross und gehoert benannt.
        base = results.get(tuple(baseline_for(DEV.gpu_name)[k] for k in keys))
        default = results.get(tuple(DEFAULT_CONFIG[k] for k in keys))
        gain = ""
        if base and base["ok"]:
            gain = f", Baseline {base['tflops']:.1f} -> Gewinn {best['tflops']/base['tflops']:.3f}x"
        if default and default["ok"] and default is not base:
            gain += f" (A05-Default {default['tflops']:.1f} -> {best['tflops']/default['tflops']:.3f}x)"
        log(f"best {best['tflops']:.2f} TFLOPS | {best['cand'].label()}{gain}")
        log(f"torch.einsum {torch_tflops:.2f} TFLOPS -> Tuner {best['tflops']/torch_tflops:.2f}x")
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


def _write_csv(results, name):
    rows = []
    for r in results.values():
        c = r["cand"]
        rows.append([c.variant, c.m_prim, c.n_prim, c.k_prim, c.m_l2, c.n_l2,
                     c.order, int(r["ok"]), f"{r['ms']:.5f}", f"{r['tflops']:.3f}",
                     r["note"]])
    results_io.write_csv(f"tune_{name}.csv",
                         ["variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2",
                          "order", "ok", "ms", "tflops", "note"], rows)


def _write_log(lines, name="study.log"):
    results_io.write_log(lines, name)


if __name__ == "__main__":
    main()
