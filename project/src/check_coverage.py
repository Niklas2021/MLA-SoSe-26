# Prueft, welche Einsum-Strings die Kernel wirklich abdecken (M5.3).
# Ohne GPU:  python check_coverage.py            -- nur die Layout-Entscheidung
# Auf der GB10:  python check_coverage.py --run  -- zusaetzlich tunen + gegen torch pruefen
import datetime
import sys

import results_io
from autotuner.einsum_parser import parse_einsum
from autotuner.layout import plan_layout, UnsupportedLayout
from problems import COVERAGE


def es_of(p):
    return p["einsum"].replace(" ", "")


def main(run=False):
    lines, rows = [], []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=== M5.3 Einsum-Abdeckung ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if run:
        import torch
        from autotuner import strategies
        from autotuner.kernels import run_candidate
        from autotune import model_ranked, _batch_and_flops
        from autotuner.device_props import resolve_device_properties
        dev = resolve_device_properties()
        if not torch.cuda.is_available():
            print("keine CUDA-GPU -- laeuft nur ohne --run")
            return
        log(f"GPU: {dev.gpu_name}")
    log()

    n_ok = n_bad = 0
    for p in COVERAGE:
        e = parse_einsum(p["einsum"], p["shapes"])
        try:
            plan_layout(e)
            supported, why = True, ""
        except UnsupportedLayout as ex:
            supported, why = False, str(ex)

        if supported != p["supported"]:
            log(f"FEHLER {p['name']:14s} {p['einsum']:22s} "
                f"erwartet supported={p['supported']}, ist {supported}")
            rows.append([p["name"], es_of(p), int(supported), int(p["supported"]),
                         "", "", "abweichung", p["note"]])
            n_bad += 1
            continue

        if not supported:
            log(f"  --   {p['name']:14s} {p['einsum']:22s} abgelehnt ({p['note']})")
            rows.append([p["name"], es_of(p), 0, 0, "", "", "abgelehnt", p["note"]])
            n_ok += 1
            continue

        if not run:
            log(f"  ok   {p['name']:14s} {p['einsum']:22s} {p['note']}")
            rows.append([p["name"], es_of(p), 1, 1, "", "", "layout_ok", p["note"]])
            n_ok += 1
            continue

        # wirklich rechnen lassen: tunen, dann das Ergebnis gegen torch.einsum pruefen
        es = p["einsum"].replace(" ", "")
        torch.manual_seed(0)
        A = torch.randn(p["shapes"][0], dtype=torch.float16, device="cuda")
        B = torch.randn(p["shapes"][1], dtype=torch.float16, device="cuda")
        ref = torch.einsum(es, A.float(), B.float()).half()
        _, flops = _batch_and_flops(e)

        import triton.testing

        # ersten Fehler festhalten. Sonst steht am Ende nur "lief nicht durch" und man
        # weiss nicht, ob der Kernel nicht kompiliert oder nur falsch gerechnet hat --
        # gerade beim flex-Kernel (Verzweigung ueber ct.Constant) ist das der Unterschied.
        problem = {"err": None}

        def measure(cand):
            try:
                out = run_candidate(cand, A, B)
                torch.cuda.synchronize()
                if out.shape != ref.shape:
                    problem["err"] = problem["err"] or f"Shape {tuple(out.shape)} != {tuple(ref.shape)}"
                    return None
                if not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                    d = (out.float() - ref.float()).abs().max().item()
                    problem["err"] = problem["err"] or f"falsches Ergebnis (max. Abweichung {d:.3g})"
                    return None
                ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                             warmup=25, rep=100)
            except Exception as ex:
                problem["err"] = problem["err"] or f"{type(ex).__name__}: {str(ex)[:120]}"
                return None
            return flops / (ms * 1e-3) / 1e12

        ranked = model_ranked(p["einsum"], p["shapes"], dev)
        cand, tf, n = strategies.hybrid(ranked, measure, k=7)
        if cand is None:
            log(f"FEHLER {p['name']:14s} {p['einsum']:22s} keine von {len(ranked)} "
                f"Configs lief korrekt\n         -> {problem['err']}")
            rows.append([p["name"], es, 1, 1, "", "", f"fehler: {problem['err']}", p["note"]])
            n_bad += 1
            continue
        # Gegenprobe: liefert der Gewinner die richtige Output-Shape zurueck?
        out = run_candidate(cand, A, B)
        assert out.shape == ref.shape, f"{p['name']}: Shape {out.shape} != {ref.shape}"
        log(f"  ok   {p['name']:14s} {p['einsum']:22s} {tf:6.1f} TFLOPS "
            f"({n} Messungen, {len(ranked)} Kandidaten)  {p['note']}")
        rows.append([p["name"], es, 1, 1, f"{tf:.3f}", n, "gerechnet", p["note"]])
        n_ok += 1

    log(f"\n{n_ok} von {len(COVERAGE)} wie erwartet, {n_bad} Abweichungen")
    tag = "coverage_run" if run else "coverage"
    results_io.write_csv(f"{tag}.csv",
                         ["name", "einsum", "supported", "expected",
                          "tflops", "n_measured", "status", "note"], rows)
    results_io.write_log(lines, f"{tag}.log")
    return 1 if n_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(run="--run" in sys.argv))
