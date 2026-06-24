"""Server-Test fuer M2 - beantwortet die offenen Fragen, BEVOR wir die
Tuning-Maschinerie bauen:

  1. Kompiliert der generische ct.Constant-Kernel ueberhaupt? (Falls nein ->
     wir muessen doch zu String-Templates greifen.)
  2. Stimmt das Ergebnis gegen torch.einsum?
  3. Wie lange dauert EIN Compile (erster Launch) vs. ein gecachter Launch?
     -> hochgerechnet auf 342 / 186 Configs sehen wir, ob wir alle messen
        koennen oder Top-k-Ranking brauchen.
  4. Spezialisiert ct.Constant pro Wert? (zweite Config mit anderem M_PRIM ->
     wenn die auch erst kompiliert werden muss, ist die Antwort ja.)

Ausfuehren auf der GB10:   python measure_compile.py   (aus project/src/)
"""

import datetime
import os
import time

import torch
import triton.testing

from autotuner.kernels import run_variant_a


# A05-Problemgroesse
C_SIZE, M_SIZE, N_SIZE, K_SIZE = 4, 4096, 4096, 4096
FLOPS = 2.0 * C_SIZE * M_SIZE * N_SIZE * K_SIZE


def timed(fn):
    """fuehrt fn aus und gibt (ergebnis, sekunden) zurueck, inkl. sync."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=== M2 Compile-Test ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    if not torch.cuda.is_available():
        log("FEHLER: keine CUDA-GPU sichtbar.")
        _write(lines)
        return
    log(f"GPU: {torch.cuda.get_device_properties(0).name}")
    log("")

    torch.manual_seed(0)
    A = torch.randn((C_SIZE, M_SIZE, K_SIZE), dtype=torch.float16, device="cuda")
    B = torch.randn((C_SIZE, K_SIZE, N_SIZE), dtype=torch.float16, device="cuda")
    ref = torch.einsum("cmk,ckn->cmn", A.float(), B.float()).half()

    # --- Config 1: die A05-Hand-Config ---
    cfg1 = dict(m_prim=128, n_prim=128, k_prim=64, m_l2=8, n_l2=8)
    log(f"--- Config 1 (A05-Hand): {cfg1} ---")

    out1, t_compile = timed(lambda: run_variant_a(A, B, **cfg1))
    _,    t_cached  = timed(lambda: run_variant_a(A, B, **cfg1))

    err = (out1.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out1, ref, rtol=1e-2, atol=1e-1)
    log(f"Korrektheit:        max_err={err:.4f}  allclose={ok}")
    log(f"erster Launch:      {t_compile*1000:.1f} ms  (inkl. Compile)")
    log(f"gecachter Launch:   {t_cached*1000:.1f} ms")
    log(f"=> reine Compile-Zeit ~ {(t_compile - t_cached)*1000:.1f} ms")

    ms = triton.testing.do_bench(lambda: run_variant_a(A, B, **cfg1),
                                 warmup=200, rep=2000)
    tflops = FLOPS / (ms * 1e-3) / 1e12
    log(f"do_bench:           {ms:.3f} ms   {tflops:.2f} TFLOPS   (A05-Baseline: 66.10)")
    log("")

    # --- Config 2: anderes M_PRIM -> testet ct.Constant-Spezialisierung ---
    cfg2 = dict(m_prim=64, n_prim=128, k_prim=64, m_l2=8, n_l2=8)
    log(f"--- Config 2 (anderes M_PRIM): {cfg2} ---")
    out2, t_compile2 = timed(lambda: run_variant_a(A, B, **cfg2))
    _,    t_cached2  = timed(lambda: run_variant_a(A, B, **cfg2))
    ok2 = torch.allclose(out2, ref, rtol=1e-2, atol=1e-1)
    log(f"Korrektheit:        allclose={ok2}")
    log(f"erster Launch:      {t_compile2*1000:.1f} ms")
    log(f"gecachter Launch:   {t_cached2*1000:.1f} ms")
    if t_compile2 > 3 * t_cached2:
        log("=> Config 2 musste neu kompilieren -> ct.Constant spezialisiert PRO WERT (gut).")
    else:
        log("=> Config 2 war direkt schnell -> KEINE Spezialisierung pro Wert?! genauer ansehen.")
    log("")

    # --- Hochrechnung Tuning-Budget ---
    log("--- Hochrechnung (grob, nur Compile) ---")
    sec = max(t_compile - t_cached, t_compile2 - t_cached2)
    log(f"angenommene Compile-Zeit/Config: ~{sec:.1f} s")
    log(f"alle 342 Kandidaten:  ~{342*sec/60:.1f} min")
    log(f"nach dedup (186):     ~{186*sec/60:.1f} min")
    log("(wenn das im Minutenbereich liegt, brauchen wir kein aggressives Ranking)")

    _write(lines)


def _write(lines):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(results_dir, "measure_compile.log"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Log geschrieben in: {path}]")


if __name__ == "__main__":
    main()
