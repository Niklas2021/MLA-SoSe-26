# Misst den A06-HANDKERNEL (run_ring_a) auf allen acht A06-Shapes aus problems.py.
# Hintergrund: von Hand getunt wurde in Assignment 06 nur die Referenz-Shape
# (49.84 TFLOPS). Fuer die anderen sieben Regime gibt es keinen handgeschriebenen
# Kernel. Dieses Skript legt den Handkernel als feste Config auf jede Shape und
# misst ihn mit derselben Methodik wie tune.py/study.log (do_bench 50/200), damit
# die Zahlen 1:1 mit den Default/Tuner/torch-Balken vergleichbar sind.
#
# Zwei Spalten, weil "Handkernel auf fremder Shape" zweideutig ist:
#   hand_fixed  = das EXAKTE Referenz-Tiling (128/128/64, m_l2=2, n_l2=3, Var. A),
#                 also der eine Assignment-Kernel, stur auf jede Shape gelegt.
#                 -> auf a06 muss ~49.84 rauskommen (eingebauter Sanity-Check).
#   hand_adapt  = per Shape neu von Hand gewaehlt: 128/128-prim, k_prim = ganze
#                 p-Reduktion (auf 16 aufgerundet), m_l2/n_l2 = groesster Teiler
#                 der Blockzahl <= 4. So wie man ohne Tuner sinnvoll tilen wuerde.
#
# Auf der GB10:  python measure_hand_a06.py
# Output: results/hand_a06.csv + Tabelle auf stdout.
import os

import torch
import triton.testing

from autotuner.kernels import run_ring_a
from autotuner.einsum_parser import parse_einsum
from problems import PROBLEMS

WARMUP = 50    # identisch zu tune.py -> gleiche Methodik wie study.log
REP = 200


def ceildiv(a, b):
    return (a + b - 1) // b


def flops(einsum, shapes):
    # 2 * (Produkt Output-Dims) * (Produkt K-Dims), wie tune.flops_and_batch
    e = parse_einsum(einsum, shapes)
    out_vol = 1
    for c in e.out:
        out_vol *= e.size_of[c]
    k_vol = 1
    for c in e.k_chars:
        k_vol *= e.size_of[c]
    return 2.0 * out_vol * k_vol


def round16(v):
    return max(16, ((v + 15) // 16) * 16)


def largest_div_le(n, cap=4):
    # groesster Teiler von n, der <= cap ist (mind. 1)
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def hand_fixed_cfg(shapes):
    # der eine Assignment-Handkernel: 128/128/64, m_l2=2, n_l2=3
    return dict(m_prim=128, n_prim=128, k_prim=64, m_l2=2, n_l2=3)


def hand_adapt_cfg(shapes):
    # A=(a,c,s,p,x), B=(b,s,p,y). x=prim_m, y=prim_n, p=prim_k, s=SEQ.
    _, _, _, size_p, size_x = shapes[0]
    size_y = shapes[1][3]
    nx = ceildiv(size_x, 128)
    ny = ceildiv(size_y, 128)
    return dict(m_prim=128, n_prim=128, k_prim=round16(size_p),
                m_l2=largest_div_le(nx), n_l2=largest_div_le(ny))


def measure(problem, cfg):
    einsum = problem["einsum"]
    shapes = problem["shapes"]
    es = einsum.replace(" ", "")

    torch.manual_seed(0)   # gleiche Inputs wie tune.py
    A = torch.randn(shapes[0], dtype=torch.float16, device="cuda")
    B = torch.randn(shapes[1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(es, A.float(), B.float()).half()

    out = run_ring_a(A, B, cfg["m_prim"], cfg["n_prim"], cfg["k_prim"],
                     cfg["m_l2"], cfg["n_l2"])
    torch.cuda.synchronize()
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-1)
    if not ok:
        return dict(ok=False, ms=float("inf"), tflops=0.0)

    ms = triton.testing.do_bench(
        lambda: run_ring_a(A, B, cfg["m_prim"], cfg["n_prim"], cfg["k_prim"],
                           cfg["m_l2"], cfg["n_l2"]),
        warmup=WARMUP, rep=REP)
    return dict(ok=True, ms=ms, tflops=flops(einsum, shapes) / (ms * 1e-3) / 1e12)


def safe_measure(problem, cfg):
    # ein kaputtes Config (Compile-Fehler, ungueltige K-Kachel) soll den Lauf
    # nicht abbrechen -> als nicht-ok zurueckgeben
    try:
        return measure(problem, cfg)
    except Exception as e:
        print(f"  ! {problem['name']} {cfg}: {type(e).__name__}: {e}")
        return dict(ok=False, ms=float("inf"), tflops=0.0)


def main():
    a06 = [p for p in PROBLEMS if p["name"].startswith("a06")]
    if not torch.cuda.is_available():
        print("FEHLER: keine CUDA-GPU.")
        return

    rows = []
    print(f"A06-Handkernel-Messung  (do_bench warmup={WARMUP} rep={REP})\n")
    hdr = f"{'shape':<14} {'fixed 2x3':>12} {'cfg_fixed':>16}   {'adapt':>10} {'cfg_adapt':>16}"
    print(hdr)
    print("-" * len(hdr))
    for p in a06:
        cf = hand_fixed_cfg(p["shapes"])
        ca = hand_adapt_cfg(p["shapes"])
        rf = safe_measure(p, cf)
        ra = safe_measure(p, ca)
        sf = f"{cf['m_prim']}/{cf['n_prim']}/{cf['k_prim']} {cf['m_l2']}x{cf['n_l2']}"
        sa = f"{ca['m_prim']}/{ca['n_prim']}/{ca['k_prim']} {ca['m_l2']}x{ca['n_l2']}"
        vf = f"{rf['tflops']:.2f}" if rf["ok"] else "INCORRECT"
        va = f"{ra['tflops']:.2f}" if ra["ok"] else "INCORRECT"
        tag = "  <- erwartet ~49.84" if p["name"] == "a06" else ""
        print(f"{p['name']:<14} {vf:>12} {sf:>16}   {va:>10} {sa:>16}{tag}")
        rows.append((p["name"], rf, cf, ra, ca))

    d = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(d, exist_ok=True)
    path = os.path.abspath(os.path.join(d, "hand_a06.csv"))
    with open(path, "w") as f:
        f.write("name,kind,m_prim,n_prim,k_prim,m_l2,n_l2,ok,ms,tflops\n")
        for name, rf, cf, ra, ca in rows:
            for kind, r, c in (("fixed", rf, cf), ("adapt", ra, ca)):
                f.write(f"{name},{kind},{c['m_prim']},{c['n_prim']},{c['k_prim']},"
                        f"{c['m_l2']},{c['n_l2']},{int(r['ok'])},{r['ms']:.5f},"
                        f"{r['tflops']:.3f}\n")
    print(f"\n[CSV: {path}]")


if __name__ == "__main__":
    main()
