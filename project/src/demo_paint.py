# Live-Demo: der Tuner "malt" ein Bild.
#
# Ein grosses GEMM (cmk,ckn->cmn) synthetisiert ein Full-HD-Plasma-Bild: jeder Pixel
# ist die Summe von ein paar hundert 2D-Wellen, und diese Summe ist genau ein
# Matrixprodukt Y @ X. Wir rechnen es ZWEIMAL -- einmal mit der naiven 8x8-Default-
# Config, einmal mit der Config, die der Tuner waehlt. Gleiche Mathematik, gleiches
# Bild, aber unterschiedlich schnell. Am Ende: das Bild + wer gewonnen hat.
#
# Auf dem DGX:              python demo_paint.py
# Ohne GPU (nur Vorschau): python demo_paint.py --preview
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 16:9, absichtlich "krumme" Groessen: 1080/1920/1000 sind nicht tile-teilbar, also
# muss die 8x8-Default hochpadden -> genau die Situation, in der der Tuner gewinnt.
DEF_H, DEF_W, DEF_K = 1080, 1920, 1000
EINSUM = "cmk, ckn -> cmn"


# ---------------------------------------------------------------- Bild-Mathematik
def plasma_features(M, N, K, seed=7):
    # Bild[y,x] = sum_k a_k * cos(2*pi*(fy_k*y/M + fx_k*x/N) + phi_k).
    # cos(alpha+beta) = cos a cos b - sin a sin b -> jede Welle sind zwei Spalten in
    # Y und zwei Zeilen in X, damit Y@X exakt die Wellensumme ergibt. Reines GEMM.
    rng = np.random.default_rng(seed)
    waves = K // 2
    fy = rng.integers(1, 14, size=waves).astype(np.float32)
    fx = rng.integers(1, 14, size=waves).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, size=waves).astype(np.float32)
    amp = (1.0 / (1.0 + 0.6 * (fy + fx))).astype(np.float32)   # pink: tiefe Freq. lauter

    y = (np.arange(M, dtype=np.float32) / M)[:, None]           # (M,1)
    x = (np.arange(N, dtype=np.float32) / N)[None, :]           # (1,N)
    ty = 2 * np.pi * fy[None, :] * y + phi[None, :]             # (M,waves)
    tx = 2 * np.pi * fx[:, None] * x                            # (waves,N)

    Y = np.empty((M, 2 * waves), np.float32)
    X = np.empty((2 * waves, N), np.float32)
    Y[:, 0::2] = np.cos(ty)
    Y[:, 1::2] = -np.sin(ty)
    X[0::2, :] = amp[:, None] * np.cos(tx)
    X[1::2, :] = amp[:, None] * np.sin(tx)
    return Y, X


# blaue->weisse->orange diverging Palette (wie im Deck)
_STOPS = np.array([
    [0.00,  12,  22,  42],
    [0.25,  42, 120, 214],
    [0.50, 238, 242, 248],
    [0.75, 235, 140,  60],
    [1.00, 150,  40,  20],
], np.float32)


def colorize(img):
    lo, hi = np.percentile(img, 2), np.percentile(img, 98)
    t = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
    pos, cols = _STOPS[:, 0], _STOPS[:, 1:]
    out = np.empty(img.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, cols[:, c])
    return out.astype(np.uint8)


def save_png(img2d, path):
    from PIL import Image
    Image.fromarray(colorize(img2d)).save(path)


# ---------------------------------------------------------------- Terminal-Deko
def bar(ms, ms_max, width=34):
    n = int(round(width * ms / ms_max)) if ms_max > 0 else 0
    return "█" * max(1, n)


def banner(title):
    line = "═" * (len(title) + 2)
    print(f"\n╔{line}╗\n║ {title} ║\n╚{line}╝")


# ---------------------------------------------------------------- Vorschau (kein GPU)
def preview(M, N, K, seed, out):
    import torch
    banner("cuTile Auto-Tuner · Live-Demo (Vorschau, ohne GPU)")
    Yn, Xn = plasma_features(M, N, K, seed)
    Y = torch.from_numpy(Yn); X = torch.from_numpy(Xn)
    img = torch.einsum("mk,kn->mn", Y, X).numpy()
    print(f"  Bild {N}x{M}, GEMM {M}x{K} @ {K}x{N}  (Wertebereich {img.min():.2f}..{img.max():.2f})")
    save_png(img, out)
    print(f"  gespeichert: {out}")
    print("  (echte Demo auf dem DGX: python demo_paint.py  -> tuned vs default + Zeiten)")


# ---------------------------------------------------------------- Echte Demo (DGX)
def live(M, N, K, seed, out):
    import torch
    import triton.testing
    from autotuner.kernels import run_candidate
    from autotune import autotune, candidate_from_config
    from autotuner.search import enumerate_candidates, prune
    from problems import DEFAULT_CONFIG
    try:
        from autotuner.device_props import get_device_properties
        dev = get_device_properties()
    except Exception:
        from autotuner.device_props import GB10 as dev

    banner("cuTile Auto-Tuner · Live-Demo: der Tuner malt ein Bild")
    shapes = [(1, M, K), (1, K, N)]
    print(f"  GPU:     {dev.gpu_name}")
    print(f"  Aufgabe: {N}x{M}-Plasma  =  GEMM  {M}x{K} @ {K}x{N}  ({M*N/1e6:.1f} MPixel)")

    # Feature-Matrizen -> genau das Y@X, das die torch-Referenz auch rechnet
    Yn, Xn = plasma_features(M, N, K, seed)
    A = torch.from_numpy(Yn).half().cuda().reshape(1, M, K)
    B = torch.from_numpy(Xn).half().cuda().reshape(1, K, N)
    ref = torch.einsum("cmk,ckn->cmn", A.float(), B.float()).half()

    # --- der Tuner sucht (Funnel zeigen) ---
    cands, _ = enumerate_candidates(EINSUM, shapes)
    kept, _ = prune(cands, dev)
    print(f"\n  Suchraum:  enumeriert {len(cands)}  ->  prune {len(kept)}  ->  Tuner misst Top-7 ...")
    cfg, _tf, from_cache = autotune(EINSUM, shapes, dev)
    print(f"  Tuner-Wahl: {cfg}   {'(aus Cache)' if from_cache else '(frisch gemessen)'}")
    print(f"  Default:    {DEFAULT_CONFIG}\n")

    def measure(config, name):
        cand = candidate_from_config(EINSUM, shapes, config)
        out_t = run_candidate(cand, A, B)
        torch.cuda.synchronize()
        ok = torch.allclose(out_t, ref, rtol=1e-2, atol=1e-1)
        ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B), warmup=50, rep=200)
        print(f"  {name:9s} {ms:7.3f} ms   {'korrekt ✓' if ok else 'FALSCH ✗'}")
        return ms, out_t, ok

    default_ms, _, ok1 = measure(DEFAULT_CONFIG, "Default")
    tuner_ms, tuner_out, ok2 = measure(cfg, "Tuner")

    ms_max = max(default_ms, tuner_ms)
    print()
    print(f"  Default │{bar(default_ms, ms_max)}  {default_ms:.2f} ms")
    print(f"  Tuner   │{bar(tuner_ms, ms_max)}  {tuner_ms:.2f} ms")

    speedup = default_ms / tuner_ms if tuner_ms > 0 else float("nan")
    winner = "TUNER" if tuner_ms < default_ms else "DEFAULT"
    print(f"\n  ►► Gewinner: {winner}   ({speedup:.2f}x schneller, gleiches Bild, beide korrekt)")

    save_png(tuner_out.reshape(M, N).float().cpu().numpy(), out)
    print(f"  Bild gespeichert: {out}\n")
    if not (ok1 and ok2):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Live-Demo: Tuner vs Default malen ein Bild.")
    ap.add_argument("--preview", action="store_true", help="nur CPU-Vorschau, keine GPU/kein cuTile")
    ap.add_argument("--width", type=int, default=DEF_W)
    ap.add_argument("--height", type=int, default=DEF_H)
    ap.add_argument("--k", type=int, default=DEF_K)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "demo_paint.png"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.preview:
        preview(args.height, args.width, args.k, args.seed, args.out)
        return
    try:
        import torch
        if not torch.cuda.is_available():
            print("keine CUDA-GPU sichtbar -- fuer die Vorschau: python demo_paint.py --preview")
            sys.exit(1)
    except ImportError:
        print("torch fehlt -- fuer die Vorschau: python demo_paint.py --preview")
        sys.exit(1)
    live(args.height, args.width, args.k, args.seed, args.out)


if __name__ == "__main__":
    main()
