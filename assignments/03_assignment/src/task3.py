# Task 3 - Benchmarks fuer den matmul kernel aus task2
# a) square sweep mit tile (64,64,64)
# b) tile sweep ueber 27 kombis bei 2048^3 und 512^3

import os

import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import triton.testing

from task2 import matmul


# output dir fuer die plots
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "out", "task3")
os.makedirs(OUT_DIR, exist_ok=True)


# tflops formel aus der aufgabenstellung
def tflops(M, N, K, ms):
    sec = ms * 1e-3
    return (2.0 * M * N * K) / sec / 1e12


def bench(M, N, K, mt, nt, kt):
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')

    fn = lambda: matmul(A, B, mt, nt, kt)

    # warmup, sonst misst man den jit-compile mit
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(fn, warmup=25, rep=200)
    return ms


# ------------------------------------------------------------ #
# Task 3a                                                      #
# ------------------------------------------------------------ #

def task3a():
    print("=== Task 3a - square matmul, tiles (64,64,64) ===")
    sizes = [256, 512, 1024, 2048, 4096, 8192]

    tflops_list = []
    for n in sizes:
        ms = bench(n, n, n, 64, 64, 64)
        tf = tflops(n, n, n, ms)
        tflops_list.append(tf)
        print(f"  M=N=K={n:5d}   {ms:8.3f} ms   {tf:7.2f} TFLOPS")

    # plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, tflops_list, marker="o", linewidth=1.7, color="steelblue")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Matrix size  M = N = K")
    ax.set_ylabel("TFLOPS")
    ax.set_title("Task 3a - Square matmul (tiles 64,64,64)")
    ax.grid(alpha=0.35)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "task3a_tflops.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot gespeichert: {out}\n")


# ------------------------------------------------------------ #
# Task 3b                                                      #
# ------------------------------------------------------------ #

def task3b_size(N):
    print(f"=== Task 3b - tile sweep bei {N}^3 ===")
    tile_opts = [32, 64, 128]

    # ergebnisse als 3d-cube (m, n, k)
    cube = np.zeros((3, 3, 3))

    best_combo = None
    best_tf    = -1.0

    for i, mt in enumerate(tile_opts):
        for j, nt in enumerate(tile_opts):
            for k_, kt in enumerate(tile_opts):
                ms = bench(N, N, N, mt, nt, kt)
                tf = tflops(N, N, N, ms)
                cube[i, j, k_] = tf
                print(f"  tiles=({mt:3d},{nt:3d},{kt:3d})   "
                      f"{ms:7.3f} ms   {tf:6.2f} TFLOPS")
                if tf > best_tf:
                    best_tf = tf
                    best_combo = (mt, nt, kt)

    print(f"  --> bester tile shape bei {N}^3: {best_combo} mit {best_tf:.2f} TFLOPS\n")

    # heatmap, kt = 64 fix
    k64 = tile_opts.index(64)
    heat = cube[:, :, k64]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(heat, cmap="viridis", origin="lower")
    ax.set_xticks(range(3)); ax.set_xticklabels(tile_opts)
    ax.set_yticks(range(3)); ax.set_yticklabels(tile_opts)
    ax.set_xlabel("n_tile")
    ax.set_ylabel("m_tile")
    ax.set_title(f"Task 3b - TFLOPS bei {N}^3 (k_tile=64)")

    # zahlen in die zellen schreiben
    for i in range(3):
        for j in range(3):
            v = heat[i, j]
            color = "white" if v < heat.max() * 0.6 else "black"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, label="TFLOPS")
    plt.tight_layout()

    out = os.path.join(OUT_DIR, f"task3b_heatmap_{N}.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  heatmap gespeichert: {out}\n")

    return best_combo, best_tf


def task3b():
    best_2048 = task3b_size(2048)
    best_512  = task3b_size(512)

    print("=== Task 3b - Summary ===")
    print(f"  best @ 2048^3: tiles={best_2048[0]}  {best_2048[1]:.2f} TFLOPS")
    print(f"  best @  512^3: tiles={best_512[0]}  {best_512[1]:.2f} TFLOPS")


if __name__ == "__main__":
    task3a()
    task3b()
