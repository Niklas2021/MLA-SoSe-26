import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
import time
from math import ceil

# ---- config ----
FIXED_ROWS  = 2048
COL_SWEEP   = [16, 32, 64, 128]   # getestete N-Werte in Teil b
TILE_H      = 64                   # tile_M bleibt konstant
WARMUP_RUNS = 10
BENCH_RUNS  = 200
# ----------------


@ct.kernel
def copy_tile(src, dst, bh: ct.Constant[int], bw: ct.Constant[int]):
    # Block-ID entlang beider Gitterdimensionen
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    # Tile laden und direkt in dst schreiben
    data = ct.load(src, index=(row_blk, col_blk), shape=(bh, bw))
    ct.store(dst, index=(row_blk, col_blk), tile=data)


def launch_copy(src, dst, tile_h, tile_w):
    rows, cols = src.shape
    # Gittergröße: wie viele Tiles passen in jede Dimension
    grid = (ceil(rows / tile_h), ceil(cols / tile_w), 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        copy_tile,
        (src, dst, tile_h, tile_w),
    )


def bench_kernel(fn):
    # Warmup – JIT-Kompilierung und Cache-Effekte rausrechnen
    for _ in range(WARMUP_RUNS):
        fn()
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        fn()
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()

    # Durchschnitt in Sekunden pro Aufruf
    return (t1 - t0) / BENCH_RUNS


# ------------------------------------------------------------------ #
#  Task 4a  –  Korrektheit                                           #
# ------------------------------------------------------------------ #

def task4a():
    print("=== Task 4a – Correctness ===")

    rows, cols = 512, 64
    src = cp.random.rand(rows, cols).astype(cp.float16)
    dst = cp.zeros((rows, cols), dtype=cp.float16)

    launch_copy(src, dst, 64, 64)
    cp.cuda.Device().synchronize()

    # src und dst müssen elementweise übereinstimmen
    assert cp.allclose(src, dst), "Ausgabe stimmt nicht mit Eingabe überein"
    print("correctness check passed\n")


# ------------------------------------------------------------------ #
#  Task 4b  –  Bandwidth-Sweep                                        #
# ------------------------------------------------------------------ #

def task4b():
    print("=== Task 4b – Bandwidth Sweep ===")

    bw_vals = []
    n_vals  = []

    for ncols in COL_SWEEP:
        src = cp.random.rand(FIXED_ROWS, ncols).astype(cp.float16)
        dst = cp.zeros_like(src)

        # tile_N = N (volle Breite, laut Aufgabenstellung)
        tile_w = ncols

        elapsed = bench_kernel(lambda: launch_copy(src, dst, TILE_H, tile_w))

        # Formel aus dem Assignment: Faktor 2 für Lesen + Schreiben,
        # nochmal *2 weil fp16 = 2 Byte pro Element
        bytes_moved = 2 * FIXED_ROWS * ncols * 2
        bandwidth   = bytes_moved / (elapsed * 1e9)

        print(f"  N={ncols:4d}  elapsed={elapsed*1e6:7.2f} µs  "
              f"bandwidth={bandwidth:.2f} GB/s")

        bw_vals.append(bandwidth)
        n_vals.append(ncols)

    _plot_results(n_vals, bw_vals)


def _plot_results(n_vals, bw_vals):
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        n_vals, bw_vals,
        marker="o", linewidth=1.7, color="steelblue", label="gemessen"
    )
    ax.set_xlabel("N  (Anzahl Spalten)")
    ax.set_ylabel("Effektive Bandbreite  (GB/s)")
    ax.set_title(
        f"Copy Kernel – Speicherbandbreite\n"
        f"M={FIXED_ROWS} fest,  tile_M={TILE_H},  tile_N=N"
    )
    ax.set_xticks(n_vals)
    ax.grid(axis="y", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig("task4_bw.png", dpi=150)
    print("\nPlot gespeichert in task4_bw.png")


if __name__ == "__main__":
    task4a()
    task4b()
