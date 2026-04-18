# Kein Teil der Abgabe – nur aus eigenem Interesse entstanden.
# Erweitert Task 4 um größere Matrixdimensionen und Log-Skalen,
# um Overhead-, L2-Cache- und HBM-Regime sichtbar zu machen.

import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
import time
from math import ceil

# ---- config ----
FIXED_ROWS  = 8192
COL_SWEEP   = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TILE_H      = 64
WARMUP_RUNS = 5
BENCH_RUNS  = 50
# ----------------

# Regime-Grenzen (zwischen welchen N-Werten der Übergang liegt)
REGIME_SPLITS = [
    (128,  256, "Overhead\ndominated"),
    (512, 1024, "L2\nCache"),
    (4096, None, "HBM"),
]


@ct.kernel
def copy_tile(src, dst, bh: ct.Constant[int], bw: ct.Constant[int]):
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    data = ct.load(src, index=(row_blk, col_blk), shape=(bh, bw))
    ct.store(dst, index=(row_blk, col_blk), tile=data)


def launch_copy(src, dst, tile_h, tile_w):
    rows, cols = src.shape
    grid = (ceil(rows / tile_h), ceil(cols / tile_w), 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        copy_tile,
        (src, dst, tile_h, tile_w),
    )


def bench_kernel(fn):
    for _ in range(WARMUP_RUNS):
        fn()
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    for _ in range(BENCH_RUNS):
        fn()
    cp.cuda.Device().synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / BENCH_RUNS


def main():
    print(f"=== Bandwidth Sweep  (M={FIXED_ROWS}, tile_M={TILE_H}, tile_N=N) ===\n")

    bw_vals   = []
    n_vals    = []
    time_vals = []

    for ncols in COL_SWEEP:
        src = cp.random.rand(FIXED_ROWS, ncols).astype(cp.float16)
        dst = cp.zeros_like(src)

        tile_w  = ncols
        elapsed = bench_kernel(lambda: launch_copy(src, dst, TILE_H, tile_w))

        bytes_moved = 2 * FIXED_ROWS * ncols * 2   # read + write, fp16 = 2 Byte
        bandwidth   = bytes_moved / (elapsed * 1e9)
        mb_moved    = bytes_moved / 1e6

        print(f"  N={ncols:5d}  data={mb_moved:7.1f} MB  "
              f"elapsed={elapsed*1e3:7.3f} ms  bandwidth={bandwidth:.1f} GB/s")

        bw_vals.append(bandwidth)
        n_vals.append(ncols)
        time_vals.append(elapsed * 1e3)

    _plot(n_vals, bw_vals, time_vals)


def _annotate_regimes(ax, n_vals):
    # Hintergrund-Färbung und Label für jedes Regime
    colors = ["#d4e6f1", "#d5f5e3", "#fdebd0"]
    labels = ["Overhead\ndominated", "L2 Cache", "HBM"]
    borders = [128, 512]   # N-Werte wo der Übergang stattfindet

    x_min = n_vals[0]
    segments = [
        (x_min,   borders[0]),
        (borders[0], borders[1]),
        (borders[1], n_vals[-1]),
    ]
    for (x0, x1), col, lbl in zip(segments, colors, labels):
        ax.axvspan(x0, x1, alpha=0.35, color=col, zorder=0)
        # Label in die Mitte des Segments (log-Mitte)
        mid = (x0 * x1) ** 0.5
        ax.text(mid, ax.get_ylim()[1], lbl, ha="center", va="top",
                fontsize=7.5, color="#333333")

    # Trennlinien
    for b in borders:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.9, zorder=1)


def _plot(n_vals, bw_vals, time_vals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Bandbreite ---
    ax1.plot(n_vals, bw_vals, marker="o", linewidth=1.7,
             color="steelblue", label="gemessen", zorder=2)
    for x, y in zip(n_vals, bw_vals):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color="steelblue")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("N  (Anzahl Spalten)")
    ax1.set_ylabel("Effektive Bandbreite  (GB/s)")
    ax1.set_title(f"Bandbreite  (M={FIXED_ROWS} fest, tile_M={TILE_H})")
    ax1.set_xticks(n_vals)
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax1.tick_params(axis="x", rotation=35)
    ax1.grid(which="both", alpha=0.25)
    ax1.legend(loc="upper left")
    ax1.set_ylim(bottom=min(bw_vals) * 0.7, top=max(bw_vals) * 2.5)
    _annotate_regimes(ax1, n_vals)

    # --- Laufzeit ---
    ax2.plot(n_vals, time_vals, marker="s", linewidth=1.7,
             color="darkorange", label="Laufzeit", zorder=2)
    for x, y in zip(n_vals, time_vals):
        ax2.annotate(f"{y:.3f} ms", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7.5, color="darkorange")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("N  (Anzahl Spalten)")
    ax2.set_ylabel("Laufzeit  (ms)")
    ax2.set_title(f"Kernel-Laufzeit  (M={FIXED_ROWS} fest, tile_M={TILE_H})")
    ax2.set_xticks(n_vals)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.tick_params(axis="x", rotation=35)
    ax2.grid(which="both", alpha=0.25)
    ax2.legend(loc="upper left")
    ax2.set_ylim(bottom=min(time_vals) * 0.7, top=max(time_vals) * 2.5)
    _annotate_regimes(ax2, n_vals)

    plt.tight_layout()
    plt.savefig("task4_2_bw.png", dpi=150)
    print("\nPlot gespeichert in task4_2_bw.png")


if __name__ == "__main__":
    main()
