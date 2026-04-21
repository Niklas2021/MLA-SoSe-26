import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
import time
from math import ceil

# ---- config ----
FIXED_ROWS  = 2048
COL_SWEEP   = list(range(16, 129))  # N = 16 … 128 (jeder Wert, laut Aufgabenstellung)
TILE_H      = 64                    # tile_M bleibt konstant
WARMUP_RUNS = 10
BENCH_RUNS  = 100
# ----------------


@ct.kernel
def copy_tile(src, dst, bh: ct.Constant[int], bw: ct.Constant[int]):
    # Block-ID entlang beider Gitterdimensionen
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    # Tile laden und direkt in dst schreiben
    data = ct.load(src, index=(row_blk, col_blk), shape=(bh, bw))
    ct.store(dst, index=(row_blk, col_blk), tile=data)


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


def launch_copy(src, dst, tile_h, tile_w):
    rows, cols = src.shape
    # Tile-Dimensionen müssen Zweierpotenzen sein
    th = _next_pow2(tile_h)
    tw = _next_pow2(tile_w)
    # Array auf Tile-Vielfaches auffüllen
    pad_r = (th - rows % th) % th
    pad_c = (tw - cols % tw) % tw
    if pad_r or pad_c:
        src_p = cp.pad(src, ((0, pad_r), (0, pad_c)))
        dst_p = cp.zeros((rows + pad_r, cols + pad_c), dtype=dst.dtype)
    else:
        src_p, dst_p = src, dst

    grid = (ceil(rows / th), ceil(cols / tw), 1)
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        copy_tile,
        (src_p, dst_p, th, tw),
    )

    if pad_r or pad_c:
        cp.copyto(dst, dst_p[:rows, :cols])


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

    test_cases = [
        (512, 64,  "power-of-2 (512×64)"),
        (500, 70,  "non-power-of-2 (500×70)"),
        (1000, 96, "non-power-of-2 (1000×96)"),
    ]

    for rows, cols, label in test_cases:
        src = cp.random.rand(rows, cols).astype(cp.float16)
        dst = cp.zeros((rows, cols), dtype=cp.float16)

        launch_copy(src, dst, 64, 64)
        cp.cuda.Device().synchronize()

        assert cp.allclose(src, dst), f"Ausgabe stimmt nicht überein ({label})"
        print(f"  correctness check passed: {label}")

    print()


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
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_vals, bw_vals, linewidth=1.5, color="steelblue", label="gemessen")
    ax.set_xlabel("N  (Anzahl Spalten, 16 – 128)")
    ax.set_ylabel("Effektive Bandbreite  (GB/s)")
    ax.set_title(
        f"Copy Kernel – Speicherbandbreite\n"
        f"M={FIXED_ROWS} fest,  tile_M={TILE_H},  tile_N=next_pow2(N)"
    )
    ax.set_xticks([16, 32, 48, 64, 80, 96, 112, 128])
    ax.grid(axis="both", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig("task4_bw.png", dpi=150)
    print("\nPlot gespeichert in task4_bw.png")


if __name__ == "__main__":
    task4a()
    task4b()
