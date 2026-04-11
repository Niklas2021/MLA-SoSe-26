import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from assignment_01 import einsum_gemm, einsum_gemm_dot, einsum_loops


EINSUM_EXPR = "acsxp, bspy -> abcxy"


def benchmark_total_seconds(fn, args: tuple, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        fn(*args)

    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    return time.perf_counter() - start


def run_heavy_benchmark() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)

    A = torch.rand(2, 4, 5, 4, 3)
    B = torch.rand(3, 5, 3, 5)
    reference = torch.einsum(EINSUM_EXPR, A, B)
    reference_np = reference.numpy()

    A_np = A.numpy()
    B_np = B.numpy()

    repeats = 120
    warmup = 25

    methods = [
        ("loops", einsum_loops, (A, B), "#4C78A8"),
        ("gemm", einsum_gemm, (A, B), "#59A14F"),
        ("gemm_dot", einsum_gemm_dot, (A, B), "#F28E2B"),
        ("torch_einsum", lambda x, y: torch.einsum(EINSUM_EXPR, x, y), (A, B), "#E15759"),
        ("numpy_einsum", lambda x, y: np.einsum("acsxp,bspy->abcxy", x, y, optimize=True), (A_np, B_np), "#76B7B2"),
    ]

    totals = {}
    colors = []
    for name, fn, args, color in methods:
        result = fn(*args)
        if isinstance(result, torch.Tensor):
            ok = torch.allclose(result, reference, atol=1e-5)
        else:
            ok = np.allclose(result, reference_np, atol=1e-5)
        if not ok:
            raise RuntimeError(f"{name} ist nicht korrekt gegen torch.einsum")

        total = benchmark_total_seconds(fn, args, repeats=repeats, warmup=warmup)
        totals[name] = total
        colors.append(color)
        print(f"{name:12s} total={total:8.3f} s | pro Lauf={total/repeats:10.6f} s")

    labels = list(totals.keys())
    values = [totals[k] for k in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bars_linear = ax1.bar(labels, values, color=colors)
    bars_log = ax2.bar(labels, values, color=colors)

    ax1.set_title(f"Task 3 Heavy-Load Benchmark ({repeats} Wiederholungen)")
    ax1.set_ylabel("Gesamtzeit [Sekunden]")
    ax1.tick_params(axis="x", rotation=20)

    ax2.set_title("Gleiche Daten mit log-Skala")
    ax2.set_ylabel("Gesamtzeit [Sekunden] (log)")
    ax2.set_yscale("log")
    ax2.tick_params(axis="x", rotation=20)

    for bar, value in zip(bars_linear, values):
        label = f"{value:.2f}s" if value >= 1.0 else f"{value:.3f}s"
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, value in zip(bars_log, values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f"{value:.4f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.text(
        0.5,
        0.02,
        "Vergleichshinweis: gemessen auf einem M4 Pro chip.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    plot_path = Path(__file__).with_name("assignment_01_task3_bench_plot.png")
    fig.savefig(plot_path, dpi=180)
    print(f"Plot gespeichert unter: {plot_path}")


if __name__ == "__main__":
    run_heavy_benchmark()
