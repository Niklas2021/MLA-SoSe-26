# Task 3 - GEMM Dimension Size Sweep
# einsum: ackm, bcnk -> abnm
# klassifikation: M = {a, m}, N = {b, n}, K = {c, k}
# fix: |a|=16, |b|=16, |c|=32. variabel: m, n, k

import os

import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
import torch
import triton
import triton.testing


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "out", "task3")
os.makedirs(OUT_DIR, exist_ok=True)


# fixe größen, laut aufgabenstellung
A_SIZE = 16
B_SIZE = 16
C_SIZE = 32


# ------------------------------------------------------------ #
# Task 3 a) Kernel                                             #
# ------------------------------------------------------------ #
# pro (a, b, n_block, m_block) ein cta. iteriere c und k_blocks sequenziell.
# acc[n_tile, m_tile] = sum_{c, k_block} B[b, c, n_block, k_block] @ A[a, c, k_block, m_block]
@ct.kernel
def task3_kernel(A, B, C,
                 num_n_blocks: ct.Constant[int],
                 num_m_blocks: ct.Constant[int],
                 num_k_blocks: ct.Constant[int],
                 m_tile: ct.Constant[int],
                 n_tile: ct.Constant[int],
                 k_tile: ct.Constant[int]):
    pid = ct.bid(0)

    # decompose pid -> (a_i, b_i, n_block, m_block)
    pid_m = pid % num_m_blocks
    pid = pid // num_m_blocks
    pid_n = pid % num_n_blocks
    pid = pid // num_n_blocks
    pid_b = pid % B_SIZE
    pid = pid // B_SIZE
    pid_a = pid

    acc = ct.zeros((n_tile, m_tile), dtype=ct.float32)

    for c_i in range(C_SIZE):
        for k_i in range(num_k_blocks):
            # A: ackm, shape (16, 32, k_pad, m_pad)
            a_tile = ct.load(A, index=(pid_a, c_i, k_i, pid_m),
                             shape=(1, 1, k_tile, m_tile))
            # B: bcnk, shape (16, 32, n_pad, k_pad)
            b_tile = ct.load(B, index=(pid_b, c_i, pid_n, k_i),
                             shape=(1, 1, n_tile, k_tile))
            a_tile = ct.reshape(a_tile, (k_tile, m_tile))
            b_tile = ct.reshape(b_tile, (n_tile, k_tile))
            # output: (n_tile, m_tile) = B-tile @ A-tile
            acc = ct.mma(b_tile, a_tile, acc)

    out = ct.reshape(acc, (1, 1, n_tile, m_tile)).astype(ct.float16)
    ct.store(C, index=(pid_a, pid_b, pid_n, pid_m), tile=out)


def ceildiv(a, b):
    return (a + b - 1) // b


def run(A, B, m_tile=32, n_tile=32, k_tile=32):
    # A: (16, 32, k, m), B: (16, 32, n, k) -> C: (16, 16, n, m)
    a_size, c_size_a, k, m = A.shape
    b_size, c_size_b, n, k_b = B.shape
    assert a_size == A_SIZE and b_size == B_SIZE
    assert c_size_a == C_SIZE and c_size_b == C_SIZE
    assert k == k_b

    m_pad = ceildiv(m, m_tile) * m_tile
    n_pad = ceildiv(n, n_tile) * n_tile
    k_pad = ceildiv(k, k_tile) * k_tile

    # padding falls nötig
    if m_pad != m or k_pad != k:
        A_p = torch.zeros((A_SIZE, C_SIZE, k_pad, m_pad), dtype=A.dtype, device=A.device)
        A_p[:, :, :k, :m] = A
    else:
        A_p = A

    if n_pad != n or k_pad != k:
        B_p = torch.zeros((B_SIZE, C_SIZE, n_pad, k_pad), dtype=B.dtype, device=B.device)
        B_p[:, :, :n, :k] = B
    else:
        B_p = B

    C_p = torch.zeros((A_SIZE, B_SIZE, n_pad, m_pad), dtype=torch.float16, device=A.device)

    num_m = m_pad // m_tile
    num_n = n_pad // n_tile
    num_k = k_pad // k_tile

    grid = (A_SIZE * B_SIZE * num_n * num_m,)

    ct.launch(torch.cuda.current_stream(), grid, task3_kernel,
              (A_p, B_p, C_p, num_n, num_m, num_k, m_tile, n_tile, k_tile))

    return C_p[:, :, :n, :m]


# ------------------------------------------------------------ #
# Korrektheit                                                  #
# ------------------------------------------------------------ #

def check(m, n, k):
    torch.manual_seed(0)
    A = torch.randn((A_SIZE, C_SIZE, k, m), dtype=torch.float16, device='cuda')
    B = torch.randn((B_SIZE, C_SIZE, n, k), dtype=torch.float16, device='cuda')
    C = run(A, B)
    ref = torch.einsum('ackm,bcnk->abnm', A.float(), B.float()).half()
    err = (C - ref).abs().max().item()
    ok = torch.allclose(C, ref, rtol=1e-2, atol=1e-1)
    print(f"  m={m:4d} n={n:4d} k={k:4d}   max_err={err:.4f}   allclose={ok}")
    return ok


# ------------------------------------------------------------ #
# Benchmark + Sweep                                            #
# ------------------------------------------------------------ #

def tflops(m, n, k, ms):
    # flops = 2 * |a| * |b| * |c| * m * n * k
    flops = 2.0 * A_SIZE * B_SIZE * C_SIZE * m * n * k
    return flops / (ms * 1e-3) / 1e12


def bench(m, n, k):
    torch.manual_seed(0)
    A = torch.randn((A_SIZE, C_SIZE, k, m), dtype=torch.float16, device='cuda')
    B = torch.randn((B_SIZE, C_SIZE, n, k), dtype=torch.float16, device='cuda')

    fn = lambda: run(A, B)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=25, rep=200)


def sweep(varying, fixed_label, fixed_vals, m_fn, n_fn, k_fn, out_name, title):
    """
    varying: liste von werten für die schwankende dim
    m_fn, n_fn, k_fn: callables die aus dem schwankenden wert die dim-größe machen
    """
    print(f"=== Task 3 b) sweep {title} ===")
    tflops_list = []
    ms_list = []
    for v in varying:
        m = m_fn(v); n = n_fn(v); k = k_fn(v)
        ms = bench(m, n, k)
        tf = tflops(m, n, k, ms)
        tflops_list.append(tf)
        ms_list.append(ms)
        print(f"  m={m:3d} n={n:3d} k={k:3d}   {ms:7.4f} ms   {tf:6.2f} TFLOPS")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(varying, tflops_list, marker="o", linewidth=1.4, markersize=3, color="steelblue")
    # power-of-two werte hervorheben
    for p in [32, 64, 128]:
        if p in varying:
            ax.axvline(p, color="lightgrey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(fixed_label)
    ax.set_ylabel("TFLOPS")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  plot gespeichert: {out}\n")


def task3a():
    print("=== Task 3 a) Korrektheit ===")
    # ein paar saubere und ein paar non-pow2 cases
    check(64, 64, 64)
    check(64, 128, 64)
    check(17, 17, 17)
    check(64, 129, 64)
    check(64, 64, 129)
    check(33, 51, 97)
    print()


def task3b():
    sizes = list(range(17, 130))

    # 1) k=64, m=64, sweep n
    sweep(sizes, "n",
          fixed_vals=None,
          m_fn=lambda v: 64, n_fn=lambda v: v, k_fn=lambda v: 64,
          out_name="task3b_sweep_n.png",
          title="Task 3b - sweep n (m=64, k=64)")

    # 2) m=64, n=64, sweep k
    sweep(sizes, "k",
          fixed_vals=None,
          m_fn=lambda v: 64, n_fn=lambda v: 64, k_fn=lambda v: v,
          out_name="task3b_sweep_k.png",
          title="Task 3b - sweep k (m=64, n=64)")


if __name__ == "__main__":
    task3a()
    task3b()
