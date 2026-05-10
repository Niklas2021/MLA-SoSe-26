import cuda.tile as ct
import torch
import triton.testing

from task1 import ExecType
from task2 import generate_config
from task3 import Optimizer


# Problemgrößen aus der Aufgabe
C_SIZE = 4
M_SIZE = 4096
N_SIZE = 4096
K_SIZE = 4096

# Tile-Größen (fix), Rest wird zur Laufzeit aus den Tensor-Shapes berechnet
M_PRIM = 128
N_PRIM = 128
K_PRIM = 64
M_L2 = 8
N_L2 = 8


def ceildiv(a, b):
    return (a + b - 1) // b


# nur für die Print-Ausgabe in Task 4b
M_L2_OUTER = M_SIZE // (M_PRIM * M_L2)
N_L2_OUTER = N_SIZE // (N_PRIM * N_L2)
K_OUTER = K_SIZE // K_PRIM


def print_config(cfg, label=""):
    if label:
        print(f"--- {label} ---")
    print(f"  data_type:  {cfg.data_type.name}")
    print(f"  prim_main:  {cfg.prim_main.name}")
    print(f"  prim_last:  {cfg.prim_last.name}")
    print(f"  prim_first: {cfg.prim_first.name}")
    print(f"  dim_types:  {[d.name for d in cfg.dim_types]}")
    print(f"  exec_types: {[e.name for e in cfg.exec_types]}")
    print(f"  dim_sizes:  {cfg.dim_sizes}")
    print("  strides:")
    for i, s in enumerate(cfg.strides):
        print(f"    tensor {i}: {s}")
    print()


# Task 4a) Initiale Config aus generate_config
def task4a():
    print("=== Task 4a: initiale Config ===")
    cfg = generate_config("cmk, ckn -> cmn",
                          [(C_SIZE, M_SIZE, K_SIZE), (C_SIZE, K_SIZE, N_SIZE)])
    print_config(cfg, "generate_config Output")
    return cfg


# Task 4b) Variante A: m_l2/n_l2 als PAR -> BID-Swizzling à la Lecture 3
def task4b(cfg):
    print("=== Task 4b (Variante A) ===")
    print(f"  Tile-Größen: m_prim={M_PRIM}, n_prim={N_PRIM}, k_prim={K_PRIM}, m_l2={M_L2}, n_l2={N_L2}")

    opt = Optimizer(cfg)

    # m und n in (l2_outer, l2, prim) zerlegen, k in (k_outer, k_prim)
    opt.split_dim(1, 32, M_PRIM)
    opt.split_dim(1, M_L2_OUTER, M_L2)
    opt.split_dim(5, 32, N_PRIM)
    opt.split_dim(5, N_L2_OUTER, N_L2)
    opt.split_dim(4, K_OUTER, K_PRIM)
    # jetzt: [c, m_l2_outer, m_l2, m_prim, k_outer, k_prim, n_l2_outer, n_l2, n_prim]

    # umsortieren -> [c, m_l2_outer, n_l2_outer, m_l2, n_l2, k_outer, m_prim, n_prim, k_prim]
    opt.permute_dims([0, 1, 6, 2, 7, 4, 3, 8, 5])
    opt.make_executable()

    print_config(cfg, "L2-optimierte Config")
    return cfg


# Strict-Variante: k_outer in [...], m_l2/n_l2 als SEQ (Folie 34 strikt)
def task4b_strict():
    print("=== Task 4b (Variante B - strict) ===")
    cfg = generate_config("cmk, ckn -> cmn",
                          [(C_SIZE, M_SIZE, K_SIZE), (C_SIZE, K_SIZE, N_SIZE)])
    opt = Optimizer(cfg)

    # gleiche splits wie A
    opt.split_dim(1, 32, M_PRIM)
    opt.split_dim(1, M_L2_OUTER, M_L2)
    opt.split_dim(5, 32, N_PRIM)
    opt.split_dim(5, N_L2_OUTER, N_L2)
    opt.split_dim(4, K_OUTER, K_PRIM)

    # strikte Reihenfolge: [c, m_l2_outer, n_l2_outer, k_outer, m_l2, n_l2, m_prim, n_prim, k_prim]
    opt.permute_dims([0, 1, 6, 4, 2, 7, 3, 8, 5])

    # exec_types von Hand setzen (make_executable würde m_l2/n_l2 als PAR machen)
    cfg.exec_types = [
        ExecType.PAR, ExecType.PAR, ExecType.PAR,
        ExecType.SEQ, ExecType.SEQ, ExecType.SEQ,
        ExecType.PRIM, ExecType.PRIM, ExecType.PRIM,
    ]
    opt.verify()

    print_config(cfg, "L2-Config strict")
    return cfg


# Kernel - Variante A
# pid -> (c, m_l2_outer, n_l2_outer, m_l2, n_l2)
@ct.kernel
def kernel_l2(A, B, C,
              num_m_l2_outer: ct.Constant[int],
              num_n_l2_outer: ct.Constant[int],
              num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)

    n_l2_idx = pid % N_L2
    pid = pid // N_L2
    m_l2_idx = pid % M_L2
    pid = pid // M_L2
    n_l2_outer_idx = pid % num_n_l2_outer
    pid = pid // num_n_l2_outer
    m_l2_outer_idx = pid % num_m_l2_outer
    pid = pid // num_m_l2_outer
    c_idx = pid

    m_block = m_l2_outer_idx * M_L2 + m_l2_idx
    n_block = n_l2_outer_idx * N_L2 + n_l2_idx

    acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)

    for k_it in range(num_k_outer):
        # OOB wird durch padding_mode genullt
        a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                         shape=(1, M_PRIM, K_PRIM),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                         shape=(1, K_PRIM, N_PRIM),
                         padding_mode=ct.PaddingMode.ZERO)
        a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
        b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
        acc = ct.mma(a_tile, b_tile, acc)

    out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
    ct.store(C, index=(c_idx, m_block, n_block), tile=out)


# Kernel - Strict-Variante: pid -> (c, m_l2_outer, n_l2_outer), m_l2/n_l2 als Loops
@ct.kernel
def kernel_l2_strict(A, B, C,
                     num_m_l2_outer: ct.Constant[int],
                     num_n_l2_outer: ct.Constant[int],
                     num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)

    n_l2_outer_idx = pid % num_n_l2_outer
    pid = pid // num_n_l2_outer
    m_l2_outer_idx = pid % num_m_l2_outer
    pid = pid // num_m_l2_outer
    c_idx = pid

    for m_l2_it in range(M_L2):
        for n_l2_it in range(N_L2):
            m_block = m_l2_outer_idx * M_L2 + m_l2_it
            n_block = n_l2_outer_idx * N_L2 + n_l2_it

            acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)

            for k_it in range(num_k_outer):
                a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                                 shape=(1, M_PRIM, K_PRIM),
                                 padding_mode=ct.PaddingMode.ZERO)
                b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                                 shape=(1, K_PRIM, N_PRIM),
                                 padding_mode=ct.PaddingMode.ZERO)
                a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
                b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
                acc = ct.mma(a_tile, b_tile, acc)

            out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
            ct.store(C, index=(c_idx, m_block, n_block), tile=out)


# Baseline ohne swizzling: pid -> (c, m_block, n_block) plain row-major
@ct.kernel
def kernel_baseline(A, B, C,
                    num_m_blocks: ct.Constant[int],
                    num_n_blocks: ct.Constant[int],
                    num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)

    n_block = pid % num_n_blocks
    pid = pid // num_n_blocks
    m_block = pid % num_m_blocks
    pid = pid // num_m_blocks
    c_idx = pid

    acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)

    for k_it in range(num_k_outer):
        a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                         shape=(1, M_PRIM, K_PRIM),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                         shape=(1, K_PRIM, N_PRIM),
                         padding_mode=ct.PaddingMode.ZERO)
        a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
        b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
        acc = ct.mma(a_tile, b_tile, acc)

    out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
    ct.store(C, index=(c_idx, m_block, n_block), tile=out)


# Launcher - Output wird zur Tile-Größe gepaddet, am Ende slicen wir zurück
def run_l2(A, B):
    c_size, m_size, k_size = A.shape
    _, _, n_size = B.shape

    # Variante A braucht Padding bis zum Super-Tile (m_l2 * m_prim)
    num_m_l2_outer = ceildiv(m_size, M_PRIM * M_L2)
    num_n_l2_outer = ceildiv(n_size, N_PRIM * N_L2)
    num_k_outer = ceildiv(k_size, K_PRIM)

    m_pad = num_m_l2_outer * M_L2 * M_PRIM
    n_pad = num_n_l2_outer * N_L2 * N_PRIM

    C_pad = torch.zeros((c_size, m_pad, n_pad), dtype=torch.float16, device='cuda')
    grid = (c_size * num_m_l2_outer * num_n_l2_outer * M_L2 * N_L2,)
    ct.launch(torch.cuda.current_stream(), grid, kernel_l2,
              (A, B, C_pad, num_m_l2_outer, num_n_l2_outer, num_k_outer))
    return C_pad[:, :m_size, :n_size]


def run_baseline(A, B):
    c_size, m_size, k_size = A.shape
    _, _, n_size = B.shape

    num_m_blocks = ceildiv(m_size, M_PRIM)
    num_n_blocks = ceildiv(n_size, N_PRIM)
    num_k_outer = ceildiv(k_size, K_PRIM)

    m_pad = num_m_blocks * M_PRIM
    n_pad = num_n_blocks * N_PRIM

    C_pad = torch.zeros((c_size, m_pad, n_pad), dtype=torch.float16, device='cuda')
    grid = (c_size * num_m_blocks * num_n_blocks,)
    ct.launch(torch.cuda.current_stream(), grid, kernel_baseline,
              (A, B, C_pad, num_m_blocks, num_n_blocks, num_k_outer))
    return C_pad[:, :m_size, :n_size]


def run_l2_strict(A, B):
    c_size, m_size, k_size = A.shape
    _, _, n_size = B.shape

    num_m_l2_outer = ceildiv(m_size, M_PRIM * M_L2)
    num_n_l2_outer = ceildiv(n_size, N_PRIM * N_L2)
    num_k_outer = ceildiv(k_size, K_PRIM)

    m_pad = num_m_l2_outer * M_L2 * M_PRIM
    n_pad = num_n_l2_outer * N_L2 * N_PRIM

    C_pad = torch.zeros((c_size, m_pad, n_pad), dtype=torch.float16, device='cuda')
    # nur (c, m_l2_outer, n_l2_outer) parallelisiert
    grid = (c_size * num_m_l2_outer * num_n_l2_outer,)
    ct.launch(torch.cuda.current_stream(), grid, kernel_l2_strict,
              (A, B, C_pad, num_m_l2_outer, num_n_l2_outer, num_k_outer))
    return C_pad[:, :m_size, :n_size]


# Task 4c) Korrektheit
def task4c():
    print("=== Task 4c: Korrektheit ===")
    torch.manual_seed(0)
    A = torch.randn((C_SIZE, M_SIZE, K_SIZE), dtype=torch.float16, device='cuda')
    B = torch.randn((C_SIZE, K_SIZE, N_SIZE), dtype=torch.float16, device='cuda')

    # Referenz in fp32 rechnen, dann nach fp16 casten
    ref = torch.einsum('cmk,ckn->cmn', A.float(), B.float()).half()

    for label, fn in [("L2 (A)", run_l2), ("L2 strict", run_l2_strict), ("Baseline", run_baseline)]:
        out = fn(A, B)
        err = (out.float() - ref.float()).abs().max().item()
        ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-1)
        print(f"  {label:10s}:  max_err={err:.4f}  allclose={ok}")

    # nochmal mit unteilbaren Größen damit man sieht dass das padding_mode tut
    print()
    print("  -- schiefe Größen (M=1234, N=567, K=890, C=3) --")
    A2 = torch.randn((3, 1234, 890), dtype=torch.float16, device='cuda')
    B2 = torch.randn((3, 890, 567), dtype=torch.float16, device='cuda')
    ref2 = torch.einsum('cmk,ckn->cmn', A2.float(), B2.float()).half()

    for label, fn in [("L2 (A)", run_l2), ("L2 strict", run_l2_strict), ("Baseline", run_baseline)]:
        out = fn(A2, B2)
        err = (out.float() - ref2.float()).abs().max().item()
        ok = torch.allclose(out, ref2, rtol=1e-2, atol=1e-1)
        print(f"  {label:10s}:  shape={tuple(out.shape)}  max_err={err:.4f}  allclose={ok}")
    print()


# Task 4d) Benchmark
def task4d():
    print("=== Task 4d: Benchmark ===")
    torch.manual_seed(0)
    A = torch.randn((C_SIZE, M_SIZE, K_SIZE), dtype=torch.float16, device='cuda')
    B = torch.randn((C_SIZE, K_SIZE, N_SIZE), dtype=torch.float16, device='cuda')

    flops = 2.0 * C_SIZE * M_SIZE * N_SIZE * K_SIZE

    # warmup gegen JIT-Compile
    for _ in range(3):
        run_l2(A, B)
        run_l2_strict(A, B)
        run_baseline(A, B)
    torch.cuda.synchronize()

    ms_l2 = triton.testing.do_bench(lambda: run_l2(A, B), warmup=200, rep=2000)
    ms_strict = triton.testing.do_bench(lambda: run_l2_strict(A, B), warmup=200, rep=2000)
    ms_b = triton.testing.do_bench(lambda: run_baseline(A, B), warmup=200, rep=2000)

    tflops_l2 = flops / (ms_l2 * 1e-3) / 1e12
    tflops_strict = flops / (ms_strict * 1e-3) / 1e12
    tflops_b = flops / (ms_b * 1e-3) / 1e12

    print(f"  L2 Variante A:          {ms_l2:.3f} ms   {tflops_l2:.2f} TFLOPS")
    print(f"  L2 Variante B (strict): {ms_strict:.3f} ms   {tflops_strict:.2f} TFLOPS")
    print(f"  Baseline:               {ms_b:.3f} ms   {tflops_b:.2f} TFLOPS")
    print(f"  Speedup A vs Baseline: {ms_b / ms_l2:.2f}x")
    print(f"  Speedup B vs Baseline: {ms_b / ms_strict:.2f}x")
    print()


if __name__ == "__main__":
    cfg = task4a()
    task4b(cfg)
    task4b_strict()
    task4c()
    task4d()
