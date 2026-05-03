# Task 2 - Kernel Fusion: contraction + elementwise multiplication
# einsum: eabklxy, ecklyz -> eabcxz, danach C *= D

import cuda.tile as ct
import cupy as cp
import torch
import triton
import triton.testing

# kernel aus task1 b) wiederverwenden für den "ohne fusion" fall
from task1 import b_contraction


# ------------------- Task 2 a) fused contraction + elwise mult
@ct.kernel
def fused_kernel(A, B, C, D,
                 e: ct.Constant[int],
                 a: ct.Constant[int],
                 b: ct.Constant[int],
                 c: ct.Constant[int],
                 k: ct.Constant[int],
                 l: ct.Constant[int],
                 x: ct.Constant[int],
                 y: ct.Constant[int],
                 z: ct.Constant[int]):
    pid = ct.bid(0)

    # decompose pid -> (e, a, b, c)
    pid_c = pid % c
    pid = pid // c
    pid_b = pid % b
    pid = pid // b
    pid_a = pid % a
    pid = pid // a
    pid_e = pid

    acc = ct.zeros((x, z), dtype=ct.float32)

    for k_i in range(k):
        for l_i in range(l):
            a_tile = ct.load(A, index=(pid_e, pid_a, pid_b, k_i, l_i, 0, 0),
                             shape=(1, 1, 1, 1, 1, x, y))
            b_tile = ct.load(B, index=(pid_e, pid_c, k_i, l_i, 0, 0),
                             shape=(1, 1, 1, 1, y, z))
            a_tile = ct.reshape(a_tile, (x, y))
            b_tile = ct.reshape(b_tile, (y, z))
            acc = ct.mma(a_tile, b_tile, acc)

    # fusion: D laden und elementweise multiplizieren BEVOR wir nach global stören
    d_tile = ct.load(D, index=(pid_e, pid_a, pid_b, pid_c, 0, 0),
                     shape=(1, 1, 1, 1, x, z))
    d_tile = ct.reshape(d_tile, (x, z)).astype(ct.float32)
    acc = acc * d_tile

    out = ct.reshape(acc, (1, 1, 1, 1, x, z)).astype(ct.float16)
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, 0, 0), tile=out)


# ------------------- Task 2 b) elementwise mult only kernel
@ct.kernel
def elwise_kernel(C, D,
                  e: ct.Constant[int],
                  a: ct.Constant[int],
                  b: ct.Constant[int],
                  c: ct.Constant[int],
                  x: ct.Constant[int],
                  z: ct.Constant[int]):
    pid = ct.bid(0)
    pid_c = pid % c
    pid = pid // c
    pid_b = pid % b
    pid = pid // b
    pid_a = pid % a
    pid = pid // a
    pid_e = pid

    c_tile = ct.load(C, index=(pid_e, pid_a, pid_b, pid_c, 0, 0),
                     shape=(1, 1, 1, 1, x, z))
    d_tile = ct.load(D, index=(pid_e, pid_a, pid_b, pid_c, 0, 0),
                     shape=(1, 1, 1, 1, x, z))
    out = c_tile * d_tile
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, 0, 0), tile=out)


def run():
    # FLOP count des contractions soll ~ 2048^3 matmul sein
    # 2 * 2048^3 ~ 1.7e10
    # 2 * (e*a*b*c*x*z) * (k*l*y) = 2 * 8388608 * 1024 ~ 1.72e10  -> passt
    e, a, b, c = 8, 8, 16, 8
    k, l       = 8, 4
    x, y, z    = 32, 32, 32

    a_input = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device='cuda')
    b_input = torch.rand((e, c, k, l, y, z),    dtype=torch.float16, device='cuda')
    c_out   = torch.zeros((e, a, b, c, x, z),   dtype=torch.float16, device='cuda')
    d_input = torch.rand((e, a, b, c, x, z),    dtype=torch.float16, device='cuda')

    total_bytes = a_input.nbytes + b_input.nbytes + c_out.nbytes + d_input.nbytes
    assert total_bytes < 32 * 1024**3, f"Too large: {total_bytes / 1024**3:.2f} GiB"

    flops = 2 * e * a * b * c * x * z * k * l * y
    print(f"=== Task 2 - Kernel Fusion ===")
    print(f"  config: e={e} a={a} b={b} c={c} k={k} l={l} x={x} y={y} z={z}")
    print(f"  contraction FLOPs: {flops/1e9:.2f} GFLOP  (2048^3 matmul: {2*2048**3/1e9:.2f} GFLOP)")

    grid = (e * a * b * c,)

    # reference: einsum * D
    ref = (torch.einsum('eabklxy,ecklyz->eabcxz', a_input.float(), b_input.float())
           * d_input.float()).half()

    # ------------- a) fused
    c_out.zero_()
    ct.launch(torch.cuda.current_stream(), grid, fused_kernel,
              (a_input, b_input, c_out, d_input, e, a, b, c, k, l, x, y, z))
    print(f"  a) fused      verification: {torch.allclose(c_out, ref, atol=1e-2, rtol=1e-2)}")

    # ------------- b) sequenziell: contraction kernel aus task1, dann elwise_kernel
    c_out.zero_()
    ct.launch(torch.cuda.current_stream(), grid, b_contraction,
              (a_input, b_input, c_out, e, a, b, c, k, l, x, y, z))
    ct.launch(torch.cuda.current_stream(), grid, elwise_kernel,
              (c_out, d_input, e, a, b, c, x, z))
    print(f"  b) sequential verification: {torch.allclose(c_out, ref, atol=1e-2, rtol=1e-2)}")

    # ------------- benchmarks
    def fn_fused():
        ct.launch(torch.cuda.current_stream(), grid, fused_kernel,
                  (a_input, b_input, c_out, d_input, e, a, b, c, k, l, x, y, z))

    def fn_seq():
        ct.launch(torch.cuda.current_stream(), grid, b_contraction,
                  (a_input, b_input, c_out, e, a, b, c, k, l, x, y, z))
        ct.launch(torch.cuda.current_stream(), grid, elwise_kernel,
                  (c_out, d_input, e, a, b, c, x, z))

    # warmup
    for _ in range(3):
        fn_fused()
        fn_seq()
    torch.cuda.synchronize()

    ms_fused = triton.testing.do_bench(fn_fused, warmup=25, rep=200)
    ms_seq   = triton.testing.do_bench(fn_seq,   warmup=25, rep=200)

    print()
    print(f"  fused       : {ms_fused:8.4f} ms")
    print(f"  sequential  : {ms_seq:8.4f} ms")
    print(f"  speedup     : {ms_seq / ms_fused:.2f}x")


if __name__ == "__main__":
    run()
