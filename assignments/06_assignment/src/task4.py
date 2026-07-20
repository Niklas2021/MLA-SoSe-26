import sys
from pathlib import Path

# 06/src immer an Position 0, damit task2/task3 aus 06 (nicht 05) geladen werden
_src06 = Path(__file__).resolve().parent
sys.path.insert(0, str(_src06))

DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'lf_tr_64_intermediate.npz'

import cuda.tile as ct
import numpy as np
import torch
import triton.testing

from task2 import EINSUM
from task3 import PRIM_M, PRIM_N, PRIM_K, M_L2, N_L2


# Dimensionsgrößen aus den Tensordaten
SIZE_A = 4
SIZE_B = 4
SIZE_C = 3
SIZE_S = 64   # k_seq
SIZE_X = 1536
SIZE_Y = 1152

X_L2_OUTER = SIZE_X // (PRIM_M * M_L2)  # 6
Y_L2_OUTER = SIZE_Y // (PRIM_N * N_L2)  # 3


@ct.kernel
def kernel_lf(A, B, C):
    pid = ct.bid(0)

    # pid -> PAR-Dimensionen (innerste zuerst, passend zur Config-Reihenfolge)
    y_l2_idx = pid % N_L2;          pid = pid // N_L2
    x_l2_idx = pid % M_L2;          pid = pid // M_L2
    y_l2_out = pid % Y_L2_OUTER;    pid = pid // Y_L2_OUTER
    b_idx    = pid % SIZE_B;        pid = pid // SIZE_B
    x_l2_out = pid % X_L2_OUTER;    pid = pid // X_L2_OUTER
    c_idx    = pid % SIZE_C;        pid = pid // SIZE_C
    a_idx    = pid

    x_block = x_l2_out * M_L2 + x_l2_idx   # 0..9  (je 128 x-Elemente)
    y_block = y_l2_out * N_L2 + y_l2_idx   # 0..11 (je 128 y-Elemente)

    acc = ct.zeros((PRIM_M, PRIM_N), dtype=ct.float32)

    # k_seq-Loop über s; p (Größe 64) ist prim_k und wird im mma abgedeckt
    for s_it in range(SIZE_S):
        # A: (a, c, s, p, x) – p hat Stride 1536, x hat Stride 1
        # Tile-Shape: (1,1,1, prim_k, prim_m) = (1,1,1, 64, 128)
        tA = ct.load(A,
                     index=(a_idx, c_idx, s_it, 0, x_block),
                     shape=(1, 1, 1, PRIM_K, PRIM_M),
                     padding_mode=ct.PaddingMode.ZERO)
        tA = ct.reshape(tA, (PRIM_K, PRIM_M))
        tA = ct.permute(tA, (1, 0))   # (K, M) -> (M, K) für mma (lokales TTGT)

        # B: (b, s, p, y) – p hat Stride 1152, y hat Stride 1
        # Tile-Shape: (1,1, prim_k, prim_n) = (1,1, 64, 128)
        tB = ct.load(B,
                     index=(b_idx, s_it, 0, y_block),
                     shape=(1, 1, PRIM_K, PRIM_N),
                     padding_mode=ct.PaddingMode.ZERO)
        tB = ct.reshape(tB, (PRIM_K, PRIM_N))

        acc = ct.mma(tA, tB, acc)

    # C: (a, b, c, y, x) – y liegt vor x im Speicher, also acc transponieren
    out = ct.permute(acc, (1, 0))                        # (M, N) -> (N, M)
    out = ct.reshape(out, (1, 1, 1, PRIM_N, PRIM_M)).astype(ct.float16)
    ct.store(C, index=(a_idx, b_idx, c_idx, y_block, x_block), tile=out)


def run_kernel(A, B):
    C = torch.zeros((SIZE_A, SIZE_B, SIZE_C, SIZE_Y, SIZE_X),
                    dtype=torch.float16, device='cuda')
    grid = (SIZE_A * SIZE_C * X_L2_OUTER * SIZE_B * Y_L2_OUTER * M_L2 * N_L2,)
    ct.launch(torch.cuda.current_stream(), grid, kernel_lf, (A, B, C))
    return C


def task4():
    print("=== Task 4 ===")

    data = np.load(DATA_PATH)
    A = torch.tensor(data['tensor_acspx'], dtype=torch.float16, device='cuda')
    B = torch.tensor(data['tensor_bspy'],  dtype=torch.float16, device='cuda')

    ref = torch.einsum(EINSUM, A.float(), B.float()).half()

    # Korrektheit
    out = run_kernel(A, B)
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-1)
    max_err = (out.float() - ref.float()).abs().max().item()
    print(f"  allclose={ok}  max_err={max_err:.4f}")

    # Benchmark
    # FLOPs: 2 * Produkt aller Output-Dims * Produkt aller K-Dims
    flops = 2.0 * SIZE_A * SIZE_B * SIZE_C * SIZE_X * SIZE_Y * SIZE_S * PRIM_K
    ms_kernel = triton.testing.do_bench(lambda: run_kernel(A, B), warmup=50, rep=200)
    ms_ref = triton.testing.do_bench(
        lambda: torch.einsum(EINSUM, A.float(), B.float()), warmup=50, rep=200
    )
    print(f"  kernel:       {ms_kernel:.3f} ms   {flops / (ms_kernel * 1e-3) / 1e12:.2f} TFLOPS")
    print(f"  torch.einsum: {ms_ref:.3f} ms   {flops / (ms_ref * 1e-3) / 1e12:.2f} TFLOPS")


if __name__ == "__main__":
    task4()
