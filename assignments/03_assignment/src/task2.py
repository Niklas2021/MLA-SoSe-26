# Task 2 - Matmul Kernel mit cuTile
# C = A @ B,  ein Block macht ein output tile

import cuda.tile as ct
import cupy as cp
import torch
import triton


@ct.kernel
def matmul_kernel(A, B, C,
                  num_n_tiles: ct.Constant[int],
                  num_k_tiles: ct.Constant[int],
                  m_tile: ct.Constant[int],
                  n_tile: ct.Constant[int],
                  k_tile: ct.Constant[int]):

    # row-major mapping (siehe Aufgabe)
    pid = ct.bid(0)
    bid_m = pid // num_n_tiles
    bid_n = pid % num_n_tiles

    # accumulator in fp32
    acc = ct.zeros((m_tile, n_tile), dtype=ct.float32)

    # k-loop
    for k in range(num_k_tiles):
        a_tile = ct.load(A, index=(bid_m, k), shape=(m_tile, k_tile))
        b_tile = ct.load(B, index=(k, bid_n), shape=(k_tile, n_tile))
        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(C, index=(bid_m, bid_n), tile=acc)


def ceildiv(a, b):
    return (a + b - 1) // b


def matmul(A, B, m_tile=64, n_tile=64, k_tile=64):
    # padden auf vielfaches der tile-size, damit auch non-pow2 shapes gehen
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    M_pad = ceildiv(M, m_tile) * m_tile
    N_pad = ceildiv(N, n_tile) * n_tile
    K_pad = ceildiv(K, k_tile) * k_tile

    if M_pad != M or K_pad != K:
        A_p = torch.zeros((M_pad, K_pad), dtype=A.dtype, device=A.device)
        A_p[:M, :K] = A
    else:
        A_p = A

    if K_pad != K2 or N_pad != N:
        B_p = torch.zeros((K_pad, N_pad), dtype=B.dtype, device=B.device)
        B_p[:K2, :N] = B
    else:
        B_p = B

    C_p = torch.zeros((M_pad, N_pad), dtype=torch.float32, device=A.device)

    n_m = M_pad // m_tile
    n_n = N_pad // n_tile
    n_k = K_pad // k_tile

    grid = (n_m * n_n, 1, 1)

    ct.launch(torch.cuda.current_stream().cuda_stream,
              grid,
              matmul_kernel,
              (A_p, B_p, C_p, n_n, n_k, m_tile, n_tile, k_tile))

    # ohne padding zurückgeben
    return C_p[:M, :N]


# kleine helper zum testen
def check(M, K, N, mt=64, nt=64, kt=64):
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')

    C    = matmul(A, B, mt, nt, kt)
    Cref = torch.matmul(A.float(), B.float())

    ok = torch.allclose(C, Cref, rtol=1e-2, atol=1e-1)
    err = (C - Cref).abs().max().item()
    print(f"  M={M:5d} K={K:5d} N={N:5d}  tiles=({mt},{nt},{kt})  max_err={err:.4f}  allclose={ok}")
    return ok


if __name__ == "__main__":
    print("=== Task 2 - Korrektheit ===")
    # erstmal die einfachen pow2 cases
    check(256, 256, 256)
    check(1024, 512, 768)

    # und jetzt non-pow2 (laut aufgabe required)
    check(100, 200, 300)
    check(513, 257, 129)
    check(1000, 999, 777)
