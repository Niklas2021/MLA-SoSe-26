#L2 Cache Optimization via Block Swizzling

import cuda.tile as ct
import torch
import cupy as cp
import math #for math.ceil(M / m_tile)

import numpy as np
import triton
import triton.testing


from task2 import matmul as matmul_rowmajor
from task3 import tflops


# a)
@ct.kernel
def kernel_swizzled(a, b, c, num_row_tiles,
                             num_col_tiles,
                             K: ct.Constant[int],
                             m_tile: ct.Constant[int],
                             n_tile: ct.Constant[int],
                             k_tile: ct.Constant[int],
                             group_size: ct.Constant[int]):
    bids_per_group  = group_size * num_col_tiles

    bid = ct.bid(0)
    group_id = bid // bids_per_group
    first_m= group_id * group_size

    if (num_row_tiles - first_m < group_size):
        this_group_size = num_row_tiles - first_m
    # set the "custom" size of the last group if its smaller than actual group size
    else: this_group_size = group_size 

    bid_row = first_m + (bid % bids_per_group) % this_group_size
    bid_col = (bid % bids_per_group) // this_group_size

    acc = ct.full((m_tile, n_tile), fill_value=0, dtype=ct.float32)
    for ki in range(K // k_tile):
        a_tile = ct.load(a, index=(bid_row, ki), shape=(m_tile, k_tile))
        b_tile = ct.load(b, index=(ki, bid_col), shape=(k_tile, n_tile))
        acc    = ct.mma(a_tile, b_tile, acc)

    ct.store(c, (bid_row, bid_col), tile=acc)


def compute_group_size(K, m_tile, n_tile, bytes_per_element):
    l2_bytes = cp.cuda.Device().attributes["L2CacheSize"]
    stripe_bytes = K * (m_tile + n_tile) * bytes_per_element 

    # if even a stripe doesnt fit into L2 -> group_size will be 1 and its the same as having row-major order
    if stripe_bytes > l2_bytes:
        print(
            f"WARNING! Stripe ({stripe_bytes/1024:.1f} KB) "
            f"doesnt fit into L2 Cache ({l2_bytes/1024:.1f} KB) "
            f"Swizzle has no effect - group_size=1"
        )
        return 1
    else:
        return l2_bytes // stripe_bytes
    
def run_task4a():
    torch.cuda.init()

    # "The kernel should work with tile sizes that are specified by the calling function"
    m_tile = n_tile = k_tile = 64

    M, N, K = 256, 256, 4096
    A = torch.rand((M, K), dtype=torch.float16, device='cuda')
    B = torch.rand((K, N), dtype=torch.float16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')

    launch_swizzled(A, B, C, m_tile, n_tile, k_tile)
    print("Verification:", torch.allclose(C, torch.matmul(A, B).float(), atol=1e-2, rtol=1e-2))


def launch_swizzled(A, B, C, m_tile,  n_tile, k_tile):
    M, K = A.shape
    K, N = B.shape
    bytes_per_element = 2 if A.dtype == torch.float16 else 4
    group_size    = compute_group_size(K, m_tile, n_tile, bytes_per_element)
    num_row_tiles = math.ceil(M / m_tile)
    num_col_tiles = math.ceil(N /  n_tile)
    grid = (num_row_tiles * num_col_tiles,  )

    ct.launch(torch.cuda.current_stream(), grid, kernel_swizzled,
              (A, B, C, num_row_tiles, num_col_tiles, K, m_tile,  n_tile, k_tile, group_size))


# b)
def bench_swizzled(M, N, K, m_tile,  n_tile, k_tile):
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')
    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    fn = lambda: launch_swizzled(A, B, C, m_tile,  n_tile, k_tile)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=25, rep=200)


def task4b_size(N):
    print(f"=== Task 4b - tile sweep (swizzled) bei {N}^3 ===")
    tile_opts = [32, 64, 128]
    cube      = np.zeros((3, 3, 3))
    best_combo, best_tf = None, -1.0

    for i, m_tile in enumerate(tile_opts):
        for j,  n_tile in enumerate(tile_opts):
            for k_, k_tile in enumerate(tile_opts):
                ms = bench_swizzled(N, N, N, m_tile,  n_tile, k_tile)
                tf = tflops(N, N, N, ms)
                cube[i, j, k_] = tf
                print(f"  tiles=({m_tile:3d},{ n_tile:3d},{k_tile:3d})   "
                      f"{ms:7.3f} ms   {tf:6.2f} TFLOPS")
                if tf > best_tf:
                    best_tf    = tf
                    best_combo = (m_tile,  n_tile, k_tile)

    print(f"  -> best tile shape {N}^3: {best_combo} - {best_tf:.2f} TFLOPS\n")

    return best_combo, best_tf


def run_task4b():
    best_2048 = task4b_size(2048)
    best_512  = task4b_size(512)

    print("=== Task 4b - Summary tile sweep ===")
    print(f"  best @ 2048^3: tiles={best_2048[0]}  {best_2048[1]:.2f} TFLOPS")
    print(f"  best @  512^3: tiles={best_512[0]}  {best_512[1]:.2f} TFLOPS")

    # swizzled vs row-major order
    print("\n=== Task 4b -  8192 x 8192 x 4096 ===")
    M, N, K = 8192, 8192, 4096
    m_tile =  n_tile = k_tile = 64

    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B = torch.randn((K, N), dtype=torch.float16, device='cuda')

    C = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    fn_row = lambda: matmul_rowmajor(A, B, m_tile,  n_tile, k_tile)
    fn_swz = lambda: launch_swizzled(A, B, C, m_tile,  n_tile, k_tile)

    for _ in range(3):
        fn_row(); fn_swz()
    torch.cuda.synchronize()

    ms_row = triton.testing.do_bench(fn_row, warmup=25, rep=200)
    ms_swz = triton.testing.do_bench(fn_swz, warmup=25, rep=200)

    tf_row = tflops(M, N, K, ms_row)
    tf_swz = tflops(M, N, K, ms_swz)

    print(f"  Row-major: {ms_row:.3f} ms  -> {tf_row:.2f} TFLOPS")
    print(f"  Swizzled:  {ms_swz:.3f} ms  ->  {tf_swz:.2f} TFLOPS")
    print(f"  Speedup:   {ms_row / ms_swz:.2f}x")


if __name__ == "__main__":
    run_task4a()
    run_task4b()