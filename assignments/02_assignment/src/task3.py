# Task 3: 4D Tensor Elementwise Addition
# a) Two cuTile kernels for A + B on tensors (M, N, K, L):
#    - Kernel 1: tile covers (K, L), parallelize over (M, N)
#    - Kernel 2: tile covers (M, N), parallelize over (K, L)
# b) Benchmark both with triton.testing.do_bench; M=16, N=128, K=16, L=128

import cuda.tile as ct
import cupy as cp
import torch
import triton.testing
