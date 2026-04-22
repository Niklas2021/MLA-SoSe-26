# Task 2: Simple Matrix Multiplication Kernel
# C = A @ B  with A(M,K), B(K,N), C(M,N)
# - One output tile (m_tile, n_tile) per block
# - Tile sizes passed by caller
# - Row-major BID mapping: BID 0 = top-left tile, BID 1 = tile to its right, wrap to next row
# - Must support non-power-of-2 shapes
# - Use ct.mma for inner accumulation
# - Verify via torch.allclose against torch.matmul

import cuda.tile as ct
import cupy as cp
import torch
import triton
