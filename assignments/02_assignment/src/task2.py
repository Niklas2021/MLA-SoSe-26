# Task 2: Matrix Reduction Kernel
# a) cuTile kernel: reduce 2D matrix (M, K) along K → output (M,) per-row sum
# b) Report theoretical impact of M/K dimension changes on parallelization

import cuda.tile as ct
import cupy as cp
import torch
