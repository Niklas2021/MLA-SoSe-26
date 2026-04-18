# Task 4: Benchmarking Bandwidth
# a) cuTile kernel: copy 2D matrix (M, N) using 2D tiles (tile_M, tile_N)
# b) M=2048 fixed, N in [16..128], tile_M=64, tile_N=N
#    Measure runtime, compute bandwidth (GB/s), plot results

import cuda.tile as ct
import cupy as cp
import matplotlib.pyplot as plt
