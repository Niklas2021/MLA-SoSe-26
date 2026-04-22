# Task 1: FP32 vs FP16 Performance
# a) Two cuTile kernels: kernel_fp16 (FP16 inputs, FP32 accumulator)
#                        kernel_fp32 (FP32 inputs, FP32 accumulator)
#    Shape: A=(64,4096), B=(4096,64), C=(64,64)
#    Use ct.mma, single CTA, tile shape (64,64,64)
#    Verify both kernels via torch.allclose against torch.matmul
# b) Benchmark with triton.testing.do_bench, report runtimes and speedup of fp16 over fp32

import cuda.tile as ct
import cupy as cp
import torch
import triton
import triton.testing
