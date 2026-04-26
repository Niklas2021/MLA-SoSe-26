# implement two cuTile kernels that each compute a matrix multiplication 
# $A x B = C$ with `shape(A) = (64, 4096)`, `shape(B) = (4096, 64)` and `shape(C) = (64, 64)`. 
#
# 1. kernel_fp16: A and B are `FP16`, output `C` is `FP32`
# 2. kernel_fp32: A and B are `FP32`, output `C` is `FP32`
# use ct.mma

import cuda.tile as ct
import cupy as cp
import torch
import triton


@ct.kernel
def kernel_fp16(a, b, acc,
                m: ct.Constant[int],
                iterate: ct.Constant[int]):

    acc_tile = ct.full((m, m), fill_value=0, dtype=ct.float32)

    for i in range(iterate):
        a_tile = ct.load(a, index=(0, i), shape=(m, m))
        b_tile = ct.load(b, index=(i, 0), shape=(m, m))
        acc_tile = ct.mma(a_tile, b_tile, acc_tile)

    ct.store(acc, (0, 0), tile=acc_tile)    

@ct.kernel
def kernel_fp32(a, b, acc,
                m: ct.Constant[int],
                iterate: ct.Constant[int]):

    acc_tile = ct.full((m, m), fill_value=0, dtype=ct.float32)

    for i in range(iterate):
        a_tile = ct.load(a, index=(0, i), shape=(m, m))
        b_tile = ct.load(b, index=(i, 0), shape=(m, m))
        acc_tile = ct.mma(a_tile, b_tile, acc_tile)

    ct.store(acc, (0, 0), tile=acc_tile)   

def run_kernels():
    grid = (1,)
    m,n = 64, 4096
    iterate = n // m
    a_fp32 = torch.rand ((m,n), 
                            dtype=torch.float32,
                            device='cuda')
    b_fp32 = torch.rand ((n,m), 
                            dtype=torch.float32,
                            device='cuda')

    # FP16 per Cast
    a_fp16 = a_fp32.half()
    b_fp16 = b_fp32.half()

    acc = torch.zeros((m,m),
                                dtype=torch.float32,
                            device='cuda')

    torch.cuda.init()

    # FLOAT 32
    ct.launch(torch.cuda.current_stream(),
                grid,
                kernel_fp32,
                (a_fp32,b_fp32,acc, m, iterate))  


    print("FP32: torch.allclose verification: ", torch.allclose(acc, torch.matmul(a_fp32, b_fp32).float(), atol=1e-3, rtol=1e-3))

    acc = acc.zero_()

    #FLOAT 16
    ct.launch(torch.cuda.current_stream(),
                grid,
                kernel_fp16,
                (a_fp16,b_fp16,acc, m, iterate))    

    print("FP16: torch.allclose verification: ", torch.allclose(acc, torch.matmul(a_fp16, b_fp16).float(), atol=1e-3, rtol=1e-3))

    # b) Benchmarks                                                  
    def bench_32():
        ct.launch(torch.cuda.current_stream(),
                    grid,
                    kernel_fp32,
                    (a_fp32,b_fp32,acc, m, iterate))  

    def bench_16():
        ct.launch(torch.cuda.current_stream(),
                    grid,
                    kernel_fp16,
                    (a_fp16,b_fp16,acc, m, iterate))    

    score32 = triton.testing.do_bench(bench_32)
    score16 = triton.testing.do_bench(bench_16)

    print(f"\nBenchmark results (average runtime):")
    print(f"  FP32: {score32:.4f} ms")
    print(f"  FP16: {score16:.4f} ms")
    print(f"\nAnalysis:")
    faster = "FP32" if score32 < score16 else "FP16"
    ratio  = max(score32, score16) / min(score32, score16)
    print(f"  {faster} is faster by a factor of ~{ratio:.2f}x.")



if __name__ =="__main__":
    run_kernels()