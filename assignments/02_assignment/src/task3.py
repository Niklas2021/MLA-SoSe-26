# Task 3: 4D Tensor Elementwise Addition
# a) Two cuTile kernels for A + B on tensors (M, N, K, L):
#    - Kernel 1: tile covers (K, L), parallelize over (M, N)
#    - Kernel 2: tile covers (M, N), parallelize over (K, L)
# b) Benchmark both with triton.testing.do_bench; M=16, N=128, K=16, L=128

import cuda.tile as ct
import cupy as cp
import torch
import triton.testing

@ct.kernel
def sum_kl (a_tensor, b_tensor,  c_tensor, 
            k: ct.Constant[int],
            l: ct.Constant[int]):
    
    pid_m = ct.bid(0)   # index on M-axis
    pid_n = ct.bid(1)   # index on N-axis

    a_tile = ct.load(a_tensor, index = (pid_m, pid_n, 0, 0), shape = (1,1,k,l))
    b_tile = ct.load(b_tensor, index = (pid_m, pid_n, 0, 0), shape = (1,1,k,l))
    
    #c_tile = a_tile + b_tile
    c_tile = ct.add(a_tile, b_tile)
    
    ct.store(c_tensor,  (pid_m, pid_n, 0, 0), tile=c_tile)


@ct.kernel
def sum_mn (a_tensor, b_tensor,  c_tensor, 
            m: ct.Constant[int],
            n: ct.Constant[int]):

    pid_k = ct.bid(0)   # index on K-axis
    pid_l = ct.bid(1)   # index on L-axis

    a_tile = ct.load(a_tensor, index = (0,0, pid_k, pid_l), shape = (m,n,1,1))
    b_tile = ct.load(b_tensor, index = (0,0,pid_k, pid_l), shape = (m,n,1,1))
    
    c_tile = ct.add(a_tile, b_tile)
    
    ct.store(c_tensor,  (0,0, pid_k, pid_l), tile=c_tile)


def run_kernels():
    # set dimension sizes
    m, n, k, l = 16, 128, 16, 128

    # torch.rand: Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
    a_tensor = torch.rand ((m,n,k,l), 
                        dtype=torch.float16,
                        device='cuda')
    b_tensor = torch.rand ((m,n,k,l), 
                        dtype=torch.float16,
                        device='cuda')

    c_tensor = torch.zeros((m,n,k,l),
                            dtype=torch.float16,
                        device='cuda')

    # Kernel 1
    ct.launch(torch.cuda.current_stream().cuda_stream,
                                    (m,n,1), # grid size: parallelize over M,N
                                    sum_kl,
                                    (a_tensor, b_tensor, c_tensor, k,l))

    print("1) torch.allclose verification: ", torch.allclose(c_tensor, a_tensor + b_tensor))

    c_tensor.zero_()

    # Kernel 2
    ct.launch(torch.cuda.current_stream().cuda_stream,
                                    (k,l,1), # grid size: parallelize over K,L
                                    sum_mn,
                                    (a_tensor, b_tensor, c_tensor, m, n))


    print("2) torch.allclose verification: ", torch.allclose(c_tensor, a_tensor + b_tensor))


    # b) Benchmarks                                                  
    def bench_kl():
        ct.launch(torch.cuda.current_stream().cuda_stream,
                  (m, n, 1),
                  sum_kl,
                  (a_tensor, b_tensor, c_tensor, k, l))

    def bench_mn():
        ct.launch(torch.cuda.current_stream().cuda_stream,
                  (k, l, 1),
                  sum_mn,
                  (a_tensor, b_tensor, c_tensor, m, n))

    ms_kl = triton.testing.do_bench(bench_kl)
    ms_mn = triton.testing.do_bench(bench_mn)

    print(f"\nBenchmark results (average runtime):")
    print(f"  sum_kl  (tile over K,L  | parallelize over M,N): {ms_kl:.4f} ms")
    print(f"  sum_mn  (tile over M,N  | parallelize over K,L): {ms_mn:.4f} ms")
    print(f"\nAnalysis:")
    faster = "sum_kl" if ms_kl < ms_mn else "sum_mn"
    ratio  = max(ms_kl, ms_mn) / min(ms_kl, ms_mn)
    print(f"  {faster} is faster by a factor of ~{ratio:.2f}x.")



if __name__ == "__main__":
    run_kernels()
