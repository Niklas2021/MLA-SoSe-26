import cuda.tile as ct
import cupy as cp
import torch
import triton

# ------------------- Task1 b)
@ct.kernel 
def b_contraction(A, B, C, 
                  e: ct.Constant[int],
                  a: ct.Constant[int],
                  b: ct.Constant[int],
                  c: ct.Constant[int],
                  k: ct.Constant[int],
                  l: ct.Constant[int],
                  x: ct.Constant[int],
                  y: ct.Constant[int],
                  z: ct.Constant[int]):
    pid = ct.bid(0)

    # Parallelize over: e,a,b,c
    #decompose pid -> (e,a,b,c)
    pid_c = pid % c
    pid = pid // c

    pid_b = pid % b
    pid = pid // b

    pid_a = pid % a
    pid = pid // a

    pid_e = pid

    acc = ct.zeros((x, z), dtype=ct.float32)

    for k_i in range(k):
        for l_i in range(l):
            a_tile = ct.load (A, index=(pid_e, pid_a, pid_b, k_i, l_i, 0, 0), shape =(1,1,1,1,1,x,y))
            b_tile = ct.load (B, index=(pid_e, pid_c, k_i, l_i, 0, 0), shape =(1,1,1,1,y,z))
            a_tile = ct.reshape(a_tile, (x, y))
            b_tile = ct.reshape(b_tile, (y, z))
            acc += ct.matmul(a_tile, b_tile)

    out = ct.reshape(acc, (1,1,1,1,x,z)).astype(ct.float16)
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c,0,0), tile=out)

# ------------------- Task 1 c)
# eabklxy, ecklyz -> eabcxz, GEMM dimansions: x,y,z
# Sequentialize all other K-dimensions, as well as the b dimension. Parallelize the remaining dimensions
@ct.kernel 
def c_contraction(A, B, C, 
                  e: ct.Constant[int],
                  a: ct.Constant[int],
                  b: ct.Constant[int],
                  c: ct.Constant[int],
                  k: ct.Constant[int],
                  l: ct.Constant[int],
                  x: ct.Constant[int],
                  y: ct.Constant[int],
                  z: ct.Constant[int]):
    pid = ct.bid(0)

    # Parallelize over: e,a,c
    #decompose pid -> (e,a,c)
    pid_c = pid % c
    pid = pid // c

    pid_a = pid % a
    pid = pid // a

    pid_e = pid

#eabklxy, ecklyz -> eabcxz
    for b_i in range(b):
      acc = ct.zeros((x, z), dtype=ct.float32)
      for k_i in range(k):
        for l_i in range(l):
            a_tile = ct.load (A, index=(pid_e, pid_a, b_i, k_i, l_i, 0, 0), shape =(1,1,1,1,1,x,y))
            b_tile = ct.load (B, index=(pid_e, pid_c, k_i, l_i, 0, 0), shape =(1,1,1,1,y,z))
            a_tile = ct.reshape(a_tile, (x, y))
            b_tile = ct.reshape(b_tile, (y, z))
            acc += ct.matmul(a_tile, b_tile)

      out = ct.reshape(acc, (1,1,1,1,x,z)).astype(ct.float16)
      ct.store(C, index=(pid_e, pid_a, b_i, pid_c,0,0), tile=out)


# --------------- Task 1 d)
# contraction eabklxy, ecklyz -> eabcxz. 
# GEMM dims: xyzl, 
# by permuting the input tiles of the ct.mma instruction,
# as well as reshaping so that y and l are merged.
@ct.kernel 
def d_contraction(A, B, C, 
                  e: ct.Constant[int],
                  a: ct.Constant[int],
                  b: ct.Constant[int],
                  c: ct.Constant[int],
                  k: ct.Constant[int],
                  l: ct.Constant[int],
                  x: ct.Constant[int],
                  y: ct.Constant[int],
                  z: ct.Constant[int]):
    pid = ct.bid(0)

    # Parallelize over: e,a,b,c
    #decompose pid -> (e,a,b,c)
    pid_c = pid % c
    pid = pid // c

    pid_b = pid % b
    pid = pid // b

    pid_a = pid % a
    pid = pid // a

    pid_e = pid

    acc = ct.zeros((x, z), dtype=ct.float32)

    # Serialize: contraction dims: k,l,y ABER OHNE l also --- k
    
    for k_i in range(k):
        # GEMM: xyzl (merge y & l)
        a_tile =ct.load(A, 
                        index=(pid_e, pid_a, pid_b, k_i, 0, 0, 0), 
                        shape =(1,1,1,1,l,x,y))
        b_tile = ct.load(B,
                         index=(pid_e, pid_c, k_i, 0,0,0),
                         shape=(1,1,1,l,y,z))
        a_tile = ct.permute(a_tile, (0,1,2,3,5,4,6)) #e,a,b,k,x,l,y

        a_tile = ct.reshape(a_tile,(x, l*y))
        b_tile = ct.reshape(b_tile,(l*y, z))

        acc = ct.mma(a_tile, b_tile, acc)
    out = ct.reshape(acc, (1,1,1,1,x,z)).astype(ct.float16)
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, 0,0), tile=out)

# eabklxy, ecklyz -> eabcxz
# GEMM dims: exyz, meaning that you perform a 3D ct.mma inside the kernel. 
# Sequentialize all other K-dimensions, parallelize the remaining dimensions.
@ct.kernel 
def e_contraction(A, B, C, 
                  e: ct.Constant[int],
                  a: ct.Constant[int],
                  b: ct.Constant[int],
                  c: ct.Constant[int],
                  k: ct.Constant[int],
                  l: ct.Constant[int],
                  x: ct.Constant[int],
                  y: ct.Constant[int],
                  z: ct.Constant[int]):
        pid = ct.bid(0)

        # Parallelize over: a,b,c
        #decompose pid -> (a,b,c)
        pid_c = pid % c
        pid = pid // c

        pid_b = pid % b
        pid = pid // b

        pid_a = pid % a
       # pid = pid // a

        #pid_e = pid

        acc = ct.zeros((e, x, z), dtype=ct.float32)

        # Serialize: contraction dims: k,l,y ABER OHNE l also --- k
        
        for k_i in range(k):
         for l_i in range(l):
            a_tile = ct.load (A, index=(0, pid_a, pid_b, k_i, l_i, 0, 0), shape =(e,1,1,1,1,x,y))
            b_tile = ct.load (B, index=(0, pid_c, k_i, l_i, 0, 0), shape =(e,1,1,1,y,z))
            a_tile = ct.reshape(a_tile, (x, y))
            b_tile = ct.reshape(b_tile, (y, z))
            acc = ct.mma(a_tile, b_tile, acc)
        out = ct.reshape(acc, (e,1,1,1,x,z)).astype(ct.float16)
        ct.store(C, index=(0, pid_a, pid_b, pid_c, 0,0), tile=out)


def run_kernels():
    e, a, b, c = 16, 8, 32, 64
    k, l       = 32, 16             # contraction dimensions
    x, y, z    = 32, 32, 16         # GEMM dimensions

    # "Use FP16 data type for tensor inputs and outputs, accumulate in FP32"
    a_input = torch.rand ((e, a, b, k, l, x, y),
                    dtype=torch.float16,
                     device='cuda' )
    b_input = torch.rand ((e,c,k,l,y,z),
                    dtype=torch.float16,
                     device='cuda' )
    c_output = torch.rand ((e,a,b,c,x,z),
                    dtype=torch.float16,
                     device='cuda' )
    
    grid = (e*a*b*c, )
    
    # Assert that all tensors will fit in memory (less than 32 GiB) first
    total_bytes = a_input.nbytes + b_input.nbytes + c_output.nbytes
    assert total_bytes < 32 * 1024**3, f"Too large: {total_bytes / 1024**3:.2f} GiB"

# Task 1 b)
    ct.launch(torch.cuda.current_stream(),
                    grid,
                    b_contraction,
                    (a_input, b_input, c_output,     #input/output tensors
                    e,a,b,c,k,l,x,y,z))             #dimensions
    
    ref = torch.einsum('eabklxy,ecklyz->eabcxz', a_input.float(), b_input.float()).half()
    print("b) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))

# Task 1 c)
    c_output = c_output.zero_() #start with empty ouput

    grid_c = (e*a*c, )

    ct.launch(torch.cuda.current_stream(),
                    grid_c,
                    c_contraction,
                    (a_input, b_input, c_output,   
                    e,a,b,c,k,l,x,y,z))            
    
    print("c) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))


# Task 1 d)
    c_output = c_output.zero_() #start with empty ouput

    grid_d = (e*a*c*b, )

    ct.launch(torch.cuda.current_stream(),
                    grid_d,
                    d_contraction,
                    (a_input, b_input, c_output,   
                    e,a,b,c,k,l,x,y,z))            
    
    print("d) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))

# Task 1 e)
    c_output = c_output.zero_() 

    grid_e = (a*c*b, )

    ct.launch(torch.cuda.current_stream(),
                    grid_e,
                    e_contraction,
                    (a_input, b_input, c_output,   
                    e,a,b,c,k,l,x,y,z))            
    
    print("e) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    run_kernels()
