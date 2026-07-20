import cuda.tile as ct
import cupy as cp
import torch
import triton


# Tile-Dims muessen Zweierpotenzen sein. Fuer beliebige GEMM-Groessen runden wir
# die Tile-Shape auf die naechste Zweierpotenz auf; die ueberzaehligen Zeilen/
# Spalten werden beim Laden per padding_mode=ZERO genullt und beim Store wieder
# verworfen (ct.store ignoriert OOB-Elemente). Die Tensoren bleiben real gross -
# kein Padding im Speicher.
def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


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
            a_tile = ct.load (A, index=(pid_e, pid_a, pid_b, k_i, l_i, 0, 0), shape =(1,1,1,1,1,x,y), padding_mode=ct.PaddingMode.ZERO)
            b_tile = ct.load (B, index=(pid_e, pid_c, k_i, l_i, 0, 0), shape =(1,1,1,1,y,z), padding_mode=ct.PaddingMode.ZERO)
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
            a_tile = ct.load (A, index=(pid_e, pid_a, b_i, k_i, l_i, 0, 0), shape =(1,1,1,1,1,x,y), padding_mode=ct.PaddingMode.ZERO)
            b_tile = ct.load (B, index=(pid_e, pid_c, k_i, l_i, 0, 0), shape =(1,1,1,1,y,z), padding_mode=ct.PaddingMode.ZERO)
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
                        shape =(1,1,1,1,l,x,y),
                        padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B,
                         index=(pid_e, pid_c, k_i, 0,0,0),
                         shape=(1,1,1,l,y,z),
                         padding_mode=ct.PaddingMode.ZERO)
        a_tile = ct.permute(a_tile, (0,1,2,3,5,4,6)) #e,a,b,k,x,l,y

        a_tile = ct.reshape(a_tile,(x, l*y))
        b_tile = ct.reshape(b_tile,(l*y, z))

        acc = ct.mma(a_tile, b_tile, acc)
    out = ct.reshape(acc, (1,1,1,1,x,z)).astype(ct.float16)
    ct.store(C, index=(pid_e, pid_a, pid_b, pid_c, 0,0), tile=out)

# --------------- Task 1 e)
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
            a_tile = ct.load (A, index=(0, pid_a, pid_b, k_i, l_i, 0, 0), shape =(e,1,1,1,1,x,y), padding_mode=ct.PaddingMode.ZERO)
            b_tile = ct.load (B, index=(0, pid_c, k_i, l_i, 0, 0), shape =(e,1,1,1,y,z), padding_mode=ct.PaddingMode.ZERO)
            a_tile = ct.reshape(a_tile, (e,x, y))
            b_tile = ct.reshape(b_tile, (e,y, z))
            acc = ct.mma(a_tile, b_tile, acc)
        out = ct.reshape(acc, (e,1,1,1,x,z)).astype(ct.float16)
        ct.store(C, index=(0, pid_a, pid_b, pid_c, 0,0), tile=out)


def run_kernels():
    e, a, b, c = 16, 8, 32, 64
    k, l       = 32, 16             # contraction dimensions
    x, y, z    = 32, 32, 16         # GEMM dimensions

    # GEMM-/Tile-Dims auf Zweierpotenz aufrunden (hier schon pow2 -> unveraendert).
    # Wird pro Kernel in den passenden Slot gegeben, Loop-/Grid-Dims bleiben real.
    xp, yp, zp = next_pow2(x), next_pow2(y), next_pow2(z)
    lp, ep     = next_pow2(l), next_pow2(e)

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
    args_b = (a_input, b_input, c_output, e,a,b,c,k,l, xp,yp,zp)
    ct.launch(torch.cuda.current_stream(), grid, b_contraction, args_b)

    ref = torch.einsum('eabklxy,ecklyz->eabcxz', a_input.float(), b_input.float()).half()
    print("b) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))
    ms_b = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid, b_contraction, args_b))
    print(f"Runtime: {ms_b:.3f} ms")

# Task 1 c)
    c_output = c_output.zero_() #start with empty ouput

    grid_c = (e*a*c, )

    args_c = (a_input, b_input, c_output, e,a,b,c,k,l, xp,yp,zp)
    ct.launch(torch.cuda.current_stream(), grid_c, c_contraction, args_c)

    print("c) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))
    ms_c = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_c, c_contraction, args_c))
    print(f"Runtime: {ms_c:.3f} ms")


# Task 1 d)
    c_output = c_output.zero_()

    grid_d = (e*a*c*b, )

    args_d = (a_input, b_input, c_output, e,a,b,c,k, lp, xp,yp,zp)
    ct.launch(torch.cuda.current_stream(), grid_d, d_contraction, args_d)

    print("d) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))
    ms_d = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_d, d_contraction, args_d))
    print(f"Runtime: {ms_d:.3f} ms")

# Task 1 e)
    c_output = c_output.zero_()

    grid_e = (a*c*b, )

    args_e = (a_input, b_input, c_output, ep, a,b,c,k,l, xp,yp,zp)
    ct.launch(torch.cuda.current_stream(), grid_e, e_contraction, args_e)

    print("e) Verification:", torch.allclose(c_output, ref, atol=1e-2, rtol=1e-2))
    ms_e = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_e, e_contraction, args_e))
    print(f"Runtime: {ms_e:.3f} ms")


def verify_arbitrary():
    # zeigt dass alle Kernels mit beliebigen (nicht-zweierpotenten) Groessen laufen
    print("\n=== Task 1 - arbitrary (non-power-of-2) sizes ===")
    e, a, b, c = 3, 2, 5, 3
    k, l       = 7, 6
    x, y, z    = 17, 13, 9

    a_in = torch.rand((e, a, b, k, l, x, y), dtype=torch.float16, device='cuda')
    b_in = torch.rand((e, c, k, l, y, z),    dtype=torch.float16, device='cuda')
    ref  = torch.einsum('eabklxy,ecklyz->eabcxz', a_in.float(), b_in.float()).half()

    xp, yp, zp = next_pow2(x), next_pow2(y), next_pow2(z)
    lp, ep     = next_pow2(l), next_pow2(e)

    def run_one(kernel, grid, tail, name):
        C = torch.zeros((e, a, b, c, x, z), dtype=torch.float16, device='cuda')
        ct.launch(torch.cuda.current_stream(), grid, kernel, (a_in, b_in, C) + tail)
        print(f"  {name}) verification:", torch.allclose(C, ref, atol=1e-2, rtol=1e-2))

    run_one(b_contraction, (e*a*b*c,), (e, a, b, c, k, l,  xp, yp, zp), "b")
    run_one(c_contraction, (e*a*c,),   (e, a, b, c, k, l,  xp, yp, zp), "c")
    run_one(d_contraction, (e*a*b*c,), (e, a, b, c, k, lp, xp, yp, zp), "d")
    run_one(e_contraction, (a*b*c,),   (ep, a, b, c, k, l, xp, yp, zp), "e")


def compare_bc_bd():
    def bench(e,a,b,c,k,l,x,y,z):
        a_input = torch.rand((e,a,b,k,l,x,y), dtype=torch.float16, device='cuda')
        b_input = torch.rand((e,c,k,l,y,z), dtype=torch.float16, device='cuda')
        c_output = torch.zeros((e,a,b,c,x,z), dtype=torch.float16, device='cuda')

        # Assert that all tensors will fit in memory (less than 32 GiB) first
        total_bytes = a_input.nbytes + b_input.nbytes + c_output.nbytes
        assert total_bytes < 32 * 1024**3, f"Too large: {total_bytes / 1024**3:.2f} GiB"

        xp, yp, zp = next_pow2(x), next_pow2(y), next_pow2(z)
        lp = next_pow2(l)

        grid_b = (e*a*b*c,)
        grid_c = (e*a*c,)
        grid_d = (e*a*b*c,)
        args_bc = (a_input, b_input, c_output, e,a,b,c,k,l, xp,yp,zp)
        args_d  = (a_input, b_input, c_output, e,a,b,c,k, lp, xp,yp,zp)
        ms_b = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_b, b_contraction, args_bc))
        ms_c = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_c, c_contraction, args_bc))
        ms_d = triton.testing.do_bench(lambda: ct.launch(torch.cuda.current_stream(), grid_d, d_contraction, args_d))
        return ms_b, ms_c, ms_d

    # test1: b) better than c)
    print("\nB VS C:")
    e,a,b,c,k,l,x,y,z = 2, 4, 64, 4, 8, 4, 32, 32, 16
    ms_b, ms_c, _ = bench(e,a,b,c,k,l,x,y,z)
    print("large b:")
    print(f"  b) {ms_b:.3f} ms c) {ms_c:.3f} ms  -> b) {'faster' if ms_b < ms_c else 'slower'}")

    # test2: c) better than b)
    e,a,b,c,k,l,x,y,z = 32, 32, 2, 32, 2, 4, 8, 8, 8
    ms_b, ms_c, _ = bench(e,a,b,c,k,l,x,y,z)
    print("large e,a,b:")
    print(f"  b) {ms_b:.3f} ms c) {ms_c:.3f} ms  -> c) {'faster' if ms_c < ms_b else 'slower'}")

    # test3: b) better than d)
    print("\nB VS D:")
    e,a,b,c,k,l,x,y,z = 4, 8, 8, 8, 16, 1, 16,16,16
    print("small GEMM dimensions:")
    ms_b, _, ms_d = bench(e,a,b,c,k,l,x,y,z)
    print(f"  b) {ms_b:.3f} ms d) {ms_d:.3f} ms  -> b) {'faster' if ms_b < ms_d else 'slower'}")

    # test4: d) better than b)
    e,a,b,c,k,l,x,y,z = 2, 4, 8, 4, 4, 32, 32, 32, 16
    print("large l:")
    ms_b, _, ms_d = bench(e,a,b,c,k,l,x,y,z)
    print(f"  b) {ms_b:.3f} ms d) {ms_d:.3f} ms  -> d) {'faster' if ms_d < ms_b else 'slower'}")

if __name__ == "__main__":
    run_kernels()
    verify_arbitrary()
    compare_bc_bd()
