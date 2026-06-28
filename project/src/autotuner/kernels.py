# Ein generischer cuTile-Kernel pro Variante, Tile-Groessen als ct.Constant
# (JIT spezialisiert pro Wert). Vorlage: kernel_l2 / kernel_l2_strict aus A05.
# Laeuft nur auf der GPU.
import torch
import cuda.tile as ct


def ceildiv(a, b):
    return (a + b - 1) // b


# Variante A: m_l2/n_l2 ueber die bid verteilt (swizzle)
@ct.kernel
def matmul_variant_a(A, B, C,
                     M_PRIM: ct.Constant[int],
                     N_PRIM: ct.Constant[int],
                     K_PRIM: ct.Constant[int],
                     M_L2: ct.Constant[int],
                     N_L2: ct.Constant[int],
                     num_m_l2_outer: ct.Constant[int],
                     num_n_l2_outer: ct.Constant[int],
                     num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)
    n_l2_idx = pid % N_L2
    pid = pid // N_L2
    m_l2_idx = pid % M_L2
    pid = pid // M_L2
    n_l2_outer_idx = pid % num_n_l2_outer
    pid = pid // num_n_l2_outer
    m_l2_outer_idx = pid % num_m_l2_outer
    pid = pid // num_m_l2_outer
    c_idx = pid

    m_block = m_l2_outer_idx * M_L2 + m_l2_idx
    n_block = n_l2_outer_idx * N_L2 + n_l2_idx

    acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)
    for k_it in range(num_k_outer):
        a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                         shape=(1, M_PRIM, K_PRIM), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                         shape=(1, K_PRIM, N_PRIM), padding_mode=ct.PaddingMode.ZERO)
        a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
        b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
        acc = ct.mma(a_tile, b_tile, acc)

    out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
    ct.store(C, index=(c_idx, m_block, n_block), tile=out)


def run_variant_a(A, B, m_prim, n_prim, k_prim, m_l2, n_l2):
    c_size, m_size, k_size = A.shape
    _, _, n_size = B.shape
    num_m_l2_outer = ceildiv(m_size, m_prim * m_l2)
    num_n_l2_outer = ceildiv(n_size, n_prim * n_l2)
    num_k_outer = ceildiv(k_size, k_prim)
    m_pad = num_m_l2_outer * m_l2 * m_prim
    n_pad = num_n_l2_outer * n_l2 * n_prim

    C_pad = torch.zeros((c_size, m_pad, n_pad), dtype=torch.float16, device="cuda")
    grid = (c_size * num_m_l2_outer * num_n_l2_outer * m_l2 * n_l2,)
    ct.launch(torch.cuda.current_stream(), grid, matmul_variant_a,
              (A, B, C_pad, m_prim, n_prim, k_prim, m_l2, n_l2,
               num_m_l2_outer, num_n_l2_outer, num_k_outer))
    return C_pad[:, :m_size, :n_size]


# Variante B (strict): m_l2/n_l2 als Loops im CTA -> weniger CTAs
@ct.kernel
def matmul_variant_b(A, B, C,
                     M_PRIM: ct.Constant[int],
                     N_PRIM: ct.Constant[int],
                     K_PRIM: ct.Constant[int],
                     M_L2: ct.Constant[int],
                     N_L2: ct.Constant[int],
                     num_m_l2_outer: ct.Constant[int],
                     num_n_l2_outer: ct.Constant[int],
                     num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)
    n_l2_outer_idx = pid % num_n_l2_outer
    pid = pid // num_n_l2_outer
    m_l2_outer_idx = pid % num_m_l2_outer
    pid = pid // num_m_l2_outer
    c_idx = pid

    for m_l2_it in range(M_L2):
        for n_l2_it in range(N_L2):
            m_block = m_l2_outer_idx * M_L2 + m_l2_it
            n_block = n_l2_outer_idx * N_L2 + n_l2_it
            acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)
            for k_it in range(num_k_outer):
                a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                                 shape=(1, M_PRIM, K_PRIM), padding_mode=ct.PaddingMode.ZERO)
                b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                                 shape=(1, K_PRIM, N_PRIM), padding_mode=ct.PaddingMode.ZERO)
                a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
                b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
                acc = ct.mma(a_tile, b_tile, acc)
            out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
            ct.store(C, index=(c_idx, m_block, n_block), tile=out)


def run_variant_b(A, B, m_prim, n_prim, k_prim, m_l2, n_l2):
    c_size, m_size, k_size = A.shape
    _, _, n_size = B.shape
    num_m_l2_outer = ceildiv(m_size, m_prim * m_l2)
    num_n_l2_outer = ceildiv(n_size, n_prim * n_l2)
    num_k_outer = ceildiv(k_size, k_prim)
    m_pad = num_m_l2_outer * m_l2 * m_prim
    n_pad = num_n_l2_outer * n_l2 * n_prim

    C_pad = torch.zeros((c_size, m_pad, n_pad), dtype=torch.float16, device="cuda")
    grid = (c_size * num_m_l2_outer * num_n_l2_outer,)
    ct.launch(torch.cuda.current_stream(), grid, matmul_variant_b,
              (A, B, C_pad, m_prim, n_prim, k_prim, m_l2, n_l2,
               num_m_l2_outer, num_n_l2_outer, num_k_outer))
    return C_pad[:, :m_size, :n_size]


def run_candidate(cand, A, B):
    if cand.variant == "A":
        return run_variant_a(A, B, cand.m_prim, cand.n_prim, cand.k_prim, cand.m_l2, cand.n_l2)
    elif cand.variant == "B":
        return run_variant_b(A, B, cand.m_prim, cand.n_prim, cand.k_prim, cand.m_l2, cand.n_l2)
    raise ValueError(f"unbekannte Variante: {cand.variant!r}")
