from dataclasses import dataclass


@dataclass
class Einsum:
    """Ground Truth, aus einsum string geparst (hängt nur von einsum und input shapes ab)
    ist gleich für alle Kandidaten
    """
    in_a: str               
    in_b: str               
    out: str             
    orig_m: int             # ungepaddete originalgrößen
    orig_n: int
    orig_k: int
    m_char: str             
    n_char: str
    k_char: str
    batch_chars: list     
    all_dims: list          #  Buchstaben in Auftritts-Reihenfolge
    size_of: dict           # char -> int (Original-Größen)


def parse_einsum(einsum_str, input_shapes):
    lhs, rhs = einsum_str.replace(" ", "").split("->")
    in_a, in_b = lhs.split(",")
    out = rhs

    all_dims = []
    for s in (in_a, in_b, out):
        for c in s:
            if c not in all_dims:
                all_dims.append(c)

    set_a, set_b, set_out = set(in_a), set(in_b), set(out)
    m_chars, n_chars, k_chars, batch_chars = [], [], [], []
    for d in all_dims:
        if d in set_a and d in set_b and d in set_out:
            batch_chars.append(d)
        elif d in set_a and d in set_b:
            k_chars.append(d)
        elif d in set_a and d in set_out:
            m_chars.append(d)
        else:
            n_chars.append(d)

    # nur ein M/N/K. Mehrfach-K (A06) kann das hier nicht -> M4
    if not (len(m_chars) == 1 and len(n_chars) == 1 and len(k_chars) == 1):
        raise NotImplementedError(
            f"nur ein M/N/K. Gefunden M={m_chars} N={n_chars} K={k_chars}")

    size_of = {}
    for tensor_str, shape in zip((in_a, in_b), input_shapes):
        for c, s in zip(tensor_str, shape):
            size_of[c] = s

    m_char, n_char, k_char = m_chars[0], n_chars[0], k_chars[0]
    orig_m, orig_n, orig_k = size_of[m_char], size_of[n_char], size_of[k_char]

    return Einsum(
        in_a=in_a, 
        in_b=in_b, 
        out=out,
        orig_m=orig_m, 
        orig_n=orig_n, 
        orig_k=orig_k,
        m_char=m_char, 
        n_char=n_char, 
        k_char=k_char,
        batch_chars=batch_chars,
        all_dims=all_dims,
        size_of=size_of,
    )