from dataclasses import dataclass, field


@dataclass
class Einsum:
    """Ground Truth, aus einsum string geparst (hängt nur von einsum und input shapes ab)
    ist gleich für alle Kandidaten
    """
    in_a: str
    in_b: str
    out: str
    orig_m: int             # ungepaddete originalgrößen (der prim-Dims)
    orig_n: int
    orig_k: int
    m_char: str             # die als prim getilte M/N/K-Dim (innerste, stride 1)
    n_char: str
    k_char: str
    batch_chars: list       # geteilter Batch (C-Typ, in A, B und out)
    all_dims: list          #  Buchstaben in Auftritts-Reihenfolge
    size_of: dict           # char -> int (Original-Größen)
    # ab hier fuer den Mehrdim-Fall (A06). Single-Fall: Listen einelementig, Extras leer.
    m_chars: list = field(default_factory=list)
    n_chars: list = field(default_factory=list)
    k_chars: list = field(default_factory=list)
    extra_m_chars: list = field(default_factory=list)   # weitere M-Dims -> A-seitiger PAR-Batch
    extra_n_chars: list = field(default_factory=list)   # weitere N-Dims -> B-seitiger PAR-Batch
    seq_k_chars: list = field(default_factory=list)      # weitere K-Dims -> SEQ-Reduktionsloop

    def is_multi(self):
        return bool(self.extra_m_chars or self.extra_n_chars or self.seq_k_chars)


def _innermost(tensor_str, chars):
    # letzter (= innerster, stride 1) Buchstabe aus chars im Tensor-String
    for c in reversed(tensor_str):
        if c in chars:
            return c
    return None


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

    if not (m_chars and n_chars and k_chars):
        raise NotImplementedError(
            f"brauche mind. je ein M/N/K. Gefunden M={m_chars} N={n_chars} K={k_chars}")

    size_of = {}
    for tensor_str, shape in zip((in_a, in_b), input_shapes):
        for c, s in zip(tensor_str, shape):
            size_of[c] = s

    # prim = die innerste (stride 1) Dim ihrer Sorte im jeweiligen Tensor. Fuer den
    # Single-Fall ist das schlicht die einzige. Der Rest wird PAR-Batch bzw. SEQ-K.
    m_char = _innermost(in_a, set(m_chars))
    n_char = _innermost(in_b, set(n_chars))
    k_char = _innermost(in_a, set(k_chars))
    # prim-K muss in beiden Inputs die innerste K-Dim sein, sonst braeuchte der
    # mma-Load erst eine Fusion/Transponierung -> nicht abgedeckt.
    if _innermost(in_b, set(k_chars)) != k_char:
        raise NotImplementedError(
            f"prim-K nicht in A und B innerste K-Dim (A={k_char}, "
            f"B={_innermost(in_b, set(k_chars))}) -> Fusion/Transpose noetig")

    extra_m_chars = [c for c in m_chars if c != m_char]
    extra_n_chars = [c for c in n_chars if c != n_char]
    seq_k_chars = [c for c in k_chars if c != k_char]

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
        m_chars=m_chars,
        n_chars=n_chars,
        k_chars=k_chars,
        extra_m_chars=extra_m_chars,
        extra_n_chars=extra_n_chars,
        seq_k_chars=seq_k_chars,
    )


if __name__ == "__main__":
    # A05: ein M/N/K, geteilter Batch c
    e = parse_einsum("cmk, ckn -> cmn", [(4, 4096, 4096), (4, 4096, 4096)])
    print("A05:", "prim m/n/k =", e.m_char, e.n_char, e.k_char,
          "batch =", e.batch_chars, "multi =", e.is_multi())
    assert (e.m_char, e.n_char, e.k_char) == ("m", "n", "k")
    assert e.batch_chars == ["c"] and not e.is_multi()

    # A06: M={a,c,x} N={b,y} K={s,p}, kein geteilter Batch
    e = parse_einsum("acspx, bspy -> abcyx",
                     [(4, 3, 64, 64, 1536), (4, 64, 64, 1152)])
    print("A06:", "prim m/n/k =", e.m_char, e.n_char, e.k_char,
          "| extra_m =", e.extra_m_chars, "extra_n =", e.extra_n_chars,
          "seq_k =", e.seq_k_chars, "| multi =", e.is_multi())
    assert (e.m_char, e.n_char, e.k_char) == ("x", "y", "p")     # innerste Dims
    assert e.extra_m_chars == ["a", "c"] and e.extra_n_chars == ["b"]
    assert e.seq_k_chars == ["s"] and e.batch_chars == []
    assert (e.orig_m, e.orig_n, e.orig_k) == (1536, 1152, 64)
    print("ok")
