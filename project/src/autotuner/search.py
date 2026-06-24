"""M1 - Config-Suchraum.

Reines Python, KEIN cuTile-Import, damit das Ganze auch lokal ohne GPU laeuft
und wir den Enumerator testen koennen bevor irgendwas auf der Spark kompiliert.

Hier drin: M1.1 (die Knoepfe) und M1.2 (Enumerator). Pruning und Ranking
(M1.3/M1.4) kommen als naechstes dran.

Ausfuehren als Selbsttest:   python -m autotuner.search   (aus project/src/)
"""

from dataclasses import dataclass, field

from .config import Config, DimType, ExecType
from .generate import generate_config
from .optimizer import Optimizer


def ceildiv(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# M1.1  -  Die Knoepfe / der Suchraum
# ---------------------------------------------------------------------------
# Bewusst klein gehalten. Die Werte stammen aus dem Pitch; ob sie auf der GB10
# wirklich alle Sinn ergeben, sehen wir spaetestens beim Pruning/Messen.

M_PRIM_CHOICES = [64, 128, 256]
N_PRIM_CHOICES = [64, 128, 256]
K_PRIM_CHOICES = [32, 64, 128]

M_L2_CHOICES = [2, 4, 8]
N_L2_CHOICES = [2, 4, 8]

# Die zwei Exec-Muster aus A05:
#   "A" -> m_l2/n_l2 laufen als PAR (Swizzling)        -> kernel_l2
#   "B" -> m_l2/n_l2 laufen als SEQ-Loops (strict)     -> kernel_l2_strict
VARIANT_CHOICES = ["A", "B"]


@dataclass
class SearchSpace:
    m_prim_choices: list = field(default_factory=lambda: list(M_PRIM_CHOICES))
    n_prim_choices: list = field(default_factory=lambda: list(N_PRIM_CHOICES))
    k_prim_choices: list = field(default_factory=lambda: list(K_PRIM_CHOICES))
    m_l2_choices:   list = field(default_factory=lambda: list(M_L2_CHOICES))
    n_l2_choices:   list = field(default_factory=lambda: list(N_L2_CHOICES))
    variants:       list = field(default_factory=lambda: list(VARIANT_CHOICES))

    def size(self):
        """Wie viele Kombinationen das (vor dem Pruning) sind."""
        return (len(self.m_prim_choices) * len(self.n_prim_choices) *
                len(self.k_prim_choices) * len(self.m_l2_choices) *
                len(self.n_l2_choices) * len(self.variants))


# Ein Kandidat ist nicht nur die Config, sondern auch die Knoepfe die dazu
# gefuehrt haben + die Original-M/N/K-Groessen. Die brauchen wir spaeter:
# - die Knoepfe fuer Codegen und Pruning
# - die Original-Groessen fuer die TFLOPS-Rechnung (NICHT die gepaddeten!)
@dataclass
class Candidate:
    config: Config
    variant: str          # "A" oder "B"
    m_prim: int
    n_prim: int
    k_prim: int
    m_l2: int
    n_l2: int
    # Original-Groessen der Kontraktion (ungepaddet)
    orig_m: int
    orig_n: int
    orig_k: int

    def label(self):
        return (f"{self.variant}: m_prim={self.m_prim} n_prim={self.n_prim} "
                f"k_prim={self.k_prim} m_l2={self.m_l2} n_l2={self.n_l2}")


# ---------------------------------------------------------------------------
# Kleine Helfer
# ---------------------------------------------------------------------------

def _classify_einsum(einsum_str):
    """Findet heraus, welcher Buchstabe M, N bzw. K ist, und welche Dims uebrig
    bleiben (Batch / C). Gleiche Logik wie generate_config, nur dass wir hier
    die Buchstaben behalten wollen.

    Liefert: (alle_dims_in_reihenfolge, m_char, n_char, k_char, batch_chars)
    """
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
            batch_chars.append(d)          # C-Typ (Batch)
        elif d in set_a and d in set_b:    # in beiden Inputs, nicht im Output
            k_chars.append(d)
        elif d in set_a and d in set_out:
            m_chars.append(d)
        else:
            n_chars.append(d)

    # M1 kann nur den einfachen Fall: genau eine M-, N- und K-Dim.
    # Der A06-Fall mit zwei Reduktionsdims (s und p) faellt hier bewusst raus,
    # das ist M4.
    if not (len(m_chars) == 1 and len(n_chars) == 1 and len(k_chars) == 1):
        raise NotImplementedError(
            f"M1 unterstuetzt nur genau ein M/N/K. Gefunden: "
            f"M={m_chars}, N={n_chars}, K={k_chars}. Mehrfach-K (z.B. A06) ist M4."
        )

    return all_dims, m_chars[0], n_chars[0], k_chars[0], batch_chars


def _split_tracked(opt, labels, target_label, outer_size, inner_size,
                   outer_label, inner_label):
    """split_dim auf dem Optimizer aufrufen UND unsere parallele labels-Liste
    mitfuehren, damit wir Dims immer ueber ihren Namen wiederfinden statt ueber
    Indizes (die sich bei jedem Split verschieben)."""
    idx = labels.index(target_label)
    opt.split_dim(idx, outer_size, inner_size)
    labels[idx] = outer_label
    labels.insert(idx + 1, inner_label)


# ---------------------------------------------------------------------------
# M1.2  -  Enumerator
# ---------------------------------------------------------------------------

def build_one_config(einsum_str, input_shapes, variant,
                     m_prim, n_prim, k_prim, m_l2, n_l2):
    """Baut genau eine fertige (executable) Config fuer die uebergebenen Knoepfe.

    Padding-Konvention: split_dim verlangt exakte Teilbarkeit. Krumme Groessen
    runden wir deshalb hoch auf das naechste Vielfache von prim*l2 (M/N) bzw.
    prim (K). Die Config-dim_sizes sind also die GEPADDETEN Groessen; den
    Ueberhang nullt spaeter der Kernel ueber PaddingMode.ZERO.
    """
    all_dims, m_char, n_char, k_char, batch_chars = _classify_einsum(einsum_str)

    # Original-Groessen einsammeln
    size_of = {}
    lhs = einsum_str.replace(" ", "").split("->")[0]
    in_a, in_b = lhs.split(",")
    for tensor_str, shape in zip((in_a, in_b), input_shapes):
        for c, s in zip(tensor_str, shape):
            size_of[c] = s

    orig_m, orig_n, orig_k = size_of[m_char], size_of[n_char], size_of[k_char]

    # gepaddete Groessen
    m_l2_outer = ceildiv(orig_m, m_prim * m_l2)
    n_l2_outer = ceildiv(orig_n, n_prim * n_l2)
    k_outer    = ceildiv(orig_k, k_prim)

    padded_m = m_l2_outer * m_l2 * m_prim
    padded_n = n_l2_outer * n_l2 * n_prim
    padded_k = k_outer * k_prim

    padded_size = dict(size_of)
    padded_size[m_char] = padded_m
    padded_size[n_char] = padded_n
    padded_size[k_char] = padded_k

    padded_shapes = [tuple(padded_size[c] for c in in_a),
                     tuple(padded_size[c] for c in in_b)]

    cfg = generate_config(einsum_str, padded_shapes)
    opt = Optimizer(cfg)

    # labels parallel zur Config fuehren (Reihenfolge = generate_config-Reihenfolge)
    labels = list(all_dims)

    # M -> [m_l2_outer, m_l2, m_prim]   (m_prim landet ganz rechts -> wird PRIM)
    _split_tracked(opt, labels, m_char, padded_m // m_prim, m_prim, "m_rest", "m_prim")
    _split_tracked(opt, labels, "m_rest", m_l2_outer, m_l2, "m_l2_outer", "m_l2")

    # N -> [n_l2_outer, n_l2, n_prim]
    _split_tracked(opt, labels, n_char, padded_n // n_prim, n_prim, "n_rest", "n_prim")
    _split_tracked(opt, labels, "n_rest", n_l2_outer, n_l2, "n_l2_outer", "n_l2")

    # K -> [k_outer, k_prim]
    _split_tracked(opt, labels, k_char, k_outer, k_prim, "k_outer", "k_prim")

    if variant == "A":
        # m_l2/n_l2 als PAR: make_executable macht den Rest (Reihenfolge +
        # exec_types). Es nimmt jeweils die rechteste M/N/K-Dim als PRIM, also
        # genau m_prim/n_prim/k_prim.
        opt.make_executable()

    elif variant == "B":
        # strict: m_l2/n_l2 als SEQ-Loops. Reihenfolge + exec_types von Hand,
        # so wie in A05 task4b_strict.
        target_order = (list(batch_chars) +
                        ["m_l2_outer", "n_l2_outer", "k_outer",
                         "m_l2", "n_l2",
                         "m_prim", "n_prim", "k_prim"])
        perm = [labels.index(lbl) for lbl in target_order]
        opt.permute_dims(perm)
        labels = target_order

        n_batch = len(batch_chars)
        cfg.exec_types = (
            [ExecType.PAR] * (n_batch + 2) +   # batch + m_l2_outer + n_l2_outer
            [ExecType.SEQ] * 3 +               # k_outer + m_l2 + n_l2
            [ExecType.PRIM] * 3                # m_prim + n_prim + k_prim
        )
        opt.verify()
    else:
        raise ValueError(f"unbekannte Variante: {variant!r}")

    return Candidate(
        config=cfg, variant=variant,
        m_prim=m_prim, n_prim=n_prim, k_prim=k_prim, m_l2=m_l2, n_l2=n_l2,
        orig_m=orig_m, orig_n=orig_n, orig_k=orig_k,
    )


def enumerate_candidates(einsum_str, input_shapes, space=None):
    """Alle gueltigen Kandidaten fuer eine Kontraktion aufzaehlen.

    Configs, die der Optimizer als ungueltig ablehnt (verify wirft), lassen wir
    einfach weg. Das ist noch KEIN Pruning - hier fliegt nur raus was strukturell
    gar nicht laufen kann.
    """
    if space is None:
        space = SearchSpace()

    candidates = []
    skipped = 0
    for variant in space.variants:
        for m_prim in space.m_prim_choices:
            for n_prim in space.n_prim_choices:
                for k_prim in space.k_prim_choices:
                    for m_l2 in space.m_l2_choices:
                        for n_l2 in space.n_l2_choices:
                            try:
                                cand = build_one_config(
                                    einsum_str, input_shapes, variant,
                                    m_prim, n_prim, k_prim, m_l2, n_l2)
                                candidates.append(cand)
                            except (ValueError, NotImplementedError):
                                skipped += 1
    return candidates, skipped


# ---------------------------------------------------------------------------
# Selbsttest (M1-Akzeptanztest): laeuft die A05-Hand-Config durch und ist sie
# im enumerierten Set drin?
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EINSUM = "cmk, ckn -> cmn"
    SHAPES = [(4, 4096, 4096), (4, 4096, 4096)]   # C=4, M=N=K=4096

    space = SearchSpace()
    print(f"Suchraum (ungeprunt): {space.size()} Kombinationen")

    cands, skipped = enumerate_candidates(EINSUM, SHAPES, space)
    print(f"enumeriert: {len(cands)} gueltig, {skipped} verworfen (verify/strukturell)")

    # die handoptimierte A05-Config: Variante A, 128/128/64, 8x8
    hand = [c for c in cands
            if c.variant == "A" and c.m_prim == 128 and c.n_prim == 128
            and c.k_prim == 64 and c.m_l2 == 8 and c.n_l2 == 8]

    if hand:
        print(f"OK - A05-Hand-Config gefunden: {hand[0].label()}")
        print(f"     dim_types : {[d.name for d in hand[0].config.dim_types]}")
        print(f"     exec_types: {[e.name for e in hand[0].config.exec_types]}")
        print(f"     dim_sizes : {hand[0].config.dim_sizes}")
    else:
        print("FEHLER: A05-Hand-Config NICHT im enumerierten Set!")
