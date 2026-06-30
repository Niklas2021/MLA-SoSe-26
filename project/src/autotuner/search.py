# Config-Suchraum: enumerieren, prunen, ranken. Reines Python, kein cuTile,
# damit man es ohne GPU testen kann.
from dataclasses import dataclass, field

from .config import Config, DimType, ExecType
from .generate import generate_config
from .optimizer import Optimizer
from .einsum_parser import parse_einsum


def ceildiv(a, b):
    return (a + b - 1) // b


# die Knoepfe (aus dem Pitch)
M_PRIM_CHOICES = [64, 128, 256]
N_PRIM_CHOICES = [64, 128, 256]
K_PRIM_CHOICES = [32, 64, 128]
M_L2_CHOICES = [2, 4, 8]
N_L2_CHOICES = [2, 4, 8]
VARIANT_CHOICES = ["A", "B"]   # A = m_l2/n_l2 als PAR (swizzle), B = als SEQ-Loops


@dataclass
class SearchSpace:
    m_prim_choices: list = field(default_factory=lambda: list(M_PRIM_CHOICES))
    n_prim_choices: list = field(default_factory=lambda: list(N_PRIM_CHOICES))
    k_prim_choices: list = field(default_factory=lambda: list(K_PRIM_CHOICES))
    m_l2_choices:   list = field(default_factory=lambda: list(M_L2_CHOICES))
    n_l2_choices:   list = field(default_factory=lambda: list(N_L2_CHOICES))
    variants:       list = field(default_factory=lambda: list(VARIANT_CHOICES))

    def size(self):
        return (len(self.m_prim_choices) * len(self.n_prim_choices) *
                len(self.k_prim_choices) * len(self.m_l2_choices) *
                len(self.n_l2_choices) * len(self.variants))


@dataclass
class Candidate:
    config: Config
    variant: str
    m_prim: int
    n_prim: int
    k_prim: int
    m_l2: int
    n_l2: int
    orig_m: int          # ungepaddet, fuer die TFLOPS-Rechnung
    orig_n: int
    orig_k: int
    padded_m: int        # so wie es in der Config steht
    padded_n: int
    padded_k: int

    def label(self):
        return (f"{self.variant}: m_prim={self.m_prim} n_prim={self.n_prim} "
                f"k_prim={self.k_prim} m_l2={self.m_l2} n_l2={self.n_l2}")


def _split_tracked(opt, labels, target_label, outer_size, inner_size,
                   outer_label, inner_label):
    # split_dim aufrufen und die labels-Liste mitfuehren, damit wir Dims ueber
    # ihren Namen finden statt ueber Indizes (die verschieben sich beim Split).
    idx = labels.index(target_label)
    opt.split_dim(idx, outer_size, inner_size)
    labels[idx] = outer_label
    labels.insert(idx + 1, inner_label)


def build_one_config(einsum_props, variant,
                     m_prim, n_prim, k_prim, m_l2, n_l2):
    # split_dim will exakte Teilbarkeit, also runden wir krumme Groessen hoch.
    # dim_sizes sind damit gepaddet, der Ueberhang wird im Kernel genullt.
    m_l2_outer = ceildiv(einsum_props.orig_m, m_prim * m_l2)
    n_l2_outer = ceildiv(einsum_props.orig_n, n_prim * n_l2)
    k_outer = ceildiv(einsum_props.orig_k, k_prim)

    padded_m = m_l2_outer * m_l2 * m_prim
    padded_n = n_l2_outer * n_l2 * n_prim
    padded_k = k_outer * k_prim

    # Kopie, damit die invariante size_of der einsum_props NICHT mutiert wird.
    padded_size = {**einsum_props.size_of,
                   einsum_props.m_char: padded_m,
                   einsum_props.n_char: padded_n,
                   einsum_props.k_char: padded_k}

    cfg = generate_config(einsum_props, padded_size)
    opt = Optimizer(cfg)
    labels = list(einsum_props.all_dims)

    # M/N -> l2_outer, l2, prim   (prim ganz rechts -> wird PRIM); K -> outer, prim
    _split_tracked(opt, labels, einsum_props.m_char, padded_m // m_prim, m_prim, "m_rest", "m_prim")
    _split_tracked(opt, labels, "m_rest", m_l2_outer, m_l2, "m_l2_outer", "m_l2")
    _split_tracked(opt, labels, einsum_props.n_char, padded_n // n_prim, n_prim, "n_rest", "n_prim")
    _split_tracked(opt, labels, "n_rest", n_l2_outer, n_l2, "n_l2_outer", "n_l2")
    _split_tracked(opt, labels, einsum_props.k_char, k_outer, k_prim, "k_outer", "k_prim")

    if variant == "A":
        opt.make_executable()
    elif variant == "B":
        # strict wie A05 task4b_strict: m_l2/n_l2 als SEQ
        target_order = (list(einsum_props.batch_chars) +
                        ["m_l2_outer", "n_l2_outer", "k_outer",
                         "m_l2", "n_l2", "m_prim", "n_prim", "k_prim"])
        opt.permute_dims([labels.index(lbl) for lbl in target_order])
        labels = target_order
        n_batch = len(einsum_props.batch_chars)
        cfg.exec_types = ([ExecType.PAR] * (n_batch + 2) +
                          [ExecType.SEQ] * 3 + [ExecType.PRIM] * 3)
        opt.verify()
    else:
        raise ValueError(f"unbekannte Variante: {variant!r}")

    return Candidate(
        config=cfg, variant=variant,
        m_prim=m_prim, n_prim=n_prim, k_prim=k_prim, m_l2=m_l2, n_l2=n_l2,
        orig_m=einsum_props.orig_m, orig_n=einsum_props.orig_n, orig_k=einsum_props.orig_k,
        padded_m=padded_m, padded_n=padded_n, padded_k=padded_k,
    )


def enumerate_candidates(einsum_str, input_shapes, space=None):
    # ungueltige Configs (verify wirft) fallen raus. Noch kein Pruning.
    if space is None:
        space = SearchSpace()
    # einmal parsen -- die Ground Truth ist für alle Kandidaten gleich
    einsum_props = parse_einsum(einsum_str, input_shapes)
    candidates = []
    skipped = 0
    for variant in space.variants:
        for m_prim in space.m_prim_choices:
            for n_prim in space.n_prim_choices:
                for k_prim in space.k_prim_choices:
                    for m_l2 in space.m_l2_choices:
                        for n_l2 in space.n_l2_choices:
                            try:
                                candidates.append(build_one_config(
                                    einsum_props, variant,
                                    m_prim, n_prim, k_prim, m_l2, n_l2))
                            except (ValueError, NotImplementedError):
                                skipped += 1
    return candidates, skipped


# --- Pruning ---
# Heuristik, kein Beweis. SMEM haengt nur an den Prim-Groessen, nicht an l2/Variante.
MMA_ALIGN = 16
DEFAULT_BUFFER_STAGES = 2     # double buffering angenommen
DEFAULT_REG_FRACTION = 0.5    # max Akku-Anteil an der Registerdatei
DEFAULT_MAX_PADDING = 8.0


def estimate_smem_bytes(cand, buffer_stages):
    # die beiden fp16-Operand-Tiles mal Stages. Akku liegt in Registern.
    a_tile = cand.m_prim * cand.k_prim
    b_tile = cand.k_prim * cand.n_prim
    return (a_tile + b_tile) * 2 * buffer_stages


def estimate_acc_registers(cand):
    return cand.m_prim * cand.n_prim   # M_PRIM x N_PRIM fp32


def padding_ratio(cand):
    orig = cand.orig_m * cand.orig_n * cand.orig_k
    padded = cand.padded_m * cand.padded_n * cand.padded_k
    return padded / orig if orig > 0 else float("inf")


def prune_reason(cand, dev, buffer_stages, reg_fraction, max_padding, smem_limit):
    if cand.m_prim % MMA_ALIGN or cand.n_prim % MMA_ALIGN or cand.k_prim % MMA_ALIGN:
        return "mma_align"
    if estimate_smem_bytes(cand, buffer_stages) > smem_limit:
        return "smem_exceeded"
    if estimate_acc_registers(cand) > dev.regs_per_block * reg_fraction:
        return "acc_registers"
    if padding_ratio(cand) > max_padding:
        return "padding_waste"
    return None


def prune(candidates, dev,
          buffer_stages=DEFAULT_BUFFER_STAGES,
          reg_fraction=DEFAULT_REG_FRACTION,
          max_padding=DEFAULT_MAX_PADDING,
          smem_limit=None):
    # gibt (kept, rejected) mit rejected = [(cand, grund), ...]
    if smem_limit is None:
        smem_limit = dev.usable_smem_per_block()
    kept, rejected = [], []
    for cand in candidates:
        reason = prune_reason(cand, dev, buffer_stages, reg_fraction,
                              max_padding, smem_limit)
        (kept if reason is None else rejected).append(
            cand if reason is None else (cand, reason))
    return kept, rejected


# --- Ranking (Kostenmodell) ---
# GB10 ist bandbreitenlimitiert, FLOPs sind fuer alle gleich -> DRAM-Traffic
# entscheidet. Groessere Gruppe = weniger Nachladen = weniger Traffic.

def estimate_dram_bytes(cand, dtype_bytes=2):
    # A einmal pro Gruppen-Spalte, B einmal pro Gruppen-Zeile, C einmal schreiben
    group_cols = ceildiv(cand.padded_n, cand.n_l2 * cand.n_prim)
    group_rows = ceildiv(cand.padded_m, cand.m_l2 * cand.m_prim)
    a_bytes = cand.padded_m * cand.padded_k * dtype_bytes * group_cols
    b_bytes = cand.padded_k * cand.padded_n * dtype_bytes * group_rows
    c_bytes = cand.padded_m * cand.padded_n * dtype_bytes
    return a_bytes + b_bytes + c_bytes


def estimate_grid(cand):
    blocks = (cand.padded_m // cand.m_prim) * (cand.padded_n // cand.n_prim)
    if cand.variant == "A":
        return blocks
    return blocks // (cand.m_l2 * cand.n_l2)   # B macht l2 als Loop


def occupancy_factor(grid, dev):
    return min(1.0, grid / dev.number_sm)


def rank(candidates, dev, batch=1, model="bw"):
    # model "bw" = reine Bandbreiten-Zeit, "bw_occ" = durch Occupancy geteilt
    # (bestraft zu kleine Grids). Beide Werte stehen in metrics.
    bw = dev.peak_dram_bandwidth()
    ranked = []
    for cand in candidates:
        dram = estimate_dram_bytes(cand) * batch
        grid = estimate_grid(cand) * batch
        est_ms = dram / bw * 1e3
        occ = occupancy_factor(grid, dev)
        ranked.append((cand, {"dram_bytes": dram, "grid": grid, "occupancy": occ,
                              "est_ms": est_ms, "est_ms_occ": est_ms / occ}))
    key = "est_ms_occ" if model == "bw_occ" else "est_ms"
    ranked.sort(key=lambda x: (x[1][key], -x[1]["grid"]))
    return ranked


def dedup_mn_symmetry(candidates):
    # nur quadratisch und verlustbehaftet (M/N sind im Speicher nicht symmetrisch).
    # gespiegelte (m_prim,m_l2)/(n_prim,n_l2)-Paare auf einen Vertreter.
    seen = set()
    kept = []
    for c in candidates:
        if c.orig_m == c.orig_n:
            sides = tuple(sorted([(c.m_prim, c.m_l2), (c.n_prim, c.n_l2)]))
            key = (c.variant, c.k_prim, sides)
            if key in seen:
                continue
            seen.add(key)
        kept.append(c)
    return kept


if __name__ == "__main__":
    from collections import Counter
    from .device_props import GB10

    EINSUM = "cmk, ckn -> cmn"
    SHAPES = [(4, 4096, 4096), (4, 4096, 4096)]

    space = SearchSpace()
    cands, skipped = enumerate_candidates(EINSUM, SHAPES, space)
    print(f"enumeriert {len(cands)} (von {space.size()}), {skipped} verworfen")

    hand = [c for c in cands if c.variant == "A" and c.m_prim == 128
            and c.n_prim == 128 and c.k_prim == 64 and c.m_l2 == 8 and c.n_l2 == 8]
    print("A05-Hand-Config im Set:", "ja" if hand else "NEIN")

    kept, rejected = prune(cands, GB10)
    print(f"prune: {len(cands)} -> {len(kept)}, Gruende {dict(Counter(r for _, r in rejected))}")
    print("Hand ueberlebt prune:",
          any(c.variant == "A" and c.m_prim == 128 and c.n_prim == 128
              and c.k_prim == 64 and c.m_l2 == 8 and c.n_l2 == 8 for c in kept))

    print("Ranking Top-5:")
    for i, (c, m) in enumerate(rank(kept, GB10, batch=4)[:5]):
        print(f"  #{i+1} est={m['est_ms']:.2f}ms grid={m['grid']} | {c.label()}")
