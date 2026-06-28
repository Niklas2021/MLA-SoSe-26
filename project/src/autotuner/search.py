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
    # gepaddete Groessen (so wie sie in der Config stehen)
    padded_m: int
    padded_n: int
    padded_k: int

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
        padded_m=padded_m, padded_n=padded_n, padded_k=padded_k,
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
# M1.3  -  Static Pruning
# ---------------------------------------------------------------------------
# Alles rauswerfen, was schon ohne Kompilieren chancenlos ist. Das ist eine
# HEURISTIK, kein Beweis - der echte Schutz ist das try/except ums Kompilieren
# in M2/M3. Wichtig: das pro-Block-SMEM haengt nur an den Prim-Groessen, NICHT
# an m_l2/n_l2 oder der Variante. Pruning entfernt also ganze Prim-Kombis.

MMA_ALIGN = 16          # fp16-Tensor-Cores wollen M/N/K-Prim als Vielfache von 16
DEFAULT_BUFFER_STAGES = 2   # Annahme: Double Buffering. cuTile koennte mehr fahren -> Parameter
DEFAULT_REG_FRACTION = 0.5  # max Anteil der Register, den der Akku belegen darf
DEFAULT_MAX_PADDING = 8.0   # prune wenn gepaddetes Volumen > Faktor * Original


def estimate_smem_bytes(cand, buffer_stages):
    """Geschaetztes SMEM pro Block: die beiden fp16-Operand-Tiles, mal Stages.
    Der Akku liegt (wie bei Triton) in Registern, nicht im SMEM."""
    a_tile = cand.m_prim * cand.k_prim
    b_tile = cand.k_prim * cand.n_prim
    return (a_tile + b_tile) * 2 * buffer_stages   # 2 Byte pro fp16


def estimate_acc_registers(cand):
    """Akku ist M_PRIM x N_PRIM in fp32 -> so viele 32-bit-Register pro Block."""
    return cand.m_prim * cand.n_prim


def padding_ratio(cand):
    orig = cand.orig_m * cand.orig_n * cand.orig_k
    padded = cand.padded_m * cand.padded_n * cand.padded_k
    return padded / orig if orig > 0 else float("inf")


def prune_reason(cand, dev, buffer_stages, reg_fraction, max_padding, smem_limit):
    """Liefert den Grund, warum der Kandidat rausfliegt - oder None wenn er bleibt.
    Reihenfolge = vom Fundamentalsten zum Weichsten."""

    # 1) MMA-Teilbarkeit (Guard - triggert beim aktuellen Knopf-Set eigentlich nie)
    if (cand.m_prim % MMA_ALIGN or cand.n_prim % MMA_ALIGN or cand.k_prim % MMA_ALIGN):
        return "mma_align"

    # 2) SMEM-Budget (der eigentliche harte Filter)
    if estimate_smem_bytes(cand, buffer_stages) > smem_limit:
        return "smem_exceeded"

    # 3) Register-Druck durch den Akku (optional, weicher)
    if estimate_acc_registers(cand) > dev.regs_per_block * reg_fraction:
        return "acc_registers"

    # 4) Padding-Verschwendung (optional, nur bei krummen/kleinen Shapes relevant)
    if padding_ratio(cand) > max_padding:
        return "padding_waste"

    return None


def prune(candidates, dev,
          buffer_stages=DEFAULT_BUFFER_STAGES,
          reg_fraction=DEFAULT_REG_FRACTION,
          max_padding=DEFAULT_MAX_PADDING,
          smem_limit=None):
    """Filtert die Kandidatenliste. Gibt (kept, rejected) zurueck, wobei
    rejected eine Liste von (candidate, grund) ist - damit nachvollziehbar
    bleibt was warum wegfaellt."""
    if smem_limit is None:
        smem_limit = dev.usable_smem_per_block()

    kept, rejected = [], []
    for cand in candidates:
        reason = prune_reason(cand, dev, buffer_stages, reg_fraction,
                              max_padding, smem_limit)
        if reason is None:
            kept.append(cand)
        else:
            rejected.append((cand, reason))
    return kept, rejected


# ---------------------------------------------------------------------------
# M1.4  -  Ranking (analytisches Kostenmodell)
# ---------------------------------------------------------------------------
# Das ist NICHT zum Zeitsparen da, sondern das eigentliche Forschungsstueck:
# wir messen spaeter alle Kandidaten (Ground Truth) und pruefen, ob dieses
# Modell die wirklich beste Config in seine Top-k / Top-1 zieht.
#
# GB10 ist bandbreitenlimitiert (~270 GB/s, 25 MB L2). Die FLOPs sind fuer alle
# Kandidaten praktisch gleich -> entscheidend ist der DRAM-Traffic. Den senkt
# eine groessere Swizzle-Gruppe, weil A/B seltener nachgeladen werden.

def estimate_dram_bytes(cand, dtype_bytes=2):
    """Geschaetzte Bytes, die von/zu DRAM bewegt werden (pro Batch-Element).
    Reuse-Modell: A wird einmal pro Gruppen-Spalte gelesen, B einmal pro
    Gruppen-Zeile, C einmal geschrieben. Auf gepaddeten Groessen, weil der
    Kernel die Null-Ueberhaenge tatsaechlich mitlaedt."""
    group_cols = ceildiv(cand.padded_n, cand.n_l2 * cand.n_prim)
    group_rows = ceildiv(cand.padded_m, cand.m_l2 * cand.m_prim)
    a_bytes = cand.padded_m * cand.padded_k * dtype_bytes * group_cols
    b_bytes = cand.padded_k * cand.padded_n * dtype_bytes * group_rows
    c_bytes = cand.padded_m * cand.padded_n * dtype_bytes
    return a_bytes + b_bytes + c_bytes


def estimate_grid(cand):
    """Anzahl CTAs pro Batch-Element. Variante A verteilt m_l2/n_l2 ueber die
    bid (mehr CTAs), Variante B macht sie als Loop (weniger CTAs)."""
    blocks = (cand.padded_m // cand.m_prim) * (cand.padded_n // cand.n_prim)
    if cand.variant == "A":
        return blocks
    return blocks // (cand.m_l2 * cand.n_l2)


def occupancy_factor(grid, dev):
    """Grober Auslastungsfaktor in (0,1]: wenn das Grid weniger CTAs hat als die
    GPU SMs, kann die Speicherbandbreite nicht gesaettigt werden. >= 1 Wave -> 1."""
    return min(1.0, grid / dev.number_sm)


def rank(candidates, dev, batch=1, model="bw"):
    """Sortiert die Kandidaten nach vorhergesagter Laufzeit (beste zuerst).

    Es werden ZWEI Modelle berechnet, damit wir sie gegeneinander (und gegen die
    Messung) evaluieren koennen:
      - "bw"     : reine DRAM-Bandbreiten-Zeit
      - "bw_occ" : dieselbe Zeit, aber durch den Occupancy-Faktor geteilt
                   (bestraft zu kleine Grids, z.B. Variante B mit grosser Gruppe)
    model waehlt nur die Sortier-Reihenfolge; beide Werte stehen in metrics.

    Liefert eine Liste von (cand, metrics)."""
    bw = dev.peak_dram_bandwidth()
    ranked = []
    for cand in candidates:
        dram = estimate_dram_bytes(cand) * batch
        grid = estimate_grid(cand) * batch
        est_ms = dram / bw * 1e3
        occ = occupancy_factor(grid, dev)
        ranked.append((cand, {
            "dram_bytes": dram,
            "grid": grid,
            "occupancy": occ,
            "est_ms": est_ms,
            "est_ms_occ": est_ms / occ,
        }))

    key = "est_ms_occ" if model == "bw_occ" else "est_ms"
    ranked.sort(key=lambda x: (x[1][key], -x[1]["grid"]))
    return ranked


# ---------------------------------------------------------------------------
# Optionale Reduktion: M/N-Symmetrie bei quadratischen Problemen
# ---------------------------------------------------------------------------
# ACHTUNG - das ist NICHT verlustfrei. M und N sind im Speicher unterschiedlich
# angeordnet (A ist [M,K] mit K contiguous, B ist [K,N] mit N contiguous, C ist
# [M,N] mit N contiguous). Eine gespiegelte Config kann also real anders schnell
# sein. Deshalb per Default NICHT in der Pipeline - nur als Werkzeug fuer einen
# schnellen ersten Mess-Durchlauf, wenn man das Risiko bewusst eingeht.

def dedup_mn_symmetry(candidates):
    """Bei quadratischen Problemen (orig_m == orig_n) die gespiegelten
    (m_prim,m_l2)/(n_prim,n_l2)-Paare auf je einen Vertreter zusammenfassen.
    Nicht-quadratische Kandidaten bleiben unangetastet."""
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

    # --- M1.3: Pruning ---
    from collections import Counter
    from .device_props import GB10

    print()
    print(f"Pruning gegen {GB10.gpu_name} "
          f"(nutzbares SMEM/Block = {GB10.usable_smem_per_block()} B)")
    kept, rejected = prune(cands, GB10)
    print(f"  {len(cands)} -> {len(kept)} bleiben, {len(rejected)} verworfen")
    print(f"  Gruende: {dict(Counter(r for _, r in rejected))}")

    # Akzeptanz: die A05-Hand-Config darf NICHT weggepruned werden
    hand_kept = any(c.variant == "A" and c.m_prim == 128 and c.n_prim == 128
                    and c.k_prim == 64 and c.m_l2 == 8 and c.n_l2 == 8 for c in kept)
    print("  A05-Hand-Config ueberlebt Pruning:", "OK" if hand_kept else "FEHLER!")

    # Sanity: die dicksten Tiles (256x256) sollten alle weg sein
    big_left = [c for c in kept if c.m_prim == 256 and c.n_prim == 256]
    print(f"  256x256-Kombis nach Pruning: {len(big_left)} (erwartet 0)")

    # optionale Symmetrie-Reduktion (nur quadratisch, verlustbehaftet)
    dedup = dedup_mn_symmetry(kept)
    print(f"  optional dedup_mn_symmetry: {len(kept)} -> {len(dedup)} "
          f"(A05 ist quadratisch)")

    # --- M1.4: Ranking ---
    print()
    print("Ranking (Modell-Vorhersage, beste zuerst), Top-10:")
    ranked = rank(kept, GB10, batch=4)   # C=4 bei A05
    for i, (c, m) in enumerate(ranked[:10]):
        print(f"  #{i+1:2d}  est={m['est_ms']:6.2f} ms  grid={m['grid']:6d}  | {c.label()}")

    # wo landet die gemessene 66.54-TFLOPS-Config (128/128/64, 8x8, A)?
    for i, (c, m) in enumerate(ranked):
        if (c.variant == "A" and c.m_prim == 128 and c.n_prim == 128
                and c.k_prim == 64 and c.m_l2 == 8 and c.n_l2 == 8):
            print(f"\n  gemessene Referenz-Config steht im Modell auf Rang #{i+1} von {len(ranked)}")
            break
