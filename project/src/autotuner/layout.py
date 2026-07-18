# Was die Kernel an Speicher-Layout erwarten -- und wie man eine Einsum-Shape da
# hinbiegt, soweit das gratis geht. Reines Python, kein cuTile.
#
# parse_einsum klassifiziert nur M/N/K/Batch, sagt aber nichts darueber, ob die Dims
# auch in der Reihenfolge liegen, die der Kernel indiziert. Genau da war die Luecke:
# z.B. "cmk,cnk->cmn" (B als NT) parst sauber, der Kernel liest dann das falsche
# Layout und es faellt erst beim allclose als "incorrect" auf -- ununterscheidbar von
# einem echten Rechenfehler. plan_layout schliesst das: entweder gratis hinbiegen
# oder mit klarer Begruendung ablehnen.
#
# Gratis ist nur Umsortieren/Reshapen der Batch-Achsen. Transponieren (NT/TN) ist es
# nicht -- kein Vertauschen von Indizes aendert, was physisch stride-1 liegt.
import math
from dataclasses import dataclass, field


class UnsupportedLayout(NotImplementedError):
    pass


@dataclass
class Layout:
    family: str                 # "gemm" oder "ring"
    batch_sizes: list = field(default_factory=list)   # Original-Batch-Dims, der Reihe nach
    add_dummy: bool = False     # kein Batch im Einsum -> fuer den Kernel eine 1 davor
    # Transponier-Flags: A als (batch,K,M) statt (batch,M,K), B als (batch,N,K) statt
    # (batch,K,N), C als (batch,N,M) statt (batch,M,N). Kein view() moeglich -- das
    # loest der flex-Kernel per Tile-Transpose (ct.permute), so wie matmul_ring_a.
    trans_a: bool = False
    trans_b: bool = False
    trans_c: bool = False

    def needs_transpose(self):
        return self.trans_a or self.trans_b or self.trans_c

    def to_kernel(self, A, B):
        # Batch-Dims zu einer zusammenfassen. view() statt reshape(), weil reshape bei
        # nicht-zusammenhaengenden Tensoren still kopieren wuerde -- das waere eine
        # unsichtbare Kopie mitten im Benchmark.
        if self.family == "ring":
            return A, B
        nb = math.prod(self.batch_sizes) if self.batch_sizes else 1
        return A.view(nb, *A.shape[len(self.batch_sizes):]), \
               B.view(nb, *B.shape[len(self.batch_sizes):])

    def from_kernel(self, C):
        # (batch, M, N) zurueck auf die Original-Batch-Achsen
        if self.family == "ring":
            return C
        if self.add_dummy:
            return C.view(*C.shape[1:])
        return C.view(*self.batch_sizes, *C.shape[1:])


def plan_layout(e):
    # e = Einsum aus parse_einsum. -> Layout, oder UnsupportedLayout mit Begruendung.
    return _plan_ring(e) if e.is_multi() else _plan_gemm(e)


def _plan_gemm(e):
    # Kanonisch indizieren die Kernel A als (batch, M, K), B als (batch, K, N),
    # C als (batch, M, N). Batch-Dims muessen aussen und in allen drei Tensoren gleich
    # sortiert sein -- dann ist das Fusionieren ein reines Reshape. Die jeweils
    # gedrehte Variante geht auch, kostet aber einen Tile-Transpose im Kernel.
    b = "".join(e.batch_chars)
    flags = {}
    for who, got, plain, flipped in (
            ("A", e.in_a, b + e.m_char + e.k_char, b + e.k_char + e.m_char),
            ("B", e.in_b, b + e.k_char + e.n_char, b + e.n_char + e.k_char),
            ("C", e.out, b + e.m_char + e.n_char, b + e.n_char + e.m_char)):
        if got == plain:
            flags[who] = False
        elif got == flipped:
            flags[who] = True
        else:
            raise UnsupportedLayout(
                f"{who}-Layout '{got}' passt zu keinem Kernel (erwartet '{plain}' "
                f"oder '{flipped}'). Die Batch-Dims muessen aussen stehen und in "
                f"allen drei Tensoren gleich sortiert sein.")
    return Layout(family="gemm",
                  batch_sizes=[e.size_of[c] for c in e.batch_chars],
                  add_dummy=not e.batch_chars,
                  trans_a=flags["A"], trans_b=flags["B"], trans_c=flags["C"])


def _plan_ring(e):
    # matmul_ring_a indiziert A als (a, c, s, p, x), B als (b, s, p, y),
    # C als (a, b, c, y, x). Das ist genau die A06-Topologie: zwei A-seitige
    # Batches, einer auf der B-Seite, eine SEQ-Reduktion.
    if len(e.extra_m_chars) != 2 or len(e.extra_n_chars) != 1 or len(e.seq_k_chars) != 1:
        raise UnsupportedLayout(
            f"Ring-Kernel deckt genau die A06-Topologie ab (2 extra-M, 1 extra-N, "
            f"1 SEQ-K). Hier: extra_m={e.extra_m_chars} extra_n={e.extra_n_chars} "
            f"seq_k={e.seq_k_chars}.")
    a, c = e.extra_m_chars
    bb, = e.extra_n_chars
    s, = e.seq_k_chars
    want_a = a + c + s + e.k_char + e.m_char
    want_b = bb + s + e.k_char + e.n_char
    want_out = a + bb + c + e.n_char + e.m_char
    for got, want, who in ((e.in_a, want_a, "A"), (e.in_b, want_b, "B"), (e.out, want_out, "C")):
        if got != want:
            raise UnsupportedLayout(
                f"{who}-Layout '{got}' passt nicht zum Ring-Kernel (erwartet '{want}').")
    return Layout(family="ring")


def is_supported(e):
    try:
        plan_layout(e)
        return True, ""
    except UnsupportedLayout as ex:
        return False, str(ex)


if __name__ == "__main__":
    from .einsum_parser import parse_einsum

    # (einsum, shapes, laeuft?) -- die Abdeckungstabelle
    # (einsum, shapes, laeuft?, erwartete Transponier-Flags)
    CASES = [
        ("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)], True, ""),        # A05
        ("mk,kn->mn", [(4096, 4096), (4096, 4096)], True, ""),                 # kein Batch
        ("bcmk,bckn->bcmn", [(2, 3, 512, 512), (2, 3, 512, 512)], True, ""),   # zwei Batches
        ("bhqk,bhkd->bhqd", [(2, 8, 512, 512), (2, 8, 512, 64)], True, ""),    # Attention-Out
        ("acspx,bspy->abcyx", [(4, 3, 64, 64, 1536), (4, 64, 64, 1152)], True, ""),  # A06
        ("cmk,cnk->cmn", [(4, 4096, 4096), (4, 4096, 4096)], True, "B"),       # NT
        ("ckm,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)], True, "A"),       # TN
        ("cmk,ckn->cnm", [(4, 4096, 4096), (4, 4096, 4096)], True, "C"),       # Out transponiert
        ("bhqd,bhkd->bhqk", [(2, 8, 512, 64), (2, 8, 512, 64)], True, "B"),    # Attention-Scores
        ("ckm,cnk->cnm", [(4, 4096, 4096), (4, 4096, 4096)], True, "ABC"),     # alles gedreht
        ("mck,ckn->mcn", [(4096, 4, 4096), (4, 4096, 4096)], False, ""),       # Batch nicht aussen
    ]
    for es, shapes, expect, want_flags in CASES:
        e = parse_einsum(es, shapes)
        ok, why = is_supported(e)
        flags = ""
        if ok:
            lay = plan_layout(e)
            flags = ("A" if lay.trans_a else "") + ("B" if lay.trans_b else "") + \
                    ("C" if lay.trans_c else "")
        mark = "ok " if ok else "-- "
        detail = (f"transponiert: {flags}" if flags else "kanonisch") if ok else why[:60]
        print(f"{mark}{es:22s} {detail}")
        assert ok == expect, f"{es}: erwartet {expect}, ist {ok}"
        assert flags == want_flags, f"{es}: Flags {flags!r} != {want_flags!r}"

    # Round-Trip mit echten Tensoren. Wichtig ist der gepaddete Fall: der Kernel gibt
    # C_pad[:, :m, :n] zurueck, also einen nicht-zusammenhaengenden Slice -- view()
    # muss darauf noch gehen, sonst faellt das erst auf der GPU auf.
    try:
        import torch
    except ImportError:
        print("\nlayout ok (Round-Trip uebersprungen, kein torch)")
        raise SystemExit
    for es, shapes in (("mk,kn->mn", [(4096, 4096), (4096, 4096)]),
                       ("bcmk,bckn->bcmn", [(2, 3, 512, 512), (2, 3, 512, 512)]),
                       ("cmk,ckn->cmn", [(4, 1000, 1000), (4, 1000, 1000)])):
        lay = plan_layout(parse_einsum(es, shapes))
        A, B = torch.zeros(shapes[0]), torch.zeros(shapes[1])
        Ak, Bk = lay.to_kernel(A, B)
        assert Ak.dim() == 3 and Bk.dim() == 3, f"{es}: Kernel will 3D"
        m, n = Ak.shape[1], Bk.shape[2]
        # so wie run_variant_a: gepaddet rechnen, zurueckschneiden
        C_pad = torch.zeros(Ak.shape[0], m + 28, n + 28)
        out = lay.from_kernel(C_pad[:, :m, :n])
        want = tuple(list(lay.batch_sizes) + [m, n])
        assert tuple(out.shape) == want, f"{es}: {tuple(out.shape)} != {want}"
    print("\nlayout ok")
