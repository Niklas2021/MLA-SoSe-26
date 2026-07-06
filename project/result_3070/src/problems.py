# Shapes die der Tuner durchmisst. Frei editierbar.
# name -> tune_<name>.csv, einsum (zwei Inputs, ein M/N/K, optional Batch), shapes, regime-Label.
# Verschiedene Regime: quadratisch / rechteckig / klein-K / gross-K / unteilbar / Batch.

PROBLEMS = [
    # die Referenz aus A05 (Heimvorteil: dafuer wurde von Hand getunt)
    dict(name="a05", einsum="cmk, ckn -> cmn",
         shapes=[(4, 4096, 4096), (4, 4096, 4096)], regime="square, batch=4 (Referenz)"),

    # quadratisch ohne Batch (C=1; der Kernel erwartet das 3D-Layout)
    dict(name="square_1b", einsum="cmk, ckn -> cmn",
         shapes=[(1, 4096, 4096), (1, 4096, 4096)], regime="square, batch=1"),

    # rechteckig: viele Zeilen, wenige Spalten und umgekehrt
    dict(name="tall", einsum="cmk, ckn -> cmn",
         shapes=[(1, 8192, 4096), (1, 4096, 1024)], regime="rechteckig (M>>N)"),
    dict(name="wide", einsum="cmk, ckn -> cmn",
         shapes=[(1, 1024, 4096), (1, 4096, 8192)], regime="rechteckig (N>>M)"),

    # K-Extreme: klein-K ist eher bandbreitenlimitiert (da koennte das Modell
    # ziehen), gross-K ist eher compute-limitiert
    dict(name="small_k", einsum="cmk, ckn -> cmn",
         shapes=[(1, 4096, 512), (1, 512, 4096)], regime="klein-K (bandbreite)"),
    dict(name="large_k", einsum="cmk, ckn -> cmn",
         shapes=[(1, 1024, 8192), (1, 8192, 1024)], regime="gross-K (compute)"),

    # unteilbare Groessen -> testet den Padding-Pfad auf der echten GPU
    dict(name="krumm", einsum="cmk, ckn -> cmn",
         shapes=[(2, 1500, 1000), (2, 1000, 3000)], regime="unteilbar (padding)"),

    # viele kleine Batches
    dict(name="batch16", einsum="cmk, ckn -> cmn",
         shapes=[(16, 1024, 1024), (16, 1024, 1024)], regime="batch=16"),

    # M4: Tensor-Ring aus A06 (acspx,bspy->abcyx). Zweite Struktur-Familie:
    # unabhaengige Batches (a,c nur in A, b nur in B), zwei Reduktionen (s,p).
    # x/y sind die getilten M/N-Dims, p ist prim_k, s der SEQ-Loop. Laeuft ueber
    # den Ring-Kernel. Dieselben Regime wie oben, nur in der Ring-Familie.

    # die Referenz aus A06 (Original-Shapes aus dem Assignment)
    dict(name="a06", einsum="acspx, bspy -> abcyx",
         shapes=[(4, 3, 64, 64, 1536), (4, 64, 64, 1152)],
         regime="tensor-ring (Referenz)"),

    # x == y (quadratische M/N-Tiles)
    dict(name="a06_square", einsum="acspx, bspy -> abcyx",
         shapes=[(2, 2, 64, 64, 2048), (2, 64, 64, 2048)], regime="ring, x==y"),

    # rechteckige M/N: x>>y und y>>x
    dict(name="a06_tall", einsum="acspx, bspy -> abcyx",
         shapes=[(2, 2, 64, 64, 4096), (2, 64, 64, 512)], regime="ring, x>>y"),
    dict(name="a06_wide", einsum="acspx, bspy -> abcyx",
         shapes=[(2, 2, 64, 64, 512), (2, 64, 64, 4096)], regime="ring, y>>x"),

    # Reduktion (s*p) klein vs. gross
    dict(name="a06_small_k", einsum="acspx, bspy -> abcyx",
         shapes=[(4, 2, 8, 32, 2048), (4, 8, 32, 2048)], regime="ring, klein-K (s*p=256)"),
    dict(name="a06_large_k", einsum="acspx, bspy -> abcyx",
         shapes=[(2, 2, 128, 128, 1024), (2, 128, 128, 1024)], regime="ring, gross-K (s*p=16384)"),

    # unteilbar in x, y UND p -> Padding-Pfad auf allen drei Achsen
    dict(name="a06_krumm", einsum="acspx, bspy -> abcyx",
         shapes=[(3, 2, 48, 48, 1500), (3, 48, 48, 1000)], regime="ring, unteilbar (padding)"),

    # viele unabhaengige Batches (a*c auf A-Seite, b auf B-Seite) -> grosses Grid
    dict(name="a06_batch", einsum="acspx, bspy -> abcyx",
         shapes=[(8, 4, 32, 64, 1024), (8, 32, 64, 1024)], regime="ring, viele Batches (a8 c4 b8)"),
]

# Die Config, die man OHNE Tuner nehmen wuerde (die handoptimierte A05-Wahl).
# Dient als Baseline fuer den "Tuner-Gewinn".
DEFAULT_CONFIG = dict(variant="A", m_prim=128, n_prim=128, k_prim=64, m_l2=8, n_l2=8)
