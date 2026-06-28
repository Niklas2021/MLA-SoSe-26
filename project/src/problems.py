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
]

# Die Config, die man OHNE Tuner nehmen wuerde (die handoptimierte A05-Wahl).
# Dient als Baseline fuer den "Tuner-Gewinn".
DEFAULT_CONFIG = dict(variant="A", m_prim=128, n_prim=128, k_prim=64, m_l2=8, n_l2=8)
