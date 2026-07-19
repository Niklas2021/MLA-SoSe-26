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

# Die handoptimierte A05-Wahl. ACHTUNG: das ist NICHT die faire Baseline, auch wenn wir
# sie lange so benutzt haben. Sie wurde fuer eine einzige Shape (4096^3) hergeleitet,
# und ihre Gruppen-Ausdehnung ist mit 8*128 = 1024 so gross, dass M und N auf Vielfache
# von 1024 hochgepaddet werden. Auf Shapes, die das nicht sind, kostet das brutal:
# bei a06 (x=1536, y=1152) ist das gepaddete Volumen 2.37x das echte. Sie steht hier
# noch als Referenz "naiv von einer anderen Shape/GPU uebernommen".
DEFAULT_CONFIG = dict(variant="A", m_prim=128, n_prim=128, k_prim=64, m_l2=8, n_l2=8, order=0)

# Die faire Baseline: EINE feste Config pro GPU, so wie sie jemand waehlt, der einmal
# auf der Zielkarte nachmisst und sich dann festlegt. Hergeleitet mit baselines_study.py
# (beste feste Config ueber alle Shapes; leave-one-out bestaetigt die Wahl).
#
# Warum sie besser ist, laesst sich ohne Messung begruenden: der Akkumulator ist mit
# 64*256 = 16384 fp32 genau so gross wie bei 128x128, kostet also dieselben Register --
# aber die Gruppen-Ausdehnung halbiert sich auf 512x512. Man tauscht ~25 % Arithmetic
# Intensity (Operanden-Traffic (64+256) statt (128+128) pro K) gegen deutlich weniger
# Padding-Quantisierung. Auf glatt teilbaren Shapes verliert man dadurch 0-5 %, auf
# krummen gewinnt man 25-92 %.
BASELINE_CONFIGS = {
    # GB10: aus der Vollmessung (results_dgx_v2), 89.5 % im leave-one-out
    "NVIDIA GB10": dict(variant="A", m_prim=64, n_prim=256, k_prim=64,
                        m_l2=8, n_l2=2, order=0),
    # 3070: aus baseline_probe (results_3070_v2). Erreicht nur 75.9 % des
    # Per-Shape-Optimums -- auf dieser Karte gibt es keine gute feste Config, die
    # Optima liegen viel weiter auseinander als auf der GB10. Genau deshalb lohnt
    # Per-Shape-Tuning hier mehr (1.32x statt 1.12x).
    # Der frueher hier stehende Wert (64/256/32, 4x2) kam aus dem alten 3070-Sweep
    # und war unbrauchbar: dort waren alle batch=1-Shapes 3-5x zu langsam gemessen.
    "NVIDIA GeForce RTX 3070": dict(variant="A", m_prim=64, n_prim=128, k_prim=64,
                                    m_l2=8, n_l2=2, order=0),
}


def baseline_for(gpu_name):
    # unbekannte GPU -> A05-Default, aber das ist dann eben nur eine Notloesung
    return BASELINE_CONFIGS.get(gpu_name, DEFAULT_CONFIG)


# M5.3: welche Einsum-Strings die Kernel abdecken. Bewusst NICHT in PROBLEMS, damit
# die Studie oben vergleichbar bleibt -- das hier prueft Abdeckung, nicht Performance.
# supported=False heisst: muss mit klarer Meldung abgelehnt werden (frueher hat der
# Kernel da still das falsche Layout gelesen und es fiel erst beim allclose auf).
COVERAGE = [
    dict(name="gemm_batch", einsum="cmk, ckn -> cmn",
         shapes=[(4, 1024, 1024), (4, 1024, 1024)], supported=True,
         note="A05-Form, ein Batch (Regression)"),
    dict(name="gemm_nobatch", einsum="mk, kn -> mn",
         shapes=[(2048, 2048), (2048, 2048)], supported=True,
         note="kein Batch -> Dummy-Achse"),
    dict(name="gemm_2batch", einsum="bcmk, bckn -> bcmn",
         shapes=[(2, 3, 512, 512), (2, 3, 512, 512)], supported=True,
         note="zwei Batches -> fusioniert"),
    dict(name="attn_out", einsum="bhqk, bhkd -> bhqd",
         shapes=[(2, 8, 1024, 1024), (2, 8, 1024, 64)], supported=True,
         note="Attention-Output, b/h fusioniert"),
    dict(name="ring", einsum="acspx, bspy -> abcyx",
         shapes=[(2, 2, 32, 64, 1024), (2, 32, 64, 1024)], supported=True,
         note="A06-Form (Regression)"),

    # ab hier gedrehte Layouts -> Tile-Transpose im flex-Kernel (nur Variante A)
    dict(name="gemm_nt", einsum="cmk, cnk -> cmn",
         shapes=[(4, 1024, 1024), (4, 1024, 1024)], supported=True,
         note="B als (N,K) -- NT"),
    dict(name="gemm_tn", einsum="ckm, ckn -> cmn",
         shapes=[(4, 1024, 1024), (4, 1024, 1024)], supported=True,
         note="A als (K,M) -- TN"),
    dict(name="gemm_outT", einsum="cmk, ckn -> cnm",
         shapes=[(4, 1024, 1024), (4, 1024, 1024)], supported=True,
         note="Output transponiert"),
    dict(name="attn_scores", einsum="bhqd, bhkd -> bhqk",
         shapes=[(2, 8, 1024, 64), (2, 8, 1024, 64)], supported=True,
         note="Attention-Scores, B als NT"),
    dict(name="gemm_all_t", einsum="ckm, cnk -> cnm",
         shapes=[(4, 1024, 1024), (4, 1024, 1024)], supported=True,
         note="A, B und C gedreht"),

    # das bleibt abgelehnt: Batch nicht aussen -> kein Reshape, kein Tile-Transpose hilft
    dict(name="batch_inner", einsum="mck, ckn -> mcn",
         shapes=[(1024, 4, 1024), (4, 1024, 1024)], supported=False,
         note="Batch-Dim nicht aussen"),
]
