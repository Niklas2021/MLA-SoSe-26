Task 4: L2-Optimized Batched Contraction
==========================================

Aufgabenstellung
----------------

Batched Matmul ``cmk, ckn -> cmn`` mit ``|c| = 4`` und ``|m| = |n| = |k| = 4096``.

**a)** Mit ``generate_config`` aus Task 2 die initiale Config erzeugen und reporten.

**b)** Mit ``Optimizer`` aus Task 3 die Config in eine L2-optimierte Form überführen,
nach dem Schema von Folie 34:

.. code-block:: text

   dim_sizes = [ [...], m_l2, n_l2, m_prim, n_prim, k_prim ]

Die Größen ``m_l2``, ``n_l2``, ``m_prim``, ``n_prim`` müssen begründet werden.

**c)** cuTile-Kernel implementieren der diese Config umsetzt, gegen ``torch.einsum``
verifizieren.

**d)** Mit ``triton.testing.do_bench`` messen und mit einer Baseline vergleichen,
die BIDs in plain row-major über ``(c, m, n)`` verteilt.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/05_assignment/src/task4.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/05_assignment/out/task4.log
   :language: text

Teilaufgabe a) – Initiale Config
---------------------------------

``generate_config("cmk, ckn -> cmn", [(4, 4096, 4096), (4, 4096, 4096)])`` liefert:

- ``dim_types  = [C, M, K, N]``
- ``exec_types = [SEQ, SEQ, SEQ, SEQ]``
- ``dim_sizes  = [4, 4096, 4096, 4096]``
- Strides für A, B, C jeweils row-major (siehe Log).

Teilaufgabe b) – Wahl der Größen und Transformation
----------------------------------------------------

Wahl
^^^^

==========  ======  =========================================================
Parameter   Wert    Begründung
==========  ======  =========================================================
``m_prim``  128     übliche GEMM-Tile-Höhe für die Tensor Cores
``n_prim``  128     übliche GEMM-Tile-Breite
``k_prim``  64      Kontraktionsdimension pro MMA-Iter
``m_l2``    8       8 Output-Tiles pro Super-Tile in M-Richtung
``n_l2``    8       8 Output-Tiles pro Super-Tile in N-Richtung
==========  ======  =========================================================

Die Wahl von ``m_l2 = n_l2 = 8`` zielt direkt auf den L2-Cache des GB10
(gemessen ~25 MB, siehe Assignment 02 Task 1, NVIDIA-Spec: 24 MB).

Innerhalb einer 8×8-Super-Tile-Gruppe von 64 Output-Tiles werden insgesamt
``m_l2`` unique A-Zeilen-Streifen und ``n_l2`` unique B-Spalten-Streifen
gebraucht:

.. code-block:: text

   A-Streifen:   m_l2 * m_prim * k * 2 B  =  8 * 128 * 4096 * 2  =  8 MiB
   B-Streifen:   k * n_l2 * n_prim * 2 B  =  4096 * 8 * 128 * 2  =  8 MiB
   ----------------------------------------------------------------
   Σ Working-Set einer Super-Tile-Gruppe:                          16 MiB

Mit 24 MB L2 passt das mit etwas Headroom rein. Größeres ``m_l2``/``n_l2``
würde sprengen (z.B. 16×16 → 32 MB). Innerhalb der Gruppe wird jede A-Zeile
``n_l2 = 8`` Mal und jede B-Spalte ``m_l2 = 8`` Mal wiederverwendet –
DRAM-Traffic für A und B sinkt jeweils um den Faktor 8.

Optimizer-Schritte
^^^^^^^^^^^^^^^^^^

Ausgehend von ``[c, m, k, n]``:

1. ``split_dim`` auf ``m``: ``4096 → (32, 128)``
2. ``split_dim`` auf das neue ``m_outer``: ``32 → (4, 8)``  ⇒ ``m_l2_outer, m_l2``
3. ``split_dim`` auf ``n``: ``4096 → (32, 128)``
4. ``split_dim`` auf das neue ``n_outer``: ``32 → (4, 8)``
5. ``split_dim`` auf ``k``: ``4096 → (64, 64)``  ⇒ ``k_outer, k_prim``
6. ``permute_dims`` auf ``[c, m_l2_outer, n_l2_outer, m_l2, n_l2, k_outer, m_prim, n_prim, k_prim]``
7. ``make_executable`` → setzt ``exec_types`` und finalisiert die
   ``PAR | SEQ | PRIM``-Reihenfolge.

Resultat (siehe Log, *L2-optimierte Config*):

.. code-block:: text

   dim_types  = [C, M, N, M, N, K, M, N, K]
   exec_types = [PAR, PAR, PAR, PAR, PAR, SEQ, PRIM, PRIM, PRIM]
   dim_sizes  = [4, 4, 4, 8, 8, 64, 128, 128, 64]

Anmerkung zur strikten Folie-34-Lesart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Folie 34 zeigt das Schema ``[..., m_l2, n_l2, m_prim, n_prim, k_prim]`` – die
beiden ``l2``-Dims direkt vor den PRIM-Dims. In unserer Variante A liegen
``m_l2`` und ``n_l2`` als ``PAR`` zusammen mit den anderen Parallel-Dims,
damit das BID-Swizzling à la Lecture 3 entstehen kann. Zwischen ``n_l2`` und
``m_prim`` steht dadurch ``k_outer`` (SEQ) – sonst würde die ``PAR | SEQ |
PRIM``-Reihenfolge aus ``verify`` verletzt.

Eine **strikte** Variante B gibt es zusätzlich (siehe Log, ``Variante B –
strict``): dort sind ``m_l2`` und ``n_l2`` als ``SEQ`` markiert und liegen
direkt vor den PRIM-Dims:

.. code-block:: text

   dim_types  = [C, M, N, K, M, N, M, N, K]
   exec_types = [PAR, PAR, PAR, SEQ, SEQ, SEQ, PRIM, PRIM, PRIM]
   dim_sizes  = [4, 4, 4, 64, 8, 8, 128, 128, 64]

Diese Variante folgt der Folie strenger, kostet aber im Benchmark Performance
(siehe Teilaufgabe d).

Teilaufgabe c) – Kernel
-----------------------

Drei Kernels:

- ``kernel_l2`` (Variante A) – BIDs werden auf
  ``(c, m_l2_outer, n_l2_outer, m_l2, n_l2)`` zerlegt; jeder Block berechnet
  ein einziges ``(m_prim, n_prim)``-Output-Tile, die Reihenfolge der BIDs
  liefert das Swizzling automatisch.
- ``kernel_l2_strict`` (Variante B) – BIDs werden nur auf
  ``(c, m_l2_outer, n_l2_outer)`` zerlegt; ``m_l2`` und ``n_l2`` sind
  zwei zusätzliche SEQ-Loops im Kernel, der Block schreibt also
  ``m_l2 · n_l2 = 64`` Output-Tiles raus.
- ``kernel_baseline`` – BIDs in plain row-major über ``(c, m_block, n_block)``,
  kein Swizzling, kein L2-Reuse-Pattern.

Beliebige Tensorgrößen
^^^^^^^^^^^^^^^^^^^^^^

Die Kernels nehmen ``num_m_l2_outer``, ``num_n_l2_outer`` und
``num_k_outer`` als ``ct.Constant[int]``-Argumente und sind damit unabhängig
von festen Größen. Die Launcher berechnen alles per ``ceildiv`` aus den
Tensor-Shapes:

.. code-block:: python

   num_m_l2_outer = ceildiv(m_size, M_PRIM * M_L2)
   num_n_l2_outer = ceildiv(n_size, N_PRIM * N_L2)
   num_k_outer    = ceildiv(k_size, K_PRIM)

Die ``ct.load``-Calls verwenden ``padding_mode=ct.PaddingMode.ZERO``, damit
Out-of-Bounds-Reads stillschweigend ``0`` ergeben. Der Output wird auf das
nächste Tile-Vielfache aufgerundet allokiert (``C_pad``) und nach dem Launch
auf den gültigen Bereich gesliced. Damit muss **A und B nicht gepaddet
werden** – nur das Output-Tensor, was Speicher spart.

Im Korrektheits-Check (Log) sieht man dazu zusätzlich einen Test mit
nicht-teilbaren Größen (``M=1234, N=567, K=890, C=3``), den alle drei Kernels
sauber bestehen.

Teilaufgabe d) – Benchmark und Vergleich
------------------------------------------

Setup: vor jeder Messung 3 manuelle Warmups, dann
``triton.testing.do_bench(warmup=200, rep=2000)``.

Ergebnisse aus dem Log:

==========================  ============  ===============
Kernel                      Laufzeit      Performance
==========================  ============  ===============
L2 Variante A               8.32 ms       66.10 TFLOPS
L2 Variante B (strict)      12.48 ms      44.04 TFLOPS
Baseline (kein Swizzling)   14.24 ms      38.60 TFLOPS
==========================  ============  ===============

* **Variante A vs. Baseline → 1.71× Speedup**

  Beide Kernels haben dieselbe Block-Anzahl (``c · m_blocks · n_blocks =
  4096``), der einzige Unterschied ist die Reihenfolge, in der die BIDs auf
  Tiles abgebildet werden. Bei Variante A laufen 64 aufeinander folgende BIDs
  durch eine 8×8-Super-Tile-Gruppe – die A-Zeilen und B-Spalten dieser
  Gruppe bleiben also im L2 stehen, wenn der nächste Block in der Gruppe sie
  braucht. Genau der Effekt aus Folie 33 / Lecture 3.

  Über die GB10-Bandbreite (273 GB/s) lässt sich der Speedup quantitativ
  einordnen. Beim Baseline werden A-Zeilen natürlicherweise reused, B-Spalten
  aber nicht (32 MiB pro ``m_block``-Reihe > 24 MB L2). Bei Variante A hält
  der L2 den kompletten 16 MiB-Working-Set einer Super-Tile-Gruppe:

  .. code-block:: text

     Baseline:    ~4.1 GiB  / 273 GB/s  →  ~15 ms      (gemessen 14.24 ms)
     Variante A:  ~1   GiB  / 273 GB/s  →  ~3.7 ms     (BW-floor)
                  5.5e11 FLOPs / 125 TFLOPS →  ~4.4 ms (Compute-floor)

  Baseline ist klar BW-bound. Bei Variante A liegen BW- und Compute-Floor
  gleichauf – das Swizzling hat die LPDDR5X-Bandbreite so weit entlastet,
  dass jetzt **Compute der Engpass** ist. Gemessen: 8.32 ms (Differenz
  durch Overhead und nicht-perfekte Tile-Auslastung).

* **Variante B vs. Baseline → 1.14× Speedup**

  B hat zwar L2-Reuse über die SEQ-Loops innerhalb eines Blocks, parallelisiert
  aber nur über ``c · m_l2_outer · n_l2_outer = 64`` Blöcke. Der GB10 hat 48
  SMs – 64 Blöcke ergeben damit ~1.3 Wellen Arbeit (eine volle Welle plus eine
  Tail-Welle mit nur 16 belegten SMs, also ~67 % der SMs leer in der zweiten
  Welle). Außerdem hat der Scheduler kaum Spielraum, Latenzen über zusätzliche
  Wellen zu verstecken. Variante A hat dagegen 4096 Blöcke = ~85 Wellen,
  reichlich Material zum Latenz-Hiding. Der L2-Reuse-Vorteil von B wird
  dadurch fast vollständig vom Parallelitätsverlust aufgefressen.

* **Variante A vs. Variante B → 1.50×**

  Bestätigt direkt: ``m_l2`` und ``n_l2`` als ``PAR`` zu fahren (Lecture-3-
  Style) ist deutlich besser, weil dann das BID-Swizzling den L2-Reuse
  liefert *ohne* Parallelität zu opfern.

Einordnung gegenüber der Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Der GB10 (siehe `NVIDIA-Spec
<https://docs.nvidia.com/dgx/dgx-spark/hardware.html>`_) ist deutlich kleiner
als z.B. eine H100 mit 132 SMs:

================  ===================  ============================
Property          GB10 (DGX Spark)     Quelle
================  ===================  ============================
SMs               48                   NVIDIA Spec / Reviews
CUDA Cores        6144                 NVIDIA Spec
L2-Cache          24 MB                NVIDIA Spec (gemessen 25 MB)
Memory-BW         273 GB/s (LPDDR5X)   NVIDIA Spec
Marketing Peak    1 PFLOP FP4 sparse   NVIDIA Marketing
≈ Dense FP16      ~125 TFLOPS          aus Marketing abgeleitet
================  ===================  ============================

Aus dem 1 PFLOP FP4-mit-Sparsity-Wert folgt nach Halbierung für Dense
und nochmaliger Halbierung pro Datenformat-Stufe (FP4 → FP8 → FP16) eine
Dense FP16/FP32-acc-Obergrenze von ~125 TFLOPS. Reale Matmul-Performance
auf dieser Klasse von Hardware liegt typischerweise bei 50–80 % davon,
also ~60–100 TFLOPS für gut getunte Kernel.

Vergleich mit unseren Messungen aus Assignment 03 (gleiche Hardware):

* Square Matmul ``2048³`` swizzled: **49.30 TFLOPS** (Task 4)
* ``8192×8192×4096`` swizzled: **59.02 TFLOPS** (Task 4)

Unsere ``66.10 TFLOPS`` bei ``4096³`` mit Variante A liegt damit über
den Werten aus Assignment 03 und entspricht ungefähr 50 % der
abgeleiteten Peak-Leistung – für einen handgeschriebenen Kernel ohne
Software-Pipelining oder async-Copies ein gutes Ergebnis.

Zusammenfassung
---------------

* Der geforderte Vergleich (Variante A vs. Baseline) ergibt **1.71× Speedup**
  durch BID-Swizzling.
* Die strikte Folie-34-Variante kostet Parallelität und ist deshalb langsamer
  als die PAR-Variante.
* Korrektheit ist sowohl bei ``4096³`` als auch bei nicht-teilbaren Größen
  sichergestellt; die Kernels selbst sind shape-unabhängig.
