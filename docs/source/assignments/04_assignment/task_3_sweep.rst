Task 3: GEMM Dimension Size Sweep
==================================

Aufgabenstellung
----------------

**a)** Contraction-Kernel für ``ackm, bcnk -> abnm``. Fix:
``|a| = 16``, ``|b| = 16``, ``|c| = 32``. ``m, n, k`` beliebig.

**b)** Sweeps mit ``triton.testing.do_bench``:

1. ``|k| = |m| = 64``, ``n`` von **17 bis 129**
2. ``|m| = |n| = 64``, ``k`` von **17 bis 129**

Klassifikation der Indizes
--------------------------

Nach Folie 8:

==========  =====================  =====================  ============
Typ         A (``ackm``)           B (``bcnk``)           C (``abnm``)
==========  =====================  =====================  ============
**M**       ``a, m``               –                      ``a, m``
**N**       –                      ``b, n``               ``b, n``
**K**       ``c, k``               ``c, k``               –
==========  =====================  =====================  ============

Keine Batch-Dimension. Pro CTA wird ein ``(n_tile, m_tile)``-Output-Tile
berechnet, die K-Reduktion läuft über ``c`` und ``k``.

Lösung
------

Kernel
^^^^^^

Pro ``(a, b, n_block, m_block)`` ein CTA. ``c`` und ``k`` werden im
Kernel sequenziell geloopt:

.. code-block:: python

   acc = ct.zeros((n_tile, m_tile), dtype=ct.float32)

   for c_i in range(C_SIZE):
       for k_i in range(num_k_blocks):
           a_tile = ct.load(A, index=(pid_a, c_i, k_i, pid_m),
                            shape=(1, 1, k_tile, m_tile),
                            padding_mode=ct.PaddingMode.ZERO)
           b_tile = ct.load(B, index=(pid_b, c_i, pid_n, k_i),
                            shape=(1, 1, n_tile, k_tile),
                            padding_mode=ct.PaddingMode.ZERO)
           a_tile = ct.reshape(a_tile, (k_tile, m_tile))
           b_tile = ct.reshape(b_tile, (n_tile, k_tile))
           acc = ct.mma(b_tile, a_tile, acc)

Reihenfolge ``mma(b_tile, a_tile, acc)``: das Output-Tile hat Form
``(n, m)``, also passt das ``(n, k)``-Tile als linker, das ``(k, m)``-Tile
als rechter MMA-Operand.

Beliebige Größen – Padding im Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Wir nutzen ``m_tile = n_tile = k_tile = 32``. Die Tile-Anzahl wird mit
``ceildiv`` berechnet, die Tensoren selbst bleiben ungepaddet.
``padding_mode=ct.PaddingMode.ZERO`` nullt OOB-Loads; OOB-Stores werden
ignoriert. Damit funktionieren auch beliebige ``m, n, k`` ohne zusätzlichen
Speicher für gepaddete Tensoren.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/04_assignment/src/task3.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/04_assignment/out/task3/task3_log.txt
   :language: text

Teilaufgabe a) – Korrektheit
-----------------------------

Sechs Test-Cases (drei davon mit non-pow2 Größen 17, 33, 51, 97, 129)
laufen mit ``allclose`` ``True`` und ``max_err ≤ 0.125``. Der Referenzpfad
akkumuliert ebenfalls in FP32 und rundet das Ergebnis anschließend auf FP16;
die beobachtete maximale Abweichung liegt innerhalb der gewählten
``atol=0.1, rtol=0.01``-Toleranz.

Teilaufgabe b) – Sweep ``n`` (m=64, k=64)
------------------------------------------

.. image:: ../../_static/task3b_sweep_n_a04.png
   :alt: TFLOPS sweep n von 17 bis 129
   :width: 95%

Sägezahn-Muster mit Peaks bei ``n = 32, 64, 96, 128``:

==========  =========  =============  =========  =============
n           TFLOPS     n+1            TFLOPS     Drop
==========  =========  =============  =========  =============
32          15.97      33             10.20      −36 %
64          18.05      65             13.41      −26 %
96          18.66      97             15.49      −17 %
128         19.28      129            16.05      −17 %
==========  =========  =============  =========  =============

Sweep ``k`` (m=64, n=64)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/task3b_sweep_k_a04.png
   :alt: TFLOPS sweep k von 17 bis 129
   :width: 95%

Anders als beim ``n``-Sweep zeigt die Kurve Spitzen bei Vielfachen von **8**,
mit den höchsten Werten bei Vielfachen von 32:

* ``k = 32``: 14.45 TFLOPS, ``k = 33``: 3.26 TFLOPS → **−77 %**
* Peaks bei ``k = 64 / 96 / 128``: 17.79 / 19.31 / 19.68 TFLOPS
* dazwischen, wenn ``k`` **kein** Vielfaches von 8 ist: nur 3–6 TFLOPS

``k`` ist im ``B``-Tile die zusammenhängende Ladedimension. Die 8er-Periodik
deutet daher auf Alignment/Vektorisierung der maskierten FP16-Loads hin. Beim
``n``-Sweep bleibt diese Achse mit ``k=64`` aligned.

Erklärung – Tile Quantization
------------------------------

Das Phänomen heißt **Tile Quantization** und ist im NVIDIA-Performance-Guide
beschrieben (`Tile Quantization
<https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#tile-quant>`_).
Sobald eine Dimension nicht durch die Tile-Größe teilbar ist, muss eine
ganze zusätzliche Tile-Reihe gerechnet werden, die kaum etwas zum Output
beiträgt.

Bei uns mit Tile-Größe 32:

* ``n = 32``: ein Tile, 100 % Nutzung → Peak.
* ``n = 33``: zwei Tile-Reihen berechnet, davon nur ``33/64 ≈ 52 %``
  echtes Output → Throughput fällt deutlich (−36 %).
* zwischen den Klippen wächst Throughput linear, weil Tile-Anzahl
  konstant bleibt und nur das echte Output ansteigt.
* mit größerem ``n`` werden die Drops kleiner, weil das relative
  Padding sinkt (bei ``n = 129`` nur noch ``129/160 = 81 %`` Nutzung).

Ein einfaches Nutzungsmodell ist

.. math::

   \frac{\text{TFLOPS}(n)}{\text{TFLOPS}(n_\text{pad})} \approx
   \frac{n}{n_\text{pad}}

Das Modell passt bei größeren ``n`` gut; bei kleinen Größen spielen weitere
Launch- und Scheduling-Effekte mit hinein.

Warum überhaupt?
^^^^^^^^^^^^^^^^

Tensor-Core-MMA arbeitet mit festen Fragmentgrößen; teilweise belegte
Rand-Tiles vermeiden daher nicht automatisch die Arbeit des vollständigen
Tiles. Workarounds in der Praxis:

* **mehrere Tile-Größen + Heuristik** wie bei cuBLAS/cuDNN
* **maskierte Loads/Stores** (CUTLASS, Triton) statt Padding
* **persistente Kernels** mit Tile-Schleife im Kernel

Hier verwenden wir maskierte Loads mit ``padding_mode=ZERO``. Dadurch laufen
beliebige Größen ohne extra Tensor-Padding; im ``k``-Sweep reagieren die
Rand-Loads allerdings deutlich auf ``k % 8``.
