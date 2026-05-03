Task 2: Kernel Fusion
======================

Aufgabenstellung
----------------

**a)** cuTile-Kernel für ``eabklxy, ecklyz -> eabcxz``, der zusätzlich
eine elementweise Multiplikation mit Tensor ``D`` (Form ``eabcxz``)
in das Output-Tile fusioniert.

**b)** Separater Kernel der **nur** die elementweise Multiplikation
macht. Vergleich Laufzeit fused vs. (Contraction + elwise sequenziell).
Tensorgrößen so, dass die FLOP-Zahl in der Größenordnung einer
``2048³``-Matmul liegt.

Wahl der Tensorgrößen
---------------------

FLOPs der Contraction: ``2 · (e·a·b·x) · (c·z) · (k·l·y)``. Mit

.. code-block:: text

   e=8  a=8  b=16  c=8
   k=8  l=4
   x=32 y=32 z=32

ergibt das ``≈ 17,18 GFLOP``, also genau ``2 · 2048³``. A=16 MB, B=1 MB,
C=2 MB, D=2 MB – Speicher unkritisch.

Lösung
------

Fused Kernel
^^^^^^^^^^^^

Idee aus Folie 24: das Output-Tile **vor** dem ``ct.store`` direkt mit
dem D-Tile multiplizieren, dann erst rausschreiben. Spart den
Roundtrip ``C → HBM → C``, den zwei separate Kernels brauchen.

.. code-block:: python

   acc = ct.zeros((x, z), dtype=ct.float32)
   for k_i in range(k):
       for l_i in range(l):
           # ... Contraction wie in task1 b)
           acc = ct.mma(a_tile, b_tile, acc)

   d_tile = ct.load(D, ..., shape=(1, 1, 1, 1, x, z))
   d_tile = ct.reshape(d_tile, (x, z)).astype(ct.float32)
   acc = acc * d_tile

   out = ct.reshape(acc, (1, 1, 1, 1, x, z)).astype(ct.float16)
   ct.store(C, ..., tile=out)

``d_tile`` wird auf FP32 gecastet damit der Akku-Datentyp bei der
Multiplikation erhalten bleibt; der finale Cast auf FP16 passiert beim
Store.

Elwise-Mult-Kernel
^^^^^^^^^^^^^^^^^^

Ein CTA pro ``(e, a, b, c)``-Position. Lädt C-Tile und D-Tile,
multipliziert, schreibt in-place zurück:

.. code-block:: python

   c_tile = ct.load(C, ..., shape=(1, 1, 1, 1, x, z))
   d_tile = ct.load(D, ..., shape=(1, 1, 1, 1, x, z))
   ct.store(C, ..., tile=c_tile * d_tile)

Wiederverwendung Contraction-Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cuTile-Kernels sind ganz normale Python-Objekte, also kann der Kernel
``b_contraction`` aus Task 1 für die "ohne Fusion"-Variante einfach
importiert werden:

.. code-block:: python

   from task1 import b_contraction

Verifikation
^^^^^^^^^^^^

Beide Varianten (fused und sequenziell) werden gegen
``torch.einsum('eabklxy,ecklyz->eabcxz', A, B) * D`` mit
``torch.allclose(atol=1e-2, rtol=1e-2)`` geprüft – im Log unter
``a) fused verification`` und ``b) sequential verification``.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/04_assignment/src/task2.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/04_assignment/out/task2/task2_log.txt
   :language: text

Beobachtung
-----------

==================  ============
Variante            Laufzeit
==================  ============
fused               16.85 ms
sequential          18.36 ms
**Speedup**         **1.09×**
==================  ============

Der Speedup von ~9 % ist klein, aber er passt zur Theorie: Die
Contraction selbst ist mit ``17 GFLOP`` Compute-bound und macht den
Großteil der Laufzeit aus. Der elwise-Mult ist im Vergleich winzig –
wenige MB HBM-Traffic, plus der Kernel-Launch-Overhead. Genau das spart
die Fusion ein, also liegt der Gewinn auch ungefähr in dieser
Größenordnung.

Größere Speedups durch Fusion sind dort zu erwarten, wo die
nachgelagerte Operation einen substantielleren Anteil an der Laufzeit
hat (z. B. Softmax nach Attention, oder mehrere elementweise
Operationen hintereinander). ``torch.compile`` und ``triton.jit``
machen genau solche Fusions automatisch
(`PyTorch Inductor Docs
<https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html>`_).
