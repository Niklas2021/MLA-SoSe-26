Task 1: Tiled Contraction Kernel Variants
==========================================

Aufgabenstellung
----------------

Einsum: ``eabklxy, ecklyz -> eabcxz``

Fünf Teilaufgaben:

- **a)** Dimensionen klassifizieren
- **b)** Basis-Kernel: GEMM über ``x,y,z``, parallelisiert über ``e,a,b,c``
- **c)** Wie b), aber ``b`` wird serialisiert statt parallelisiert
- **d)** GEMM über ``x,y,z,l`` – ``y`` und ``l`` werden gemerged
- **e)** 3D-MMA über ``e,x,y,z`` – ``e`` wird zur GEMM-Dimension

Alle Kernels werden gegen ``torch.einsum()`` verifiziert, auch mit
nicht-zweierpotenten Größen.

Beliebige Größen – Padding im Kernel
------------------------------------

cuTile verlangt zweierpotente Tile-Dimensionen. ``next_pow2`` rundet deshalb
nur die Kernel-Shapes auf. ``ct.load(..., padding_mode=ct.PaddingMode.ZERO)``
füllt Randbereiche mit Nullen; OOB-Stores werden ignoriert. Die Tensoren selbst
werden nicht gepaddet.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/04_assignment/src/task1.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/04_assignment/out/task1/task1_log.txt
   :language: text

Teilaufgabe a) – Dimensions-Klassifikation
-------------------------------------------

Einsum: ``eabklxy, ecklyz -> eabcxz``

- **C (Batch-dimension)**: ``e`` – taucht in beiden Inputs und im Output auf
- **M**: ``a, b, x`` – nur in A und Output, nicht in B
- **N**: ``c, z`` – nur in B und Output, nicht in A
- **K**: ``k, l, y`` – in beiden Inputs, aber nicht im Output

Für das GEMM-Mapping: ``x`` ist M-Dimension, ``z`` ist N-Dimension,
``y`` ist K-Dimension. ``k`` und ``l`` sind zusätzliche Kontraktionsdimensionen,
über die serialisiert wird.

Teilaufgabe b) – Basis-Kernel
------------------------------

.. code-block:: python

   grid = (e*a*b*c, )

Parallelisiert wird über Dimensionen ``e, a, b, c``.
Der Index vom Block wird per Modulo/Division in die vier Indizes zerlegt:

.. code-block:: python

   pid_c = pid % c
   pid_b = (pid // c) % b
   pid_a = (pid // c // b) % a
   pid_e = pid // c // b // a

Jeder Block berechnet ein ``(x, z)``-Output-Tile. Die Kontraktionsdimensionen
``k`` und ``l`` werden in einer Schleife serialisiert:

.. code-block:: python

   acc = ct.zeros((x, z), dtype=ct.float32)   # x,z = auf pow2 aufgerundet
   for k_i in range(k):
       for l_i in range(l):
           a_tile = ct.load(A, ..., shape=(1,1,1,1,1,x,y), padding_mode=ct.PaddingMode.ZERO)
           b_tile = ct.load(B, ..., shape=(1,1,1,1,y,z), padding_mode=ct.PaddingMode.ZERO)
           a_tile = ct.reshape(a_tile, (x, y))
           b_tile = ct.reshape(b_tile, (y, z))
           acc += ct.matmul(a_tile, b_tile)

Die 7D-Tiles werden auf 2D runtergereshaped, damit ``ct.matmul`` damit
arbeiten kann. Akkumulation in FP32, Store als FP16.

Teilaufgabe c) – b serialisiert
---------------------------------

Gleicher Kernel, aber ``b`` wird nicht parallelisiert sondern serialisiert:

.. code-block:: python

   grid = (e*a*c, )   # statt (e*a*b*c, )

Jeder Block loopt jetzt über ``b``:

.. code-block:: python

   for b_i in range(b):
       acc = ct.zeros((x, z), dtype=ct.float32)
       for k_i in range(k):
           for l_i in range(l):
               ...
       ct.store(C, index=(pid_e, pid_a, b_i, pid_c, 0, 0), tile=out)

Der Akkumulator wird pro ``b``-Iteration auf null gesetzt und das
Ergebnis direkt geschrieben.

**Wann ist b) besser, wann c)?**

.. code-block:: text

   large b (b=64):
     b) 2.891 ms  c) 5.544 ms  -> b) faster

   large e,a,b (e=32, a=32, b=2, c=32):
     b) 0.725 ms  c) 0.521 ms  -> c) faster

Bei großem ``b`` lohnt sich Parallelisierung – 64 Iterationen pro Block
serialisieren dauert zu lang. Bei kleinem ``b`` (zB ``b=2``) kostet die
Serialisierung fast nichts, dafür spart c) Blöcke: ``e*a*c = 32768`` statt
``e*a*b*c = 65536``.

Teilaufgabe d) – GEMM über xyzl (merged)
------------------------------------------

Statt ``y`` allein als GEMM-K-Dimension zu benutzen, werden ``l`` und ``y``
zu einer Dimension ``l*y`` zusammengefasst. Dafür wird das A-Tile erst
permutiert (``l`` und ``x`` tauschen die Position) und dann auf
``(x, l*y)`` reshaped:

.. code-block:: python

   for k_i in range(k):
       a_tile = ct.load(A, ..., shape=(1,1,1,1,l,x,y), padding_mode=ct.PaddingMode.ZERO)
       b_tile = ct.load(B, ..., shape=(1,1,1,l,y,z), padding_mode=ct.PaddingMode.ZERO)

       a_tile = ct.permute(a_tile, (0,1,2,3,5,4,6))  # -> ...,x,l,y
       a_tile = ct.reshape(a_tile, (x, l*y))
       b_tile = ct.reshape(b_tile, (l*y, z))

       acc = ct.mma(a_tile, b_tile, acc)

Damit entfällt die innere ``l``-Schleife – ``l`` steckt jetzt in
der GEMM-K-Dimension.

**Wann ist b) besser, wann d)?**

.. code-block:: text

   small GEMM dims (l=1, x=y=z=16):
     b) 0.130 ms  d) 0.125 ms  -> praktisch gleich

   large l (l=32, x=y=32, z=16):
     b) 2.643 ms  d) 1.283 ms  -> d) faster

Bei ``l=1`` rechnen beide Varianten praktisch gleich; der kleine Unterschied
ist Messrauschen. Bei ``l=32`` wächst die GEMM-K-Dimension auf ``l*y = 1024``,
wodurch d) deutlich schneller wird.

Teilaufgabe e) – 3D-MMA über exyz
-----------------------------------

``e`` wird nicht mehr parallelisiert, sondern als dritte GEMM-Dimension
behandelt:

.. code-block:: python

   grid = (a*b*c, )   
   acc = ct.zeros((e, x, z), dtype=ct.float32)

   for k_i in range(k):
       for l_i in range(l):
           a_tile = ct.load(A, ..., shape=(e,1,1,1,1,x,y), padding_mode=ct.PaddingMode.ZERO)
           b_tile = ct.load(B, ..., shape=(e,1,1,1,y,z), padding_mode=ct.PaddingMode.ZERO)
           a_tile = ct.reshape(a_tile, (e, x, y))
           b_tile = ct.reshape(b_tile, (e, y, z))
           acc = ct.mma(a_tile, b_tile, acc)

``ct.mma`` bekommt 3D-Tiles und führt ein batched Matmul über die
``e``-Dimension aus. Weniger Blöcke im Grid (``e`` weniger),
aber mehr Arbeit pro Block.
