Task 4: cuTile Kernel
======================

Aufgabenstellung
----------------

**a)** Einen cuTile-Kernel implementieren, der die Kontraktion gemäß der
optimierten Config aus Task 3 berechnet.

**b)** Korrektheit durch Vergleich mit dem ``torch.einsum``-Ergebnis aus
Task 1 mit ``torch.allclose`` verifizieren.

**c)** Mit ``triton.testing.do_bench`` die durchschnittliche Laufzeit messen
und die erreichte Performance in TFLOPS berechnen und reporten.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/06_assignment/src/task4.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/06_assignment/out/task4.log
   :language: text

Teilaufgabe a) – Kernel-Implementierung
-----------------------------------------

Der Kernel ``kernel_lf`` folgt direkt der Config aus Task 3:

- Die Block-ID wird in die sieben PAR-Dimensionen
  (a, c, x_l2_out, b, y_l2_out, x_l2, y_l2) dekodiert.
- Der ``s``-Index (K-Dim, Größe 64) wird als sequenzieller Loop ausgeführt.
- Pro Loop-Iteration wird ein ``(prim_k × prim_m)``-Tile aus A und ein
  ``(prim_k × prim_n)``-Tile aus B geladen und per ``ct.mma`` akkumuliert.
  Die ``p``-Dim (Größe 64 = ``prim_k``) ist dabei vollständig im MMA enthalten.
- Da der Output ``tensor_abcyx`` y vor x im Speicher hat (y-Stride > x-Stride),
  wird der Akkumulator vor dem Schreiben transponiert: ``(M, N) → (N, M)``.

Teilaufgabe b) – Korrektheitsnachweis
---------------------------------------

Vergleich mit ``torch.einsum`` (FP16) mit ``rtol=1e-2, atol=1e-1``:

.. code-block:: text

   allclose=True  max_err=0.0010

Der maximale absolute Fehler von **0.0010** liegt deutlich unterhalb der
FP16-Toleranz.

Teilaufgabe c) – Performance
------------------------------

FLOPs-Berechnung (2 Ops pro Multiply-Add, alle Dims):

.. code-block:: text

   FLOPs = 2 × |a| × |b| × |c| × |x| × |y| × |s| × |p|
         = 2 × 4 × 4 × 3 × 1536 × 1152 × 64 × 64
         ≈ 6.96 × 10¹¹ FLOPs

Benchmark mit ``triton.testing.do_bench`` (warmup=50, rep=200):

==========================  ============  ===============
                            Laufzeit      Performance
==========================  ============  ===============
cuTile-Kernel               13.959 ms     49.84 TFLOPS
``torch.einsum``            43.013 ms     16.18 TFLOPS
==========================  ============  ===============

Zusammenfassung
---------------

Der cuTile-Kernel erreicht **49.84 TFLOPS** und ist damit ~**3.1× schneller**
als ``torch.einsum`` – das optionale Ziel ist erfüllt. Der Geschwindigkeitsvorteil
entsteht durch die L2-optimierte Tile-Reihenfolge (BID-Swizzling über x_l2 und
y_l2) sowie durch den direkten FP16-Tensor-Core-Einsatz über ``ct.mma``.
