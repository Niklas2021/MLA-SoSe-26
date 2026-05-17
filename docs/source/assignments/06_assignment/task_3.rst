Task 3: Optimized Config
=========================

Aufgabenstellung
----------------

**a)** Den ``Optimizer`` aus Assignment 05 auf die Config aus Task 2 anwenden
und eine valide, launchbare Config erzeugen. Auf Performance optimieren.

**b)** Die finale optimierte Config mit allen Feldern reporten.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/06_assignment/src/task3.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/06_assignment/out/task3.log
   :language: text

Teilaufgabe a) – Optimierungsstrategie
-----------------------------------------

Wahl der Tile-Größen
^^^^^^^^^^^^^^^^^^^^

============  ======  ==========================================================
Parameter     Wert    Begründung
============  ======  ==========================================================
``prim_m``    128     Standard-GEMM-Tile-Höhe für Tensor Cores
``prim_n``    128     Standard-GEMM-Tile-Breite
``prim_k``    64      p hat Größe 64 – passt direkt als ``prim_k``, kein Split
``m_l2``      2       x (1536) = 6 · 2 · 128 → 6 L2-Outer-Blöcke
``n_l2``      3       y (1152) = 3 · 3 · 128 → 3 L2-Outer-Blöcke
============  ======  ==========================================================

``p`` wird nicht gesplittet, da es bereits die Größe 64 hat und direkt als
``prim_k`` dient. ``s`` (Größe 64) bleibt als einzige ``SEQ``-Dimension (K-Loop
im Kernel).

Optimizer-Schritte
^^^^^^^^^^^^^^^^^^

Ausgehend von ``[a, c, s, p, x, b, y]``:

1. ``split_dim`` auf ``x`` (1536): ``1536 → (12, 128)``
2. ``split_dim`` auf das neue ``x_outer`` (12): ``12 → (6, 2)``  ⇒ ``x_l2_outer, x_l2``
3. ``split_dim`` auf ``y`` (1152): ``1152 → (9, 128)``
4. ``split_dim`` auf das neue ``y_outer`` (9): ``9 → (3, 3)``  ⇒ ``y_l2_outer, y_l2``
5. ``permute_dims`` → Reihenfolge ``[PAR... | SEQ | PRIM]``
6. ``make_executable`` → setzt ``exec_types`` und verifiziert die Config

Teilaufgabe b) – Optimierte Config
-------------------------------------

Nach den Splits entstehen 11 Dimensionen. Die letzten drei bilden den
GEMM-Kern (``prim_m=128``, ``prim_n=128``, ``prim_k=64``), die ersten sieben
laufen parallel, nur ``s`` (K-Dim) wird sequenziell ausgeführt:

.. code-block:: text

   dim_sizes  = [4, 3, 6, 4, 3, 2, 3, 64, 128, 128, 64]
   exec_types = [PAR, PAR, PAR, PAR, PAR, PAR, PAR, SEQ, PRIM, PRIM, PRIM]
