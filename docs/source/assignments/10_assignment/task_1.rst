Task 1: Setup of the Whole NPU
==============================

Aufgabenstellung
----------------

Variablen für jede Kachel (Shim, Memory, Compute) anlegen, die gemeinsame
``ab``-Schleife passend dimensionieren, die Core-Funktion für jede Compute-Kachel
duplizieren und die FIFO-Suffixe so setzen, dass sie zu Spalte und Zeile der
Kachel passen.

Lösung
------

Kachel-Variablen
~~~~~~~~~~~~~~~~~

Wir deklarieren alle Kacheln des Arrays über ``aie.tile(col, row)``:

- **Shim** (Row 0): ``aie.tile(0..7, 0)`` — acht Spalten.
- **Memory** (Row 1): ``aie.tile(0..7, 1)`` — ein Memory-Tile je Spalte (L2-Relay).
- **Compute** (Rows 2–5): ``aie.tile(0..7, 2..5)`` — 8 × 4 = 32 Compute-Kacheln.

Spalte = ``x`` (0–7), Zeilenindex = ``y`` (0–3), wobei ``aie.tile(_, 2)``
dem Zeilenindex 0 entspricht, ``aie.tile(_, 3)`` dem Index 1 usw.

Gemeinsame ``ab``-Schleife
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Da ``x`` und ``y`` jetzt räumlich sind, berechnet **jede** Compute-Kachel nur
noch ``a·b = 2·2 = 4`` Output-Kacheln. Die innere Core-Schleife läuft daher

.. code-block:: text

   for ab in 0..4:            # a aussen, b innen  (war 0..128 in Assignment 09)
       zero(out)
       for c in 0..16:        # K-Akkumulation
           matmul(in0, in1, out)
       release out

Die äußere ``scf.for ... to 4294967295`` (Endlosschleife) sorgt dafür, dass die
Cores dauerhaft konsumieren; die endliche Datenmenge eines vollständigen
Matmuls liefert die ``runtime_sequence``.

Core-Duplikation und FIFO-Suffixe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Der Core-Body wird für alle 32 Kacheln dupliziert. Jede Kachel adressiert ihre
eigenen FIFO-Queues; die Suffixe kodieren Spalte bzw. Zeilenindex:

.. list-table::
   :header-rows: 1

   * - Kachel
     - ``in0`` (Spalte)
     - ``in1`` (Zeile)
     - ``out`` (Spalte_Zeile)
   * - ``aie.tile(0, 2)``
     - ``@in0_L2L1_0``
     - ``@in1_L2L1_0``
     - ``@out_L1L2_0_0``
   * - ``aie.tile(7, 3)``
     - ``@in0_L2L1_7``
     - ``@in1_L2L1_1``
     - ``@out_L1L2_7_1``
   * - ``aie.tile(0, 5)``
     - ``@in0_L2L1_0``
     - ``@in1_L2L1_3``
     - ``@out_L1L2_0_3``

``in0`` hängt nur von der Spalte ab (alle Zeilen einer Spalte teilen sich die
``M``-Kachel), ``in1`` nur vom Zeilenindex (alle Spalten einer Zeile teilen sich
die ``N``-Kachel) — daher tauchen diese Suffixe in Task 2 als Broadcast-Ziele
wieder auf.
