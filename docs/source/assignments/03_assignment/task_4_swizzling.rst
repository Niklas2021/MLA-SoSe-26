Task 4: L2 Cache Optimization via Block Swizzling
===================================================

Aufgabenstellung
----------------

**a)** Ein swizzled Matmul-Kernel – die Anforderungen wie in Task 2, aber
die Block-IDs werden nicht in row-major Reihenfolge auf die Output-Tiles
abgebildet. Stattdessen soll ein Mapping gewählt werden das L2-Cache-Reuse fördert

**b)** Den Tile-Shape-Sweep aus Task 3b für den swizzled Kernel wiederholen.
Beste Tile-Kombination reporten. Dann swizzled vs. row-major Kernel bei
``8192 × 8192 × 4096`` vergleichen.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/03_assignment/src/task4.py
   :language: python

Teilaufgabe a) – Das Swizzle-Mapping
--------------------------------------

**Das Problem mit row-major:**

Aufeinander folgende Blöcke laufen in
derselben Zeile und brauchen jeweils eine andere B-Spalte. Jede B-Spalte wird
also nur einmal gebraucht und dann nicht nochmal, bevor sie aus L2 entfernt wird.
Bei 8192³ ist allein 1 B-Spalte ``8192 × 4096 × 2 Byte ≈ 64 MB`` groß, das geht schon über L2-Cache Größe. Jeder Block lädt seine B-Spalte frisch aus HBM.

**Die Idee hinter dem Swizzle:**

Statt in Zeilen zu laufen, fassen wir ``group_size`` Zeilen-Tiles zu einer Gruppe
zusammen. Innerhalb der Gruppe wechseln wir nicht direkt zur nächsten Zeile,
sondern bleiben erst bei einer Spalte und arbeiten alle ``group_size`` Zeilen für
diese Spalte ab – dann kommt die nächste Spalte dran.

z.B mit ``group_size = 3`` und ``num_col_tiles = 4``:

.. code-block:: text

   BID  row-major        swizzled
   ---  ---------        --------
    0   (Zeile 0, Sp 0)  (Zeile 0, Sp 0)
    1   (Zeile 0, Sp 1)  (Zeile 1, Sp 0)  <- gleiche Spalte
    2   (Zeile 0, Sp 2)  (Zeile 2, Sp 0)  <- gleiche Spalte
    3   (Zeile 0, Sp 3)  (Zeile 0, Sp 1)
    4   (Zeile 1, Sp 0)  (Zeile 1, Sp 1)  <- gleiche Spalte
    5   (Zeile 1, Sp 1)  (Zeile 2, Sp 1)  <- gleiche Spalte
   usw...

BID 0–2 teilen sich die Spalte 0, dasselbe B-Tile bleibt im L2
und wird dreimal wiederverwendet, statt dreimal neu aus HBM geladen zu werden.
Bei row-major braucht BID 1 schon eine andere B-Spalte.

**Das Mapping im Code:**

.. code-block:: python

   bids_per_group = group_size * num_col_tiles

   bid      = ct.bid(0)
   group_id = bid // bids_per_group      # welche Gruppe
   first_m  = group_id * group_size   # wo fängt die Zeile dieser Gruppe an

   #falls num_row_tiles kein Vielfaches von group_size ist
   if (num_row_tiles - first_m < group_size):
        this_group_size = num_row_tiles - first_m
    else: this_group_size = group_size 

   bid_row = first_m + (bid % bids_per_group) % this_group_size
   bid_col = (bid % bids_per_group) // this_group_size

Zeile ist die schnelle Dimension, Spalte die langsame. Damit bleibt B-Tile
einer Spalte so lange im L2, bis alle ``group_size`` zugehörigen Zeilen-Blöcke
fertig sind

**Wie ``group_size`` bestimmt wird:**

``group_size`` soll so groß sein, dass das alle Daten einer Gruppe
in L2-Cache passen. Ein Stripe" besteht aus einem A-Tile (``K × m_tile`` Elemente)
und einem B-Tile (``K × n_tile`` Elemente):

.. code-block:: python

   stripe_bytes = K * (m_tile + n_tile) * bytes_per_element
   group_size   = l2_bytes // stripe_bytes

Falls nicht mal ein einzelner stripe reinpasst (sehr großes K oder Tiles),
wird ``group_size = 1`` gesetzt, das entspricht dann exakt row-major.

**Verification:**

Verifiziert mit ``torch.allclose(..., atol=1e-2, rtol=1e-2)`` bei
``M=N=256``, ``K=4096`` und Tile-Größe ``(64, 64, 64)``.

Teilaufgabe b) – Tile Sweep und Vergleich
------------------------------------------

Benchmark-Setup
^^^^^^^^^^^^^^^

Derselbe Sweep wie in Task 3b: alle 27 Kombinationen von
``m_tile, n_tile, k_tile ∈ {32, 64, 128}`` bei ``2048³`` und ``512³``.
Vor jeder Messung 3 manuelle Warmups, dann ``triton.testing.do_bench``
mit ``warmup=25``, ``rep=200``.

Programmausgabe
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../assignments/03_assignment/out/task4/task4_log.txt
   :language: text


Tile Sweep – Ergebnisse
^^^^^^^^^^^^^^^^^^^^^^^

Beste Konfiguration bei ``2048³``: ``(128, 128, 64)`` → **50.98 TFLOPS**.
Beste Konfiguration bei ``512³``: ``(64, 128, 64)`` → **11.85 TFLOPS**.


Vergleich Swizzled vs. Row-major bei ``8192 × 8192 × 4096``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Row-major: 40.505 ms  ->  13.57 TFLOPS
   Swizzled:   9.290 ms  ->  59.18 TFLOPS
   Speedup:    4.36x

Der Swizzle fasst genug Zeilen-Blöcke zu einer Gruppe zusammen, damit die
B-Spalte noch im L2 liegt, wenn der nächste Block in der Gruppe sie braucht.
Das reduziert HBM Datentransfer


