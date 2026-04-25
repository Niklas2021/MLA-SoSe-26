Task 2: Simple Matrix Multiplication Kernel
============================================

Aufgabenstellung
----------------

Ein cuTile-Kernel soll eine Matrixmultiplikation ``C = A @ B`` berechnen,
mit ``A`` der Form ``(M, K)``, ``B`` der Form ``(K, N)`` und ``C`` der Form
``(M, N)``.

**Anforderungen:**

* Jedes Kernel-Programm berechnet **ein** Output-Tile der Form
  ``(m_tile, n_tile)``.
* Die Tile-Größen werden vom Aufrufer übergeben.
* Die Block-IDs werden in **row-major** Reihenfolge auf die Output-Tiles
  abgebildet (BID 0 = oben links, BID 1 = rechts daneben, am Zeilenende
  wird umgebrochen).
* Der Kernel muss auch **nicht-zweierpotente** Matrixformen unterstützen.
* Für die innere Akkumulation soll ``ct.mma`` verwendet werden.

Verifikation gegen ``torch.matmul(A, B)`` mit ``torch.allclose``.

Unsere Lösung
-------------

**Kernel-Signatur:**

.. code-block:: python

   @ct.kernel
   def matmul_kernel(A, B, C,
                     num_n_tiles: ct.Constant[int],
                     num_k_tiles: ct.Constant[int],
                     m_tile: ct.Constant[int],
                     n_tile: ct.Constant[int],
                     k_tile: ct.Constant[int]):

Die Tile-Größen und die Anzahl der n/k-Tiles werden als ``ct.Constant[int]``
übergeben, weil cuTile sie zur Compile-Zeit kennen muss – sonst funktioniert
``ct.zeros((m_tile, n_tile), ...)`` nicht. ``num_m_tiles`` brauchen wir
intern nicht, weil der Kernel selbst nur sein eigenes Output-Tile berechnet.

**Row-major Mapping der Block-IDs:**

.. code-block:: python

   pid   = ct.bid(0)
   bid_m = pid // num_n_tiles
   bid_n = pid %  num_n_tiles

Wir starten den Kernel mit einem 1D-Grid der Größe
``num_m_tiles * num_n_tiles`` und teilen den linearen Block-Index in eine
2D-Position auf. So liegt BID 0 auf dem oberen linken Tile, BID 1 auf dem
Tile rechts daneben, am Zeilenende wird automatisch umgebrochen – das
entspricht genau der row-major Vorgabe aus der Aufgabe.

**Akkumulator initialisieren:**

.. code-block:: python

   acc = ct.zeros((m_tile, n_tile), dtype=ct.float32)

Der Akku liegt in FP32, obwohl die Inputs FP16 sind. Genau dafür sind die
Tensor Cores ausgelegt (Folie 6: "Mixed Precision mit automatischer
Akkumulation in höherer Genauigkeit"). Würden wir in FP16 akkumulieren,
hätten wir bei großen ``K`` deutliche Genauigkeitsprobleme.

**K-Loop mit Tensor Cores:**

.. code-block:: python

   for k in range(num_k_tiles):
       a_tile = ct.load(A, index=(bid_m, k), shape=(m_tile, k_tile))
       b_tile = ct.load(B, index=(k, bid_n), shape=(k_tile, n_tile))
       acc = ct.mma(a_tile, b_tile, acc)

Pro Iteration laden wir ein ``(m_tile, k_tile)``-Tile aus ``A`` und ein
``(k_tile, n_tile)``-Tile aus ``B``, und übergeben die zwei Tiles plus
den Akku an ``ct.mma``. Das ist die High-Level-API aus Folie 11 – unter
der Haube wird daraus eine einzige Tensor-Core-MMA-Instruktion
(``D = A @ B + C``). Genau das löst auch die Probleme aus Folie 5:
mehrere Threads laden nicht mehr unnötig die gleichen A/B-Elemente,
und die Reduktion läuft nicht mehr Element-für-Element ab.

**Store des Output-Tiles:**

.. code-block:: python

   ct.store(C, index=(bid_m, bid_n), tile=acc)

Nach der k-Schleife enthält ``acc`` das fertige Output-Tile, das wir an
die richtige 2D-Position in ``C`` schreiben.

Padding-Wrapper für non-Power-of-2 Shapes
-----------------------------------------

cuTile erlaubt als Tile-Größen nur Zweierpotenzen, und der Kernel rechnet
mit der Annahme dass die Matrixgrößen Vielfache der Tile-Größe sind. Damit
auch beliebige ``M, K, N`` funktionieren, paddet der Wrapper ``A`` und ``B``
mit Nullen auf das nächste Vielfache:

.. code-block:: python

   M_pad = ceildiv(M, m_tile) * m_tile
   N_pad = ceildiv(N, n_tile) * n_tile
   K_pad = ceildiv(K, k_tile) * k_tile

   if M_pad != M or K_pad != K:
       A_p = torch.zeros((M_pad, K_pad), dtype=A.dtype, device=A.device)
       A_p[:M, :K] = A

Am Ende wird nur der gültige Slice ``C_p[:M, :N]`` zurückgegeben. Da die
Padding-Zeilen/Spalten nur Nullen enthalten, tragen sie nichts zur
Multiplikation bei – mathematisch sauber. Den gleichen Trick haben wir
schon in Assignment 02 Task 4 verwendet.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/03_assignment/src/task2.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/03_assignment/out/task2/task2_log.txt
   :language: text

Korrektheit ist sowohl für Zweierpotenz-Formen (256³, 1024×512×768) als auch
für nicht-zweierpotente Formen (100×200×300, 513×257×129, 1000×999×777)
bestätigt. Maximaler absoluter Fehler ≤ 3·10⁻⁴, was für FP16-Inputs mit
FP32-Akkumulation absolut im Rahmen ist.
