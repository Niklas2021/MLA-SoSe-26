Task 4: Benchmarking Bandwidth
===============================

Aufgabenstellung
----------------

**a)** Ein cuTile-Kernel soll eine 2D-Matrix der Form ``(M, N)`` kacheln
und kopieren. Jedes Kernel-Programm ist für ein Tile der Größe
``(tile_M, tile_N)`` zuständig.

**b)** Die Bandbreite wird für ``M=2048`` und ``N ∈ {16, 32, 64, 128}``
gemessen. Das Tile deckt immer die volle Breite ab (``tile_N = N``).
Die Formel lautet:

.. code-block:: text

   bandwidth (GB/s) = 2 * M * N * sizeof(element) / (time_s * 1e9)

Teil a – Copy-Kernel
--------------------

Implementierter Kernel
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../assignments/02_assignment/src/task4.py
   :language: python
   :pyobject: copy_tile

.. literalinclude:: ../../../../assignments/02_assignment/src/task4.py
   :language: python
   :pyobject: launch_copy

Unsere Lösung
^^^^^^^^^^^^^

Der Kernel liest ein ``(tile_M, tile_N)``-Tile aus der Eingabematrix
und schreibt es direkt in die Ausgabe.
Das 2D-Grid ergibt sich aus ``ceil(M / tile_M)`` mal ``ceil(N / tile_N)``.
Die Korrektheit wurde mit ``cp.allclose`` gegen die Eingabematrix verifiziert.

Teil b – Bandwidth-Sweep
-------------------------

Implementierung
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../assignments/02_assignment/src/task4.py
   :language: python
   :pyobject: bench_kernel

.. literalinclude:: ../../../../assignments/02_assignment/src/task4.py
   :language: python
   :pyobject: task4b

Programmausgabe
^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../assignments/02_assignment/out/task4/task4.log
   :language: text

Messergebnisse
^^^^^^^^^^^^^^

Gemessen auf dem DGX Spark (CUDA 13.0), je 200 Runs nach 10 Warmup-Durchläufen:

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - N
     - Datenmenge (R+W)
     - Laufzeit
     - Effektive Bandbreite
   * - 16
     - 0.13 MB
     - 6.38 µs
     - 20.53 GB/s
   * - 32
     - 0.25 MB
     - 6.40 µs
     - 40.98 GB/s
   * - 64
     - 0.50 MB
     - 6.18 µs
     - 84.80 GB/s
   * - 128
     - 1.05 MB
     - 6.32 µs
     - 165.93 GB/s

.. image:: ../../_static/task4_bw.png
   :alt: Copy Kernel Bandbreite – M=2048, N=16..128
   :width: 100%

Die Laufzeit bleibt über alle N nahezu konstant bei ca. 6.3 µs –
der Kernel ist für diese kleinen Matrixgrößen vollständig
Kernel-Launch-Overhead-dominiert. Die scheinbar steigende Bandbreite
ist kein echter Skalierungseffekt, sondern eine Folge der
Overhead-Amortisierung: doppelt so viele Daten im selben Zeitfenster
ergeben doppelt so hohe scheinbare Bandbreite.

Zusatz: Erweiterter Sweep *(nicht Teil der Bewertung)*
-------------------------------------------------------

Um das Overhead-Regime sichtbar von echtem HBM-Zugriff abzugrenzen,
wurde derselbe Kernel zusätzlich für ``M=8192`` und
``N ∈ {16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`` gemessen.

.. literalinclude:: ../../../../assignments/02_assignment/out/task4_bonus_test/task4_bonus_test.log
   :language: text

Der Plot zeigt log-skaliert drei klar trennbare Regime:
Overhead-dominiert (N ≤ 128, konstante Laufzeit ~0.004 ms),
L2-Cache (N = 256–512, Bandbreite bis ~1010 GB/s)
und HBM (N ≥ 1024, stabile ~220–240 GB/s).

.. image:: ../../_static/task4_bonus_bw.png
   :alt: Erweiterter Bandwidth-Sweep – M=8192, N=16..4096
   :width: 100%
