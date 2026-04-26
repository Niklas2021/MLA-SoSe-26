Task 1: FP32 vs FP16 Performance
============================================


Vollständige Implementierung
----------------------------


.. literalinclude:: ../../../../assignments/03_assignment/src/task1.py
   :language: python

Erklärung
-------------

Beide Kernels haben dieselbe Struktur – der einzige Unterschied ist der Eingabe-Datentyp.
``kernel_fp16`` bekommt FP16-Tensoren, ``kernel_fp32`` bekommt FP32-Tensoren.
In beiden Fällen wird in FP32 akkumuliert.

.. code-block:: python

    def kernel_fp16(a, b, acc,
                m: ct.Constant[int],
                iterate: ct.Constant[int]):

    acc_tile = ct.full((m, m), fill_value=0, dtype=ct.float32)

    for i in range(iterate):
        a_tile = ct.load(a, index=(0, i), shape=(m, m))
        b_tile = ct.load(b, index=(i, 0), shape=(m, m))
        acc_tile = ct.mma(a_tile, b_tile, acc_tile)

    ct.store(acc, (0, 0), tile=acc_tile) 

Pro Iteration wird ein ``(64, 64)``-Tile aus ``A`` (Zeile ab Index 0, Spalte ``i``)
und ein ``(64, 64)``-Tile aus ``B`` (Zeile ``i``, Spalte ab index 0) geladen.
``ct.mma`` macht daraus eine Tensor-Core mma Instruktion und akkumuliert das Ergebnis.



**Single CTA:**

.. code-block:: python

   grid = (1,)
   iterate = n // m  # = 4096 // 64 = 64

Wir starten genau einen Block. Das Output ``C`` hat die Form ``(64, 64)``
ein einziges Tile, das vollständig von einen Block berechnet wird


Output
-------------

.. literalinclude:: ../../../../assignments/03_assignment/out/task1/task1_log.txt
   :language: text


Benchmark-Ergebnis
------------------

FP16 ist um Faktor **60× schneller** als FP32. Der Grund ist
dass die Tensor Cores auf modernen NVIDIA-GPUs nativ für FP16-Eingaben optimiert sind.
Der Speedup fällt besonders auf, weil die gesamte Arbeit in einem einzigen Block stattfindet, es gibt
keine Grid-Parallelität, die den Unterschied abfedern könnte
