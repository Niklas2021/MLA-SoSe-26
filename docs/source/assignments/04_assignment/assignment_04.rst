Assignment 04: Tensor Contractions on GPUs
============================================

In diesem Assignment werden GPU-Tensor-Contraction-Kernels mit
`cuTile <https://github.com/nvidia/cutile-python>`_ implementiert und
optimiert. Wir untersuchen den Einfluss von Parallelisierungs-Strategien,
Zusammenfassen von Primitiven (z. B. ``y · l``) und Kernel Fusion auf die
Performance.

Alle Tensoren liegen im **FP16-Format** vor, akkumuliert wird in **FP32**.
Es wird Row-Major-Layout angenommen.

Jede Task-Seite enthält:

1. die Aufgabenstellung,
2. unsere Lösung mit Begründung,
3. die vollständige Implementierung,
4. die gemessene Programmausgabe.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_1
   task_2_fusion
   task_3_sweep
