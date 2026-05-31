Assignment 08: XDNA GEMM Kernel
================================

In diesem Assignment schreiben wir einen hand-geschedulten XDNA2-Tensor-Kernel,
der eine Matrixmultiplikation ``out += in0 @ in1`` (mit ``M=16``, ``N=16``,
``K=64``) auf der NPU berechnet. Die Matrizen werden beim Transfer von L3 nach
L1 in 8x8-Kacheln zerlegt; der Kernel führt die Multiplikation mit nativen
``8x8x8``-bfp16-MAC-Operationen aus.

Jede Task-Seite enthält:

1. die Aufgabenstellung,
2. unsere Lösung mit Begründung,
3. (wo vorhanden) die vollständige Implementierung und Programmausgabe.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_1
   task_2
   task_3
   task_4
   task_5
   task_6
