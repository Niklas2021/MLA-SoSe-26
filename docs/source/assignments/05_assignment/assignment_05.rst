Assignment 05: Contraction Interface and L2 Optimization
==========================================================

In diesem Assignment bauen wir ein high-level Konfigurations-Interface für
Tensor-Kontraktionen, einen Optimizer der diese Configs transformiert, und
verwenden beides um eine L2-optimierte cuTile-Implementierung der batched
Matmul ``cmk, ckn -> cmn`` herzuleiten und zu benchmarken.

Alle Tensoren liegen im **FP16-Format** vor, akkumuliert wird in **FP32**.
Es wird Row-Major-Layout angenommen.

Jede Task-Seite enthält:

1. die Aufgabenstellung,
2. unsere Lösung mit Begründung,
3. die vollständige Implementierung,
4. (für Task 4) die gemessene Programmausgabe.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_1_config
   task_2_generate
   task_3_optimizer
   task_4_l2
