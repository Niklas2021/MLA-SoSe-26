Assignment 06: Multi-Input Einsum Contraction
==========================================================

In diesem Assignment kontrahieren wir zwei Zwischentensoren einer
Lichtfeld-Tensor-Ring-Zerlegung – zuerst als Referenz mit ``torch.einsum``,
dann mit einem selbst geschriebenen cuTile-Kernel, der durch die
``Config``/``Optimizer``-Schnittstelle aus Assignment 05 konfiguriert wird.

Portable Pfadauflösung
----------------------

Pfade zu ``data/``, ``results/`` und ``05_assignment/src`` werden mit
``Path(__file__)`` relativ zur jeweiligen Quelldatei gebildet. Die Skripte
funktionieren dadurch unabhängig vom aktuellen Arbeitsverzeichnis.

Jede Task-Seite enthält:

1. die Aufgabenstellung,
2. unsere Lösung mit Begründung,
3. die vollständige Implementierung,
4. (wo vorhanden) die gemessene Programmausgabe.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_1
   task_2
   task_3
   task_4
