Assignment 09: XDNA GEMM
=========================

In diesem Assignment führen wir eine größere Matrixmultiplikation
``out += in0 @ in1`` (mit ``M=256``, ``N=128``, ``K=1024``) auf der NPU aus.
Den hand-geschedulten Tensor-Kernel aus Assignment 08 verwenden wir
unverändert weiter; neu ist der **Data-Movement-Code im MLIR-AIE-Dialekt** und
die Schleifen, die den Kernel über die Tiles iterieren.

Die Matrizen liegen im Hauptspeicher (L3) in Row-major-order
(``in0: MK``, ``in1: KN``, ``out: MN``) und werden beim Transfer nach L1
umstrukturiert. Die Dimensionen werden zerlegt als

- ``M → a·p·m`` mit ``a=16``, ``p=2``, ``m=8``,
- ``N → b·q·n`` mit ``b=8``,  ``q=2``, ``n=8``, und
- ``K → c·r·k`` mit ``c=16``, ``r=8``, ``k=8``.

Das ergibt die Views ``in0: apmcrk``, ``in1: crkbqn`` und ``out: apmbqn``.
Beim Transfer in den L1-Scratchpad wird das Layout zu ``in0: prmk``,
``in1: rqkn`` und ``out: pqmn`` geändert. Die Dimensionen ``a``, ``b`` und
``c`` werden sequenziell über Schleifen auf der Compute-Kachel iteriert; die
DMAs bewegen die zugehörigen Kacheln zwischen den Speicherhierarchien. Vor der
``c``-Schleife wird die Output-Kachel mit Null vorbelegt.

Jede Task-Seite enthält:

1. die Aufgabenstellung,
2. unsere Lösung mit Begründung,
3. (wo vorhanden) die vollständige Implementierung und Programmausgabe.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_0
   task_1
   task_2
   task_3
