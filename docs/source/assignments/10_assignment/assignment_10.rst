Assignment 10: Using the whole NPU
===================================

Aufbauend auf Assignment 09 (eine Compute-Kachel) verteilen wir dieselbe
Matrixmultiplikation ``out += in0 @ in1`` (``M=256``, ``N=128``, ``K=1024``) nun
über **alle 32 Compute-Kacheln** (8 Spalten × 4 Zeilen). Tensor-Kernel
(``matmul.s``) und Zero-Kernel bleiben unverändert; neu ist ausschließlich der
**Data-Movement-Code**, der die Daten räumlich über das Tile-Array verteilt.

Die Matrizen liegen im Hauptspeicher (L3) in Row-major-order
(``in0: MK``, ``in1: KN``, ``out: MN``). Die Dimensionen werden zerlegt als

- ``M → a·x·p·m`` mit ``a=2``, ``x=8``, ``p=2``, ``m=8``,
- ``N → b·y·q·n`` mit ``b=2``, ``y=4``, ``q=2``, ``n=8``, und
- ``K → c·r·k``   mit ``c=16``, ``r=8``, ``k=8``.

Das ergibt die Views ``in0: axpmcrk``, ``in1: crkbyqn`` und ``out: axpmbyqn``.
Der entscheidende Unterschied zu Assignment 09: die Faktoren ``x`` und ``y``
werden nicht mehr sequenziell durchlaufen, sondern **räumlich** auf die Hardware
abgebildet — ``x`` auf die acht Compute-**Spalten**, ``y`` auf die vier
Compute-**Zeilen**. Sequenziell bleiben nur noch ``a``, ``b`` (gemeinsame
``ab``-Schleife der Größe ``a·b = 4``) und ``c`` (innere ``K``-Akkumulation).

Beim Transfer nach L1 wird das Layout zu ``in0: prmk``, ``in1: rqkn``,
``out: pqmn`` geändert. ``in0`` wird entlang der **Spalten** broadcastet (eine
Spalte teilt sich dieselben ``M``-Kacheln), ``in1`` entlang der **Zeilen** (eine
Zeile teilt sich dieselben ``N``-Kacheln). Auf dem Rückweg werden die vier
``out``-Kacheln einer Spalte im Memory-Tile zur Zwischenlage ``ypqmn``
**zusammengeführt (join)** und als ``MN``-Matrix nach L3 zurückgeschrieben.

Jede Task-Seite enthält die Aufgabenstellung sowie unsere Lösung mit Begründung.

.. toctree::
   :maxdepth: 1
   :caption: Tasks

   task_1
   task_2
   task_3
   task_4
   task_5
