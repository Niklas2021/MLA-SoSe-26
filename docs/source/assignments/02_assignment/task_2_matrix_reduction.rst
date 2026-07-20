Task 2: Matrix Reduction Kernel
================================

Aufgabenstellung
----------------

**a)** Ein cuTile-Kernel soll eine 2D-Matrix der Form ``(M, K)`` entlang
der letzten Dimension reduzieren und einen 1D-Ausgabevektor der Form ``(M,)``
mit den zeilenweisen Summen erzeugen.
Die Korrektheit wird gegen ``torch.sum(mat, dim=1)`` geprüft.

**b)** Der theoretische Einfluss steigender oder sinkender ``M``- und
``K``-Dimensionen auf Parallelisierung und Kernel-Last soll erläutert werden.

Implementierte Funktion
-----------------------

.. literalinclude:: ../../../../assignments/02_assignment/src/task2.py
   :language: python

Teilaufgabe a)
---------------

Jeder Block verarbeitet genau eine Zeile: ``grid = (M, 1, 1)`` und
``pid = ct.bid(0)`` liefert den Zeilenindex.

Damit läuft die Parallelisierung über ``M`` Zeilen.

Das Laden der Zeile als Tile der Form ``(1, tile_size)`` ist nötig, weil
cuTile nur Zweierpotenzen als Tile-Dimensionen akzeptiert.
``padding_mode=ct.PaddingMode.ZERO`` nullt OOB-Spalten direkt beim Laden, sodass
sie nicht zur Summe beitragen. Das Ergebnis wird als ``float16`` gespeichert.

Teilaufgabe b)
---------------
 
M-Dimension = Parallelisierungsgrad des Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
``M`` ist direkt die Grid-Größe. 

- **M steigt**: mehr Zeilen = mehr Blöcke = mehr parallele Arbeit auf der GPU. 

- **M sinkt:** Bei kleinem ``M`` gibt es zu wenig Blöcke, um alle SMs auszulasten.

K-Dimension = Arbeit pro Block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 
``K`` bestimmt die **Arbeitslast pro Block** (serielle Arbeit innerhalb
eines Blocks):
 
- **K steigt:** Jeder Block summiert mehr Elemente und der Speichertransfer
  steigt linear. Knapp über Zweierpotenzen ist der Padding-Overhead groß.
 
- **K sinkt:** weniger Arbeit pro Block, potenziell mehr
  verschwendete Tile-Kapazität durch Zero-Padding (z. B. wenn ``K=5``
  → ``tile_size=8``, 37,5 % Overhead).
