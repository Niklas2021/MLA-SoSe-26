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
-------------

Jeder Block verarbeitet genau eine Zeile der Matrix: ``grid = (M, 1, 1)``,
``pid = ct.bid(0)`` liefert dem Block den Index von der Zeile, die er verarbeiten soll.

Damit läuft die Parallelisierung über ``M``Zeilen – jeder Block arbeitet unabhängig

Das Laden der Zeile als Tile der Form ``(1, tile_size)`` ist nötig, weil
cuTile nur Zweierpotenzen als Tile-Dimensionen akzeptiert. 

Ist ``K`` keine Zweierpotenz, lädt ``ct.load`` über die echten Daten hinaus. 

Um das zu korrigieren, bauen wir eine Maske über ``ct.arange``: Indizes ``≥ k`` werden
auf ``0.0`` gesetzt, der Rest bleibt unverändert. Danach ist ``ct.sum`` auf
dem maskierten Tile korrekt – die Padding-Elemente tragen nichts zur Summe
bei.
 
Das Ergebnis wird direkt als ``float16`` via ``ct.store`` an Position ``pid``
in den Ausgabevektor geschrieben
 
----
 
Teilaufgabe b)
-------------------------------------
 
M-Dimension = Parallelisierungsgrad des Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~
 
``M`` ist direkt die Grid-Größe. 

- **M steigt**: mehr Zeilen = mehr Blöcke = mehr parallele Arbeit auf der GPU. 

- **M sinkt:** bei kleinem ``M`` bleiben die meisten SMs idle, weil nicht genug Blöcke da sind, um sie zu befüllen

K-Dimension = Arbeit pro Block
~~~~~~~~~~~~~~~~~~~~~~~~~
 
``K`` bestimmt die **Arbeitslast pro Block** (serielle Arbeit innerhalb
eines Blocks):
 
- **K steigt:** jeder Block muss mehr Elemente summieren. Der
  Speichertransfer steigt linear mit ``K``. Padding-Overhead wächst bei Werten knapp über Zweierpotenz
 
- **K sinkt:** weniger Arbeit pro Block, potenziell mehr
  verschwendete Tile-Kapazität durch Zero-Padding (z. B. wenn ``K=5``
  -> ``tile_size=8``, 37,5 % Overhead!)
