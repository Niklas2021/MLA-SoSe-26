Task 1: Dot Product
===================

Aufgabenstellung
----------------

Es soll das Skalarprodukt zweier 1D-Vektoren berechnet werden,
über eine eigene Schleife und nicht nur mit einem fertigen Torch-Call.

Implementierte Funktion
-----------------------

.. literalinclude:: ../../../../assignments/01_assignment/src/assignment_01.py
   :language: python
   :pyobject: dot_product

Unsere Lösung
-------------

Wir iterieren mit einer ``for``-Schleife über alle Einträge,
multiplizieren paarweise und summieren auf.
Die Funktion arbeitet für beliebige Vektorlängen,
sofern beide Eingaben gleich lang und 1D sind.
