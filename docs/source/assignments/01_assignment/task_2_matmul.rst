Task 2: Matrix-Matrix-Multiplikation
====================================

Aufgabenstellung
----------------

Die Multiplikation ``C = A @ B`` soll in zwei Varianten umgesetzt werden:

1. direkt mit verschachtelten Schleifen,
2. mit Wiederverwendung von Task 1 (Dot-Product-Idee auf Zeile/Spalte).

Implementierte Funktionen
-------------------------

.. literalinclude:: ../../../../assignments/01_assignment/src/assignment_01.py
   :language: python
   :pyobject: matmul_loops

.. literalinclude:: ../../../../assignments/01_assignment/src/assignment_01.py
   :language: python
   :pyobject: matmul_dot

Unsere Lösung
-------------

``matmul_loops`` baut ``C`` über drei Schleifen auf.
``matmul_dot`` nimmt jeweils eine Zeile aus ``A`` und eine Spalte aus ``B``
und ruft dafür ``dot_product`` auf.
Beide Varianten sind korrekt gegen ``torch.matmul`` getestet.
