Task 2: Generating a Basic Config
==================================

Aufgabenstellung
----------------

**a)** ``generate_config`` aus Assignment 05 mit dem Einsum-String und den
Shapes der Eingabetensoren aufrufen, um eine initiale ``Config`` zu erzeugen.

**b)** Die resultierende Config mit allen Feldern reporten.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/06_assignment/src/task2.py
   :language: python

Programmausgabe
---------------

.. literalinclude:: ../../../../assignments/06_assignment/out/task2.log
   :language: text

Teilaufgabe a) – generate_config
----------------------------------

``generate_config`` wird mit Einsum-String und Tensor-Shapes aufgerufen:

.. code-block:: python

   generate_config('acspx,bspy->abcyx',
                   [(4, 3, 64, 64, 1536), (4, 64, 64, 1152)])

Als Datentyp wurde ``FLOAT16`` gewählt.

Teilaufgabe b) – Initiale Config
-----------------------------------

Die Config enthält sieben Dimensionen in der Reihenfolge des Einsum-Strings
(a, c, s, p, x, b, y). Alle ``exec_types`` sind initial ``SEQ``.

Die Index-Typen spiegeln die Klassifikation aus Task 1a wider:

- ``M``: a, c, x (nur in A + Output)
- ``K``: s, p (in A und B, nicht im Output)
- ``N``: b, y (nur in B + Output)

Strides werden row-major berechnet; Dimensionen die in einem Tensor nicht
vorkommen erhalten Stride ``0``.
