Task 2: Generating a Basic Config
==================================

Aufgabenstellung
----------------

Eine Funktion ``generate_config(einsum_str, input_shapes)`` die aus einem
Einsum-String und den Shapes der Input-Tensoren eine Basis-``Config`` produziert.

Anforderungen:

- Jede Dimension wird automatisch klassifiziert (``M``/``N``/``K``/``C``),
  je nachdem in welchen Tensoren sie vorkommt.
- Strides werden für jeden Tensor in **Row-Major-Layout** berechnet.
  Eine fehlende Dimension bekommt Stride ``0``.
- Alle ``exec_types`` sind initial ``SEQ``.
- Defaults: ``FLOAT16``, ``GEMM``, ``NONE``, ``ZERO``.

Lösung
------

Klassifikation
^^^^^^^^^^^^^^

Für eine Kontraktion mit zwei Inputs (``A``, ``B``) und Output (``C``):

- ``C``: kommt in **A, B und Output** vor (Batch-Dimension)
- ``K``: kommt in **A und B**, aber **nicht im Output** (Kontraktion)
- ``M``: kommt in **A und Output**, aber nicht in B
- ``N``: kommt in **B und Output**, aber nicht in A

Beispiel ``cmk, ckn -> cmn``:

.. code-block:: text

   c -> C   (in A, B, Output)
   m -> M   (in A, Output, nicht B)
   k -> K   (in A, B, nicht Output)
   n -> N   (in B, Output, nicht A)

Strides
^^^^^^^

Pro Tensor wird der Stride von rechts aufgebaut: das letzte Element bekommt
Stride ``1``, alle weiteren werden mit den Größen der rechts liegenden Dims
multipliziert. Für jede globale Dimension wird dann der passende Stride
ausgesucht oder ``0`` gesetzt, wenn die Dim im Tensor nicht vorkommt.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/05_assignment/src/task2.py
   :language: python
