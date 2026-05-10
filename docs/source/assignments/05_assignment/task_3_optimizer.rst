Task 3: Optimizer Class
========================

Aufgabenstellung
----------------

Eine Klasse ``Optimizer``, die eine ``Config`` umschließt und Methoden bietet,
sie zu transformieren.

- **a)** ``split_dim(dim_id, outer_size, inner_size)``
- **b)** ``fuse_dims(dim_id_a, dim_id_b)``
- **c)** ``permute_dims(permutation)``
- **d)** ``make_executable()``
- **e)** ``verify()``

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/05_assignment/src/task3.py
   :language: python

Teilaufgabe a) – ``split_dim``
-------------------------------

Eine Dimension der Größe ``S`` wird in zwei Dimensionen ``(outer, inner)``
geteilt mit ``outer * inner = S``. Strides werden so angepasst, dass das
Speicher-Layout unverändert bleibt:

- ``stride_outer = stride_original * inner_size``
- ``stride_inner = stride_original``

Wenn die Dim in einem Tensor gar nicht vorkommt (Stride ``0``), bleiben beide
neuen Strides ebenfalls ``0``. Beide neuen Dims erben ``dim_type`` und
``exec_type`` der ursprünglichen Dimension.

Teilaufgabe b) – ``fuse_dims``
-------------------------------

Zwei Dimensionen können nur fusioniert werden, wenn sie in **jedem** Tensor,
in dem beide vorkommen, im Speicher direkt aufeinander folgen:

.. code-block:: text

   stride[a] == stride[b] * size[b]      # a ist die äußere Dim
   stride[a] * size[a] == stride[b]      # b ist die äußere Dim

Diese Bedingung wird vor der Fusion über alle Tensoren geprüft – schlägt sie
fehl, gibt es einen ``ValueError``. Nach der Fusion ist die neue Größe
``size[a] * size[b]``, der neue Stride ist der innere (kleinere) der beiden,
und die fusionierte Dim erbt ``dim_type``/``exec_type`` von ``a``.

Teilaufgabe c) – ``permute_dims``
----------------------------------

Reordert ``dim_types``, ``exec_types``, ``dim_sizes`` und jede Stride-Liste
gemäß ``permutation`` (Konvention wie ``torch.permute``: ``permutation[i]``
ist der alte Index für die neue Position ``i``).

Teilaufgabe d) – ``make_executable``
-------------------------------------

Setzt ``exec_types`` und ordnet die Dimensionen so um, dass die Config
ausführbar ist:

1. Pro ``DimType`` (M, N, K) wird die **rechteste** Dim als ``PRIM`` markiert
   – das sind die Dimensionen, die das (B)GEMM-Primitive nutzt.
2. Alle anderen K-Dims werden ``SEQ`` (Kontraktionsschleife im Kernel).
3. Alle anderen Dims (M, N, C, die nicht PRIM sind) werden ``PAR``.
4. Schließlich werden die Dims so umsortiert, dass die Reihenfolge
   ``PAR | SEQ | PRIM`` gilt.

Am Ende wird ``verify()`` aufgerufen.

Teilaufgabe e) – ``verify``
----------------------------

Prüft die vier Bedingungen aus der Aufgabenstellung und wirft pro verletzter
Bedingung einen aussagekräftigen ``ValueError``:

1. Keine ``K``-Dim darf ``PAR`` sein.
2. Alle ``SEQ``-Dims liegen links von allen ``PRIM``-Dims.
3. Alle ``PAR``-Dims liegen links von allen ``SEQ``-Dims.
4. Die rechtesten Dims sind ``PRIM`` und decken mindestens je eine ``M``-,
   ``N``- und ``K``-Dim ab.

Bedingungen 2 und 3 werden zusammen über die Reihenfolge ``PAR < SEQ < PRIM``
geprüft.
