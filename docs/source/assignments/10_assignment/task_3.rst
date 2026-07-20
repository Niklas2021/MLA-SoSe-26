Task 3: Writing the Output
==========================

Aufgabenstellung
----------------

Die benötigten ``L1L2``- und ``L2L3``-Output-FIFOs anlegen, die ``L1L2``-FIFOs
einer Spalte zur zugehörigen ``L2L3``-FIFO **zusammenführen (join)** und die
Output-``dma_memcpy_nd``-Operationen mit passenden Sizes/Strides/Offsets
anpassen.

Lösung
------

FIFOs und Join
~~~~~~~~~~~~~~~

Jede der 32 Compute-Kacheln produziert ihre ``out``-Kachel im Layout ``pqmn``
(``2×2×8×8``) über einen eigenen ``@out_L1L2_<x>_<y>``-FIFO. Pro Spalte ``x``
werden die **vier** Zeilen-FIFOs auf einen gemeinsamen ``@out_L2L3_<x>``
gejoint:

.. code-block:: text

   aie.objectfifo.link
     [@out_L1L2_x_0, @out_L1L2_x_1, @out_L1L2_x_2, @out_L1L2_x_3]
     -> [@out_L2L3_x] ([0, 256, 512, 768] [])

Die Liste ``[0, 256, 512, 768]`` sind die **src-Offsets** des Joins: Zeile ``y``
schreibt ihren 256-Elemente-Block an Offset ``y·256`` in den gemeinsamen
1024-Elemente-Puffer. Dadurch entsteht im Memory-Tile die Zwischenlage
``ypqmn``.

Layout-Wechsel ``ypqmn`` → ``ypmqn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Der Join erzeugt im Memory-Tile ``ypqmn``. Der gemeinsame FIFO-Typ
``memref<4x16x16xbf16>`` beschreibt ``y(pm)(qn)``. ``y`` wird in
``dimensionsToStream`` **nicht angegeben und nicht mit ``p`` verschmolzen**;
die Einträge ordnen nur innerhalb jedes ``y``-Blocks ``pqmn`` → ``pmqn`` um:

.. list-table::
   :header-rows: 1

   * - Dimension
     - ``<size, stride>``
     - Bemerkung
   * - ``p``
     - ``<2, 128>``
     -
   * - ``m``
     - ``<8, 8>``
     - vorgezogen vor ``q`` (Swap ``q`` ↔ ``m``)
   * - ``q``
     - ``<2, 64>``
     -
   * - ``n``
     - ``<8, 1>``
     -

Der maximale Zugriffsindex ist ``1·128 + 7·8 + 1·64 + 7 = 255`` — also innerhalb
eines Join-Inputs. Für jedes unverändert äußere ``y`` wird damit ein
``pmqn``-Block gestreamt; erst auf dem Stream ergibt sich folglich ``ypmqn``.

Output-Shim-DMA
~~~~~~~~~~~~~~~~

Der Shim erhält pro Spalte den Tensor in der Sicht ``aby(pm)(qn)`` (``x`` ist
räumlich und taucht nicht auf). Wir geben **einen Deskriptor pro ``a``** aus und
halten ``b`` und ``y`` getrennt:

.. list-table::
   :header-rows: 1

   * - Tensor
     - sizes ``[…]``
     - strides ``[…]``
     - Basis-Offset
   * - ``out_x``
     - ``[b=2, y=4, pm=16, qn=16]``
     - ``[64, 16, 128, 1]``
     - ``a·16384 + x·2048``

Herleitung der Strides aus ``out``-Element-Offset
``(a·128 + x·16 + pm)·128 + (b·64 + y·16 + qn)``:

- ``b`` → ``N``-Index ``+64`` → Stride ``64``
- ``y`` → ``N``-Index ``+16`` → Stride ``16``
- ``pm`` → ``M``-Index ``+1`` → Offset ``+128`` (eine ``N``-Zeile)
- ``qn`` → ``N``-Index ``+1`` → Stride ``1``

Basis: ``x·16·128 = x·2048`` (``M``-Streifen der Spalte), ``a``-Sprung
``128·128 = 16384``. Über alle 8 Spalten und beide ``a`` ergibt sich genau die
volle ``256×128``-Matrix, überlappungsfrei. Die ``out``-IDs sind ``0`` (a=0) und
``8`` (a=1).

Synchronisation
~~~~~~~~~~~~~~~

Zwischen den Deskriptoren für ``a=0`` und ``a=1`` ist keine Barriere nötig.
Beide Blöcke verwenden unterschiedliche BD-IDs (``0/1/2`` beziehungsweise
``8/9/10``) und werden nacheinander an die DMA-Queues übergeben.

Jede Output-FIFO erhält zwei Transfers. Daher stehen am Sequenzende zwei
Runden mit je acht ``dma_wait``. Sie warten beide Output-Hälften ab, ohne eine
Barriere zwischen ``a=0`` und ``a=1`` zu bilden.

Output-FIFOs und Joins (Auszug):

.. literalinclude:: ../../../../assignments/10_assignment/src/matmul.mlir
   :language: text
   :lines: 112-172
