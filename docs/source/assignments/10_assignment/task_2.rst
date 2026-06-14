Task 2: Broadcasting the Inputs
===============================

Aufgabenstellung
----------------

Die bestehenden ``L2L1``-Input-FIFOs um die zusätzlichen Consumer-Kacheln
(Zeile 0 für ``in0``, Spalte 0 für ``in1``) erweitern, die fehlenden ``L2L1``-
und ``L3L2``-FIFOs für die übrigen Spalten/Zeilen anlegen und die Input-
``dma_memcpy_nd``-Operationen an ihre FIFO-Queues anpassen.

Lösung
------

Broadcast über die Consumer-Liste
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ein ObjectFifo broadcastet automatisch an **alle** Consumer in seiner Liste.
Damit ist der Broadcast rein deklarativ:

- ``in0`` (Spalten-Broadcast): ein FIFO je Spalte ``x`` mit allen vier
  Compute-Kacheln dieser Spalte als Consumer, z. B.

  .. code-block:: text

     @in0_L2L1_0 : {tile_0_2, tile_0_3, tile_0_4, tile_0_5}   # ganze Spalte 0

- ``in1`` (Zeilen-Broadcast): ein FIFO je Zeilenindex ``y`` mit allen acht
  Compute-Kacheln dieser Zeile als Consumer, z. B.

  .. code-block:: text

     @in1_L2L1_0 : {tile_0_2, tile_1_2, ..., tile_7_2}        # ganze Zeile 0

So erhält jede Spalte eine andere ``in0``-Kachel und jede Zeile eine andere
``in1``-Kachel. Insgesamt brauchen wir **8** ``in0``- und **4** ``in1``-FIFO-
Paare (``L3L2`` → ``L2L1``). Die vier ``in1_L3L2``-Queues platzieren wir frei auf
den Shim-Tiles 0–3.

Shim-Belegung
~~~~~~~~~~~~~

Jedes Shim-Tile besitzt zwei DMA-Input-Streams. Die Verteilung bleibt darunter:

.. list-table::
   :header-rows: 1

   * - Shim
     - Input-FIFOs (L3→L2)
     - # Input-Streams
   * - ``0..3``
     - ``in0_L3L2_x`` + ``in1_L3L2_x``
     - 2
   * - ``4..7``
     - ``in0_L3L2_x``
     - 1

Anpassung der Shim-DMAs (``runtime_sequence``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die Layout-Umordnung zum L1-View (``prmk`` / ``rqkn``) steckt unverändert in den
``dimensionsToStream`` der ``L2L1``-FIFOs. Im Shim-DMA setzen wir nur
Offset/Size/Stride. Da ``x`` und ``y`` räumlich sind, bleibt pro Eingabe nur die
Sequenz über ``a`` (= 2). Wir geben **einen Deskriptor pro ``a`` und pro Shim**
aus:

.. list-table::
   :header-rows: 1

   * - Tensor
     - sizes ``[…]``
     - strides ``[…]``
     - Basis-Offset
   * - ``in0_x``
     - ``[b=2, c=16, pm=16, rk=64]``
     - ``[0, 64, 1024, 1]``
     - ``(a·128 + x·16)·1024``
   * - ``in1_y``
     - ``[b=2, c=16, rk=64, qn=16]``
     - ``[64, 8192, 128, 1]``
     - ``y·16`` (pro ``a`` erneut)

- ``in0`` hängt nicht von ``b`` ab → der ``b``-Repeat steht als ``stride = 0``
  ganz außen (Hardware erlaubt ``stride 0`` nur in der äußersten Dimension). Der
  Basis-Offset wählt über ``x·16`` den ``M``-Streifen der Spalte und über
  ``a·128`` den ``a``-Block.
- ``in1`` hängt nicht von ``a`` ab → wir senden es in jeder ``a``-Iteration
  erneut (die volle ``B``-Matrix fließt zweimal). ``y·16`` wählt den
  ``N``-Streifen der Zeile.

Buffer-Descriptor-IDs
~~~~~~~~~~~~~~~~~~~~~~

``a=0`` nutzt die IDs ``1`` (in0) und ``2`` (in1), ``a=1`` die IDs ``9`` und
``10``. Da unterschiedliche Shim-Tiles dieselben IDs wiederverwenden dürfen,
genügt dieses Schema pro Shim und bleibt weit unter den 16 verfügbaren BDs —
ohne ID-Wiederverwendung innerhalb der Sequenz ist keine Zwischen-Barriere nötig.
