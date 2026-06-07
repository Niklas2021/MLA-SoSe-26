Task 2: Data Layouts and Loops
===============================

Aufgabenstellung
----------------

Die Datenbewegung zwischen Hauptspeicher (L3), Shim-Tile, Memory-Tile (L2) und
Compute-Tile (L1) skizzieren und beschreiben, welche *mlir-aie*-Operation in
jedem Schritt beteiligt ist.

Lösung
------

Überblick
~~~~~~~~~

Die Daten durchlaufen drei Speicherebenen, in beide Richtungen über je ein Paar
verlinkter ObjectFIFOs. Die eigentliche **Layout-Umordnung** (Tiling) passiert
an zwei Stellen: im Shim-DMA (Row-major → ``apmcrk`` / ``crkbqn``) und im
Memory-Tile-DMA (→ L1-Layout ``prmk`` / ``rqkn``). Das Memory-Tile dient dabei
nur als Relay-Puffer (L2), das per ``aie.objectfifo.link`` ohne explizite Kopie
durchgereicht wird.

.. code-block:: text

   L3  Hauptspeicher (row-major)
       in0: MK 256x1024     in1: KN 1024x128            out: MN 256x128
        |  ^                                              ^  |
        |  |  aiex.npu.dma_memcpy_nd  (Shim-DMA, 4D)      |  |  aiex.npu.dma_memcpy_nd
        |  |  erzeugt Tiling-View apmcrk / crkbqn         |  |  + aiex.npu.dma_wait
        v  |                                              |  v
   Shim-Tile (0,0)   @in0_L3L2_0 / @in1_L3L2_0           @out_L2L3_0
        |  ^   aie.objectfifo (Producer Shim -> Consumer Mem)     ^
        |  |   aie.objectfifo.link  (Relay, keine Kopie)          |  link
        v  |                                                      |
   Memory-Tile (0,1) / L2   @in0_L2L1_0 / @in1_L2L1_0      @out_L1L2_0_0
        |  ^   aie.objectfifo  mit  dimensionsToStream            ^
        |  |   Mem-DMA-Strides -> Layout-Wechsel zu prmk / rqkn   |  (pqmn)
        v  |                                                      |
   Compute-Tile (0,2) / L1
        acquire -> zero -> (c-Loop: acquire in0,in1 -> matmul -> release) -> release

Schritte und beteiligte Operationen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

============================  =================================================  ===================================================
Schritt                       *mlir-aie*-Operation                               Funktion
============================  =================================================  ===================================================
L3 → Shim (Eingaben)          ``aiex.npu.dma_memcpy_nd``                         Programmiert den Shim-DMA mit 4D-Offset/Size/Stride;
                                                                                 liest die Row-major-Matrix und erzeugt die
                                                                                 Tiling-View ``apmcrk`` (in0) bzw. ``crkbqn`` (in1).
Shim → Memory (L2)            ``aie.objectfifo`` (``@in*_L3L2_0``)               Producer = Shim, Consumer = Memory-Tile;
                              + ``aie.objectfifo.link``                          ``link`` koppelt L3L2- und L2L1-FIFO, sodass das
                                                                                 Memory-Tile nur durchreicht (keine Zwischenkopie).
Memory (L2) → Compute (L1)    ``aie.objectfifo`` mit ``dimensionsToStream``      Der Mem-Tile-DMA streamt die Kachel mit Strides
                              (``@in*_L2L1_0``)                                  und ordnet das Layout zum L1-View ``prmk`` (in0)
                                                                                 bzw. ``rqkn`` (in1) um.
Compute: Puffer holen         ``aie.objectfifo.acquire`` /                       Holt einen L1-Slot (blockiert bis verfügbar) und
                              ``aie.objectfifo.subview.access``                  liefert den ``memref`` für den Kernel-Aufruf.
Compute: Rechnen              ``func.call @zero`` / ``func.call @matmul``        Zero-Init der Output-Kachel, danach
                                                                                 ``8x8x8``-bfp16-MAC pro ``c``-Iteration.
Compute: Puffer freigeben     ``aie.objectfifo.release``                         Gibt die Eingabe-Slots (pro ``c``) bzw. den
                                                                                 Output-Slot (nach der ``c``-Schleife) frei.
Compute → Memory (L2)         ``aie.objectfifo`` (``@out_L1L2_0_0``)             Producer = Compute, Consumer = Memory-Tile.
Memory → Shim                 ``aie.objectfifo`` mit ``dimensionsToStream``      Ordnet ``pqmn`` zurück Richtung Matrix-Layout
                              (``@out_L2L3_0``) + ``aie.objectfifo.link``        und reicht über ``link`` zum Shim durch.
Shim → L3 (Ausgabe)           ``aiex.npu.dma_memcpy_nd`` +                       Schreibt das Ergebnis als ``MN``-Matrix zurück;
                              ``aiex.npu.dma_wait``                              ``dma_wait`` blockiert, bis der Output-Transfer
                                                                                 fertig ist (erst dann ist L3 gültig lesbar).
============================  =================================================  ===================================================

Schleifen über ``a``, ``b``, ``c``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die drei zerlegten Dimensionen werden sequenziell abgearbeitet; eine
Compute-Kachel (``aie.tile(0,2)``) berechnet zu jedem Zeitpunkt eine
``out``-Kachel ``[p][q][m][n]`` für ein festes ``(a,b)``:

- ``a`` und ``b`` bilden die **äußere** Schleife des Cores
  (``a·b = 16·8 = 128`` Iterationen). Pro Iteration liefert der DMA-Strom über
  die ObjectFIFOs frische ``in0``- und ``in1``-L1-Kacheln. Da ``in0`` nicht von
  ``b`` und ``in1`` nicht von ``a`` abhängt, müssen die DMA-Zugriffsmuster die
  jeweils unabhängige Kachel über die andere Dimension wiederholen (Repeat per
  ``stride = 0``, siehe Task 3).
- ``c`` ist die **innere** Schleife (``c = 16`` Iterationen) und akkumuliert
  entlang ``K``. Genau dafür wird die Output-Kachel **vor** der ``c``-Schleife
  per ``func.call @zero`` auf Null gesetzt; danach addieren die ``matmul``-Calls
  über alle ``c`` in dieselbe L1-Kachel.

Schematisch (entspricht ``aie.core`` in ``src/matmul.mlir``):

.. code-block:: text

   for ab in 0..128:                  # a- und b-Schleife (äußere)
       acquire out (Produce)
       zero(out)                      # Output-Kachel nullen
       for c in 0..16:                # K-Akkumulation (innere)
           acquire in0, in1 (Consume)
           matmul(in0, in1, out)      # out += in0 @ in1  (8x8x8-bfp16-MACs)
           release in0, in1 (Consume)
       release out (Produce)          # fertige Kachel -> L2 -> L3

Die FIFO-Tiefe von ``2`` ermöglicht Doppelpufferung (Double Buffering): während
der Core auf einer Kachel rechnet, kann der DMA bereits die nächste Kachel laden
bzw. die fertige wegschreiben. Wie sich daraus eine durchgehend nicht-blockierende
Datenbewegung ergibt, behandelt Task 4.
