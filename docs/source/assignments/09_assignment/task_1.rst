Task 1: MLIR-AIE Operations
============================

Aufgabenstellung
----------------

Zusammenfassung dieser *mlir-aie*-Operationen:

1. ``aie.tile()``
2. ``aie.core()``
3. ``aie.runtime_sequence()``
4. ``aie.objectfifo()``
5. ``aie.objectfifo.link()``
6. ``aie.objectfifo.acquire()`` und ``aie.objectfifo.release()``
7. ``aiex.npu.dma_memcpy_nd()``
8. ``aiex.npu.dma_wait()``

Lösung
------

``aie.tile()``
~~~~~~~~~~~~~~

Deklariert eine AIE-Kachel anhand ihrer Koordinaten ``(col, row)`` in AIE array
Row 0 = Shim tile
Row 1 = Memory Tile 
Rows ≥ 2 = Compute tiles


``aie.core()``
~~~~~~~~~~~~~~

Deklariert AIEngine prozessor core Modul eines Tiles. 
Der Code wird von MLIR -> LLVM Dialekt -> ELF binary kompiliert.
ELF binary läuft dann auf dem VLIW-Prozessor des Compute 
Tiles.

``aie.runtime_sequence()``
~~~~~~~~~~~~~~~~~~~~~~~~~~
Konfiguriert DMA-Datentransfer zwischen Host und AIE Array (Shim Tile).


``aie.objectfifo()``
~~~~~~~~~~~~~~~~~~~~
Deklariert einen Datenpuffer-Kanal zwischen Tiles (egal ob Compute/Shim oder Memory).
Der eine wird "Producer" genannt (1), der andere "Consumer" (1 oder mehr).
= thread-sichere Warteschlange


``aie.objectfifo.link()``
~~~~~~~~~~~~~~~~~~~~~~~~~

Verbindet zwei ObjectFifos über ein gemeinsames Zwischentile 
sodass der Compiler dort keinen separaten Zwischenspeicher anlegt und keine Kopie stattfindet.


``aie.objectfifo.acquire()`` und ``aie.objectfifo.release()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``acquire`` blockiert solange bis ein Slot verfügbar ist
``release`` gibt die Slots wieder frei
Producer kann erst schreiben wenn Consumer release aufgerufen hat
Consumer kann erst lesen wenn Producer release aufgerufen hat (Synchronisationsmechanismus) 

``aiex.npu.dma_memcpy_nd()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Programmiert einen DMA-Buffer-Descriptor auf einem Shim Tile, der einen Transfer zwischen einem Host-Speicherpuffer (memref) und einem ObjectFifo-Endpunkt (metadata) ausführt
Offsets, Sizes und Strides werden als 4-Tupel angegeben. ``id``-Attribut identifiziert den Buffer-Descriptor. Es gibt 16 Buffer-Descriptor-IDs pro Shim-Tile.


``aiex.npu.dma_wait()``
~~~~~~~~~~~~~~~~~~~~~~~~

Blockiert die Ausführung der Runtime-Sequenz bis der DMA-Transfer des ObjectFifos abgeschlossen ist. Durch gezieltes Platzieren dieser
Erst danach ist garantiert dass der Output-Buffer die fertigen Ergebnisse enthält und sicher gelesen werden kann.