Task 5: Buffer Placement (optional)
====================================

Aufgabenstellung
----------------

Die Buffer-Placement-Operationen im gelowerten MLIR
(``build/matmul.mlir.prj/input_with_addresses.mlir``) finden. Beschreiben, wie
man die Buffer platzieren würde, um Bank-Conflicts zu reduzieren, und welche
Änderungen das am XDNA-Tensor-Kernel erfordern würde.

Lösung
------

Gefundene Platzierung
~~~~~~~~~~~~~~~~~~~~~~~

Der ``--alloc-scheme=basic-sequential`` legt die L1-Buffer der Compute-Kachel
(``%tile_0_2``) **fortlaufend** ab. Jeder ObjectFifo ist doppelt gepuffert
(Tiefe 2):

==========================  =========  =========  ==========================
Buffer                       Adresse    Größe      memref
==========================  =========  =========  ==========================
``in1_L2L1 … buff_0``        ``1024``   2048 B     ``8x2x8x8xbf16``
``in1_L2L1 … buff_1``        ``3072``   2048 B     ``8x2x8x8xbf16``
``in0_L2L1 … buff_0``        ``5120``   2048 B     ``2x8x8x8xbf16``
``in0_L2L1 … buff_1``        ``7168``   2048 B     ``2x8x8x8xbf16``
``out_L1L2 … buff_0``        ``9216``    512 B     ``2x2x8x8xbf16``
``out_L1L2 … buff_1``        ``9728``    512 B     ``2x2x8x8xbf16``
==========================  =========  =========  ==========================

Alle sechs Buffer liegen damit in den **untersten ~10 KB** der 64 KB großen L1.

Das Bank-Conflict-Problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die L1-Datenspeicher der Compute-Kachel sind in **Bänke** aufgeteilt (laut
AM027 / Vorlesung vier 512-bit-breite Bänke aus Programmierer-Sicht). Ein
**Bank-Conflict** entsteht, wenn zwei Speicherports im selben Zyklus dieselbe
Bank treffen — die Hardware serialisiert den Zugriff (Stall).

Der Kernel co-issued im Steady-State pro Zyklus eine **``vlda`` (in0, Load-Unit
A)** und eine **``vldb`` (in1, Load-Unit B)** im selben VLIW-Wort (und im
Epilog eine ``vst`` für out). Weil ``in0`` und ``in1`` durch die sequentielle
Allokation in **denselben unteren Bank(s)** liegen, kollidieren diese
gleichzeitigen Lade-Zugriffe → die Hardware fügt Stall-Zyklen ein, und der
Kernel erreicht seine geplante Issue-Rate (``II = 16``) nicht.

Bessere Platzierung
~~~~~~~~~~~~~~~~~~~~

Die drei Operanden in **verschiedene Bänke** legen, sodass die co-issued
Zugriffe nie dieselbe Bank treffen:

- ``in0`` (beide Doppelpuffer) → Bank 0
- ``in1`` (beide Doppelpuffer) → Bank 1
- ``out`` (beide Doppelpuffer) → Bank 2

Die zwei Doppelpuffer-Hälften eines Operanden dürfen sich eine Bank teilen, da
immer nur eine aktiv ist (Ping-Pong). Umsetzen ließe sich das über eine
**explizite Adress-/Bank-Zuweisung** an den Buffern (statt
``basic-sequential``) bzw. einen bank-bewussten Allokator — die Eingabe-Loads
(A/B) und der Output-Store (S) laufen dann konfliktfrei parallel.

Nötige Kernel-Änderungen
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Am MAC-Kern: keine.** Der Kernel adressiert alles **pointer-relativ**
(``p0 = in0``, ``p1 = in1``, ``p2 = out`` mit Post-Increments ``#64``/``#128``);
die Pointer werden vom Framework auf die Buffer-Basisadressen gesetzt. Solange
jeder Buffer **in sich zusammenhängend** bleibt und nur seine *Basis-Bank*
verschoben wird, bleiben alle Offset-Walks des Kernels gültig — der Kernel läuft
unverändert, nur ohne die Bank-Conflict-Stalls.

Erst wenn man einen *einzelnen* Buffer über mehrere Bänke **interleaven** würde
(statt ihn in eine Bank zu legen), müsste das Offset-Muster des Kernels
angepasst werden — genau das vermeiden wir mit der bank-getrennten Platzierung.
