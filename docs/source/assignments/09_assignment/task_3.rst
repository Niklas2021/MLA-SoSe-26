Task 3: Implementation
=======================

Aufgabenstellung
----------------

Die Datenbewegung in ``src/matmul.mlir`` implementieren und die Dimensionsgrößen
in ``src/driver.py`` anpassen (die *TODOs* ersetzen). Anschließend mit
``make run_matmul`` verifizieren.

Lösung
------

Tiling und Schleifen
~~~~~~~~~~~~~~~~~~~~~

Eine Compute-Kachel berechnet für ein festes ``(a,b)`` die Output-Kachel
``out[p,q,m,n]``. Der Core iteriert ``a·b = 16·8 = 128`` mal (äußere Schleife)
und akkumuliert je ``c = 16`` mal (innere Schleife) entlang ``K``; vor der
``c``-Schleife wird die Kachel per ``@zero`` genullt. Die globale
Konsumreihenfolge ist damit ``a`` außen, ``b`` Mitte, ``c`` innen — und genau
diese Reihenfolge müssen die DMAs bedienen.

Datenbewegung in der ``runtime_sequence``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pro Eingabe wird der ``in*_L3L2`` → ``in*_L2L1``-FIFO-Pfad benutzt; die
Layout-Umordnung zum L1-View ``prmk`` / ``rqkn`` / ``pqmn`` steckt in den
``dimensionsToStream`` der ``L2L1``-FIFOs (unverändert aus der Vorlage, gegen
die Strides verifiziert). Im Shim-DMA (``aiex.npu.dma_memcpy_nd``) sind nur noch
Offset/Size/Stride zu setzen:

==========  ==================================  ==========================  ========================================
Tensor      sizes ``[…]``                       strides ``[…]``             Bemerkung
==========  ==================================  ==========================  ========================================
``out``     ``[a=4, b=8, pm=16, qn=16]``        ``[2048, 16, 128, 1]``      ein Deskriptor pro Block (4 a-Zeilen)
``in0``     ``[b=8, c=16, pm=16, rk=64]``       ``[0, 64, 1024, 1]``        pro ``a``; ``b``-Repeat = ``stride 0``
``in1``     ``[b=8, c=16, kr=64, nc=16]``       ``[16, 8192, 128, 1]``      pro ``a`` erneut (= Repeat über ``a``)
==========  ==================================  ==========================  ========================================

Der entscheidende Punkt ist die **Wiederverwendung**: ``in0`` hängt nicht von
``b`` ab und ``in1`` nicht von ``a`` ab. Ein ``stride = 0`` wiederholt eine
Kachel, **darf aber nur die äußerste Dimension** sein (Hardware-Restriktion).

- Für ``in0`` ist der Repeat über ``b`` nötig → ``b`` steht als ``stride 0``
  ganz außen. Damit bleibt für die Tile-Indizierung kein Platz mehr für ``a``;
  deshalb wird ``in0`` **pro ``a`` einzeln** ausgegeben, wobei der Zeilen-Offset
  ``a·16·1024`` die M-Kachel auswählt.
- Für ``in1`` ist der Repeat über ``a`` nötig. Statt ``stride 0`` wird ``in1``
  einfach **in jeder ``a``-Iteration erneut** gesendet — die volle ``B``-Matrix
  fließt also 16-mal.
- ``out`` braucht keinen Repeat, also passt ``a`` als äußere reale Dimension in
  einen einzigen Deskriptor.

Buffer-Descriptor-Synchronisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Das Shim-Tile hat nur **16 Buffer-Descriptors**. Die erste Nutzung einer
``id`` braucht keine Synchronisation; vor jeder *Wiederverwendung* muss auf den
Abschluss des vorherigen Transfers gewartet werden. Wir verarbeiten die 16
M-Tile-Zeilen in **vier Blöcken zu je 4 Zeilen** und nutzen pro Block die
``id``\ s ``0`` (out), ``1,3,5,7`` (in0) und ``2,4,6,8`` (in1) — also nie mehr
als 9 BDs. Am Blockende steht ein einzelnes ``aiex.npu.dma_wait`` auf den
**Output** als Barriere: ist die Output-Kachel des Blocks geschrieben, hat der
Core ihre Eingaben bereits konsumiert, sodass die ``id``\ s im nächsten Block
gefahrlos neu belegt werden. (Wir warten bewusst auf den Output statt auf die
Eingabe-FIFOs, da Input-DMAs nicht zwingend ein Completion-Token ausgeben.)
Diese **blockierende** Block-Barriere beseitigt Task 4.

.. literalinclude:: ../../../../assignments/09_assignment/src/matmul.mlir
   :language: text

Verifikation
------------

``make run_matmul`` baut die ``xclbin``, lädt die Instruktionen und ruft
``driver.py`` auf. Die ``verify()``-Funktion vergleicht gegen die FP32-Referenz
(``atol=2``, ``rtol=0.5``).

.. note::

   Die Programmausgabe (``out/run_matmul.log``) wird nach dem Lauf auf dem
   Server ergänzt.
