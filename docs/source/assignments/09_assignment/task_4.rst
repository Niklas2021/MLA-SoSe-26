Task 4: Performance
====================

Aufgabenstellung
----------------

Die Datenbewegung im MLIR-Code so ändern, dass es **keinen blockierenden Wait**
mehr gibt — d. h. es ist immer eine Datenbewegungs-Operation ausstehend, die
ausgegeben werden kann (außer der letzten).

Lösung
------

In Task 3 stand am Ende jedes der vier Blöcke ein ``aiex.npu.dma_wait`` als
Barriere: der nächste Block durfte erst ausgegeben werden, wenn der Output des
aktuellen fertig war — die Shim-DMAs liefen also **nicht** durchgehend.

Wir lösen das mit **Double Buffering / Ping-Pong über zwei disjunkte
Buffer-Descriptor-Sets**. Statt vier Blöcken zu je 4 M-Tile-Zeilen verarbeiten
wir **acht Gruppen zu je 2 Zeilen** (``a = 2g`` und ``2g+1``) und vergeben die
``id``\ s abwechselnd:

- **gerade Gruppen — Set A:** ``0`` (out), ``1,3`` (in0), ``2,4`` (in1)
- **ungerade Gruppen — Set B:** ``8`` (out), ``9,11`` (in0), ``10,12`` (in1)

Da zwei aufeinanderfolgende Gruppen **verschiedene** BD-Sets benutzen, kann
Gruppe ``g+1`` ausgegeben werden, **während** Gruppe ``g`` noch in Flight ist —
es kommt zu keinem BD-Konflikt. Maximal sind 10 der 16 BDs gleichzeitig belegt.

Der entscheidende Kniff ist das **verzögerte (deferred) ``dma_wait``**: Wir
geben erst Gruppe ``g+1`` aus und warten *danach* auf Gruppe ``g`` — und zwar
genau dann, wenn deren BD-Set für Gruppe ``g+2`` wiederverwendet wird. Während
dieses Wait läuft Gruppe ``g+1`` bereits, der Shim ist also nie idle:

.. code-block:: text

   issue G0 (Set A)            # erste Nutzung, kein Wait noetig
   issue G1 (Set B)            # in Flight, ueberlappt mit dem Wait unten
   for g in 2..7:
       dma_wait(out)           # auf G(g-2): dessen Set wird gleich wiederverwendet
       issue G(g) (Set g%2)    # waehrend dieses Issues war G(g-1) schon in Flight
   dma_wait(out)               # G6  (vorletzte)
   dma_wait(out)               # G7  (letzte, einzige echt blockierende)

Wir warten bewusst auf den **Output** (nicht die Eingabe-FIFOs): ist der Output
einer Gruppe geschrieben, hat der Core deren Eingaben bereits konsumiert, sodass
das zugehörige BD-Set sicher frei ist. So ist — bis auf die letzte Gruppe —
immer eine Datenbewegung ausstehend.

Die berechneten Werte sind identisch zu Task 3 (gleiche Tiles, nur andere
Ausgabe-/Wartereihenfolge); die Verifikation bleibt ``[PASS]`` (max abs error
2.04).

.. literalinclude:: ../../../../assignments/09_assignment/src/matmul.mlir
   :language: text
   :lines: 41-
