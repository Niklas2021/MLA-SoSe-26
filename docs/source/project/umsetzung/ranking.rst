.. _ranking:

Ranking: die Kostenmodelle
^^^^^^^^^^^^^^^^^^^^^^^^^^

Zweck des Rankings
"""""""""""""""""""

Nach dem Prunen bleiben 342 Kandidaten. Das ist wenig genug, dass man auf der GB10
alle messen könnte — ein Compile kostet etwa 0.4 s. Das Ranking ist deshalb nicht
in erster Linie dazu da, Zeit zu sparen. Zunächst ging es darum zu prüfen, ob
unser Kostenmodell die tatsächlich beste Config nach oben zieht. Erst in einem
zweiten Schritt wird das Ranking zum Vorfilter für den praktischen Tuner, dort
werden dann nur die Top-k gemessen statt aller überlebenden Kandidaten.

``rank`` schätzt dazu pro Kandidat eine Laufzeit und sortiert aufsteigend. Es gibt
drei Modelle mit wachsender Komplexität: ``bw`` (reine Bandbreite), ``bw_occ``
(Bandbreite mit Occupancy-Strafe) und ``roofline`` (Memory- gegen Compute-Zeit).
Alle drei laufen über dieselbe Funktion und denselben Satz an Zwischenwerten:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — rank(), die Metrik pro Kandidat
   :start-at: def rank(candidates, dev
   :end-at: "bound": "compute" if compute_ms >= memory_ms_l2 else "memory"}))

Jeder Kandidat trägt also ``est_ms`` (Bandbreiten-Schätzung), ``est_ms_occ``
(Bandbreite geteilt durch Occupancy), ``compute_ms`` und ``roof_ms`` gleichzeitig
mit sich — welches Feld am Ende sortiert, entscheidet nur der ``model``-Parameter.
So bleiben die drei Modelle direkt vergleichbar, ohne dass man dieselben Kandidaten
mehrmals durchrechnen muss.

Modell 1: Bandbreite (bw)
""""""""""""""""""""""""""

Die Grundidee: die FLOPs sind für alle Kandidaten identisch (dieselbe Kontraktion,
nur unterschiedlich gekachelt), also entscheidet allein der DRAM-Traffic. Eine
größere L2-Gruppe bedeutet weniger Nachladen von A und B, also weniger Traffic und
eine niedrigere geschätzte Zeit.

``estimate_dram_bytes`` rechnet das im Worst Case durch — kein L2-Treffer über
Gruppengrenzen hinweg. A wird einmal pro Gruppen-Spalte geladen, B einmal pro
Gruppen-Zeile, C einmal geschrieben:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — estimate_dram_bytes()
   :pyobject: estimate_dram_bytes

Ohne ``dev`` liefert die Funktion konsequent den Worst Case — das ist ``est_ms``,
die Grundlage von Modell ``bw``. Mit ``dev`` schaltet sie um: passt der Working-Set
einer Swizzle-Gruppe (``(m_l2·m_prim + n_l2·n_prim) · padded_k · dtype_bytes``) ins
L2 der jeweiligen Karte, zählt nur noch das einmalige Kaltladen aller drei Tensoren.
Dieser zweite Zweig ist der portable Umschalter zwischen den Karten: auf einer GPU
mit großem L2 (wie der GB10) fällt der Traffic praktisch auf den Kaltstart-Wert,
auf einer GPU mit kleinem L2 bleibt der Worst Case bestimmend, und die
Gruppengröße zählt wieder.

Kandidaten mit gleichem Traffic werden nach Grid-Größe absteigend geordnet, weil
mehr Blöcke die SMs tendenziell besser füllen:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — rank(), bw/bw_occ-Sortierung
   :start-at: key = "est_ms_occ" if model
   :end-at: ranked.sort(key=lambda x: (x[1][key], -x[1]["grid"]))
   :dedent:

Modell 2: bw mit Occupancy (bw_occ)
""""""""""""""""""""""""""""""""""""

Reine Bandbreite übersieht einen Fall: Variante B mit einer großen L2-Gruppe kann
auf ein sehr kleines Grid kommen. 

``bw_occ`` korrigiert das, indem es die geschätzte Zeit durch die Occupancy teilt
(``est_ms_occ = memory_ms / occ``). Die Occupancy-Schätzung selbst
kommt in zwei Stufen: wie viele Blöcke pro SM überhaupt Platz finden
(begrenzt durch Register- und SMEM-Budget) und wie gut das resultierende Grid
die verfügbaren Wellen füllt:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — estimate_blocks_per_sm()
   :pyobject: estimate_blocks_per_sm

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — occupancy_util()
   :pyobject: occupancy_util

``estimate_blocks_per_sm`` schaut, wie viele Blöcke einer Config gleichzeitig auf einen SM passen. Dafür nimmt sie dieselben zwei Werte wie beim Pruning 
– SMEM-Bedarf und Register-Bedarf des Akkumulators – nur diesmal nicht als Ja/Nein-Schwelle, sondern um eine Zahl auszurechnen: die kleinere der beiden
möglichen Block-Zahlen gewinnt, weil die knappere Ressource zuerst ausgeht.

Modell 3: Roofline
""""""""""""""""""

Das dritte Modell wägt Memory- und Compute-Zeit gegeneinander ab, statt nur die
Speicherseite zu betrachten:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — rank(), Compute-Zeit und roofline-Sortierung
   :start-at: padded_flops = 2 * cand.padded_m
   :end-at: ranked.sort(key=lambda x: (x[1]["roof_ms"], x[1]["est_ms"]))
   :dedent:

``roof_ms = max(memory_ms_l2, compute_ms)`` ist der klassische Roofline-Ansatz:
je nachdem, welche Seite größer ist, entscheidet Speicherbandbreite oder
Rechenleistung über die Laufzeit. 

Der einzige Wert in dieser Rechnung, der sich nicht direkt aus Config-Größen
ableiten lässt, ist ``peak_tensor_flops`` — eine Architektur-Schätzung der
Tensor-Core-Spitzenleistung. Das beeinflusst nur, an welcher Traffic-Schwelle ein
Kandidat vom Memory- ins Compute-Regime kippt, nicht die Reihenfolge der
Kandidaten innerhalb eines Regimes — dort zählt weiterhin ``memory_ms_l2`` bzw.
``compute_ms`` unverändert.

