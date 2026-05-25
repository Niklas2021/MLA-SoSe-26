Task 4: Infer Operation Latencies
==================================

Aufgabenstellung
----------------

Der XDNA2 Compute-Tile hat **keine Stall-Logik**: Wenn eine Konsumenten-
Instruktion ein Register liest, bevor der Produzent geschrieben hat, ist
das Ergebnis undefiniert. Der Compiler (und der Programmierer bei
hand-scheduliertem Assembly) muss NOP-Instruktionen einfügen, um die
Latenz jeder Operation einzuhalten.

**Zählregel:** Latenz ist die Anzahl Zyklen vom Produzenten (Zyklus 1)
bis zur ersten abhängigen Instruktion (exklusiv). Ein NOP zwischen
Produzent und Konsument zählt als ein Zyklus.

Aus ``build/vadd.s`` soll die Latenztabelle für ``mova`` und ``vadd.f``
ausgefüllt werden.

Lösung
------

*Folgt nach Compile von* ``build/vadd.s`` *und Auszählen der Zyklen.*
