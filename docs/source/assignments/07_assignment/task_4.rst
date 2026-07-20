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

.. list-table::
   :header-rows: 1

   * - Instruction
     - Output register
     - First dependent instruction
     - Cycles apart
     - Latency
   * - ``mova``
     - ``r0``
     - ``vadd.f dm0, dm0, dm1, r0``
     - 1
     - 1
   * - ``vadd.f``
     - ``dm0``
     - ``vst.conv.bf16.fp32 cml0, [p2, #0]``
     - 6
     - 6

Bei ``mova`` folgt der Konsument direkt im naechsten Zyklus. Bei
``vadd.f`` liegen zwischen Produzent und erstem Store fuenf
vollständige Zwischenzyklen. Der abhängige Store startet somit sechs Zyklen
nach dem Produzenten; nach der Aufgabenregel ist die Latenz daher **6**.
