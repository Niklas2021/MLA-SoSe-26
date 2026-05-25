Task 5: Hand-Scheduled BF16 Vector-Add
========================================

Aufgabenstellung
----------------

Ein Assembly-Kernel soll ``C = A + B + B`` berechnen. Unter Nutzung der
ISA-Erkenntnisse aus Tasks 1–4 wird die vadd-Operation manuell in
VLIW-Instruktionen geschedult.

1. ``TODO``-Kommentare in ``src/custom_vadd.s`` durch echte Instruktionen
   ersetzen. Constraints:

   - Nur so viele NOP-Zyklen einfügen, wie die Latenzen zwingend erfordern.
   - Die fünf ``ret`` Delay-Slots dürfen Nutzinstruktionen enthalten
     (z. B. den Store).

2. Kernel assemblieren: ``make obj_custom_vadd``.
3. ``verify()`` für ``custom_vadd`` in ``src/driver.py`` implementieren
   und ausführen: ``make run_custom_vadd``.

Frage: Wie viele VLIW-Zyklen braucht der hand-scheduled Kernel? Ist
das die minimale Anzahl?

Lösung
------

*Folgt nach Implementierung des Kernels.*
