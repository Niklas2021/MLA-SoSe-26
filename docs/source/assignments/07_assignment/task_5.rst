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

Implementierung
---------------

.. literalinclude:: ../../../../assignments/07_assignment/src/custom_vadd.s
   :language: asm

Die Verifikation nutzt:

.. code-block:: python

   expected = (in0.float() + in1.float() + in1.float()).bfloat16()

Schedule und Zykluszahl
-----------------------

Der Schedule beachtet die beobachteten Abstaende aus ``vadd.s``: Load ->
``vadd.f`` nach drei Zwischenzyklen und ``vadd.f`` -> Store mit Latenz 6,
also nach fuenf vollständigen Zwischenzyklen. Für die direkte
``vadd.f``-Akkumulatorkette kann der Feedback-Pfad bereits nach zwei
Zwischenzyklen genutzt werden (Abstand 3); das ändert die in Task 4
bestimmte Latenz bis zum Store nicht.

.. list-table::
   :header-rows: 1

   * - Cycle
     - Operation
     - Kommentar
   * - 1
     - ``vlda.conv.fp32.bf16 cml0, [p0, #0]; movx r0,#60``
     - Load A low + Sign-Mask
   * - 2
     - ``vlda.conv.fp32.bf16 cmh0, [p0, #64]``
     - Load A high
   * - 3
     - ``vlda.conv.fp32.bf16 cml1, [p1, #0]``
     - Load B low
   * - 4
     - ``vlda.conv.fp32.bf16 cmh1, [p1, #64]``
     - Load B high
   * - 5-7
     - ``nop``
     - Load-Latenz fuer letzten Load
   * - 8
     - ``vadd.f dm0, dm0, dm1, r0``
     - ``dm0 = A + B``
   * - 9-10
     - ``nop``
     - Abstand fuer ``vadd.f``-Kette
   * - 11
     - ``vadd.f dm0, dm0, dm1, r0``
     - ``dm0 = A + B + B``
   * - 12-13
     - ``nop``
     - Store-Latenz
   * - 14
     - ``ret lr``
     - Return vor nutzbaren Delay-Slots
   * - 15-16
     - ``nop``
     - Delay-Slots 5 und 4
   * - 17
     - ``vst.conv.bf16.fp32 cml0, [p2, #0]``
     - Store low in Delay-Slot 3
   * - 18
     - ``vst.conv.bf16.fp32 cmh0, [p2, #64]``
     - Store high in Delay-Slot 2
   * - 19
     - ``nop``
     - Delay-Slot 1

Der Kernel braucht damit **19 VLIW-Zyklen**. Das ist unter diesen
Latenzannahmen minimal, da die vier Loads seriell laufen und der Store
erst nach der ``vadd.f``-Latenz erlaubt ist. Die ``ret``-Delay-Slots
werden fuer die Stores genutzt.

Build- und Laufstatus
---------------------

``make obj_custom_vadd`` assembliert den Kernel, ``make run_custom_vadd``
verifiziert ihn erfolgreich.

Object-Build:

.. literalinclude:: ../../../../assignments/07_assignment/out/task5/obj_custom_vadd.log
   :language: text

Kernel-Build und Verifikation:

.. literalinclude:: ../../../../assignments/07_assignment/out/task5/run_custom_vadd.log
   :language: text
