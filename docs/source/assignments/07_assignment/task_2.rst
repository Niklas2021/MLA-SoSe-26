Task 2: Identify VLIW Slots
=============================

Aufgabenstellung
----------------

Das XDNA2 VLIW-Instruktionswort hat sechs Functional-Unit-Slots.
Jeder Slot ist entweder mit einer Operation belegt, mit einem
slot-spezifischen NOP gefüllt, oder leer gelassen.

Anhand der ersten zwei VLIW-Instruktionen aus ``build/vadd.s`` sollen
die NOP-Mnemonics und die Belegung identifiziert werden:

.. code-block:: asm

   vlda.conv.fp32.bf16 cml0, [p0, #0]; nopb; nops; nopxm; nopv
   vlda.conv.fp32.bf16 cmh0, [p0, #64]; nopx

Lösung
------

**Instruktion 1 aufgeschlüsselt:**

Die erste Instruktion hat alle Slots explizit belegt – entweder mit
einer echten Operation oder mit dem jeweiligen NOP. Die Teile sind
durch ``;`` getrennt:

.. code-block:: text

   vlda.conv.fp32.bf16 cml0, [p0, #0]   → Slot A  (Load A, belegt)
   nopb                                  → Slot B  (Load B, NOP)
   nops                                  → Slot S  (Store, NOP)
   nopxm                                 → Slot XM (Scalar + Move, NOP)
   nopv                                  → Slot V  (Vector, NOP)

Daraus lassen sich die NOP-Mnemonics direkt ablesen. ``nopxm`` ist das
kombinierte NOP für die Slots X und M, wenn beide leer sind. In
Instruktion 2 steht ``nopx`` allein – das ist der NOP nur für den
Scalar-Slot.

**Instruktion 2 aufgeschlüsselt:**

Nur zwei Teile stehen explizit da. Die restlichen Slots (B, S, M, V)
werden einfach leer gelassen – weder eine Operation noch ein NOP ist
angegeben:

.. code-block:: text

   vlda.conv.fp32.bf16 cmh0, [p0, #64]  → Slot A  (Load A, belegt)
   nopx                                  → Slot X  (Scalar, NOP)
   (B, S, M, V: leer gelassen)

**Ausgefüllte Tabelle:**

==================  ======  ==============  ======================================
Functional Unit     Slot    NOP Mnemonic    Belegt in der zweiten Instruktion?
==================  ======  ==============  ======================================
Vector Unit         V       ``nopv``        Nein (leer)
Load Unit A         A       ``nopa``        Ja (``vlda.conv.fp32.bf16``)
Load Unit B         B       ``nopb``        Nein (leer)
Store Unit          S       ``nops``        Nein (leer)
Scalar/Control      X (XM)  ``nopx``        Nein (``nopx``)
Movement Unit       M (XM)  ``nopxm``       Nein (leer)
==================  ======  ==============  ======================================

``nopa`` war in der Aufgabe bereits vorgegeben. ``nopxm`` deckt X und M
zusammen ab, wenn beide leer sind. Sobald nur einer der beiden Slots
ein NOP braucht (wie ``nopx`` in Instruktion 2), wird der einzelne
Mnemonic geschrieben.
