Task 2: Identify VLIW Slots
=============================

Aufgabenstellung
----------------

Das XDNA2 VLIW hat 6 Unit slots.
Jeder Slot ist entweder mit einer Operation/ mit einem
NOP gefüllt/ leer gelassen.

Anhand der ersten zwei VLIW-Instruktionen aus ``build/vadd.s`` sollen
die NOP-Mnemonics und die Belegung identifiziert werden:

.. code-block:: asm

   vlda.conv.fp32.bf16 cml0, [p0, #0]; nopb; nops; nopxm; nopv
   vlda.conv.fp32.bf16 cmh0, [p0, #64]; nopx

Lösung
------

**Tabelle:**

==================  ======  ==============  ======================================
Functional Unit     Slot    NOP Mnemonic    Occupied in the second instruction?
==================  ======  ==============  ======================================
Vector Unit         V       ``nopv``        Nein (leer)
Load Unit A         A       ``nopa``        Ja (``vlda.conv.fp32.bf16``)
Load Unit B         B       ``nopb``        Nein (leer)
Store Unit          S       ``nops``        Nein (leer)
Scalar/Control      X (XM)  ``nopx``        Nein (``nopx``)
Movement Unit       M (XM)  ``nopxm``       Nein (leer)
==================  ======  ==============  ======================================

