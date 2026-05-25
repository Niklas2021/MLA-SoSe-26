Task 3: Identify Instructions and Register Classes per Slot
=============================================================

Aufgabenstellung
----------------

Anhand der Mnemonics und Registernamen soll für jede Instruktion der
zugehörige Slot bestimmt werden. Außerdem sollen die Register-Klassen
pro Slot identifiziert werden.

Hinweise aus der Aufgabe:

- Mnemonic prefix/suffix indicates the slot.
- Register name prefix indicates class: p → pointer register; r → scalar register; x/y → vector register; dm/cm/bm → accumulator register.

Teilaufgabe 1 – instruction table
--------------------------------------

======================================  ======  ==========================================
Instruction                             Slot    Short description (optional)
======================================  ======  ==========================================
``vlda.conv.fp32.bf16 cml0, [p0, #0]``  A       Load Unit A mit bf16→fp32 Konvertierung
``movx r6, #1``                         X       Scalar immediate move
``vldb x1, [p1, #0]``                   B       Vector Load über Load Unit B
``vmov bmhl2, bmhh4``                   M       Move zwischen Akkumulator-Registern
``mova r0, #60``                        A       Scalar immediate move im A-Slot
``vadd.f dm0, dm0, dm1, r0``            V       Vektor-/Akkumulator-Addition
``ret lr``                              X       Return (Control Flow)
``mov p1, p4``                          M       Move zwischen Pointer-Registern
``vst.conv.bf16.fp32 cml0, [p2, #0]``   S       Store mit fp32→bf16 Konvertierung
======================================  ======  ==========================================



Teilaufgabe 2 – register class table
-------------------------------------------

======  ==================================================  ==============================
Slot    Register classes (dst / src)                        Example registers
======  ==================================================  ==============================
V       Accumulator / Vector / Scalar modifier              ``dm0``, ``dm1``, ``x6``, ``r0``
A       Accumulator or scalar dst, pointer/immediate src     ``cml0``, ``cmh0``, ``p0``, ``r0``
B       Vector dst, pointer src                              ``x1``, ``x4``, ``p1``
S       Accumulator/vector src, pointer address              ``cml0``, ``cmh0``, ``p2``
X       Scalar / Control                                     ``r6``, ``r29``, ``lr``
M       Pointer / Vector / Accumulator moves                 ``p1``, ``p4``, ``bmhl2``
XM      Shared X/M encoding                                  ``r0``, ``p1``, ``x1``, ``bmhh4``
======  ==================================================  ==============================
