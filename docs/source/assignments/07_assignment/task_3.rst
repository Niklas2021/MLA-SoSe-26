Task 3: Identify Instructions and Register Classes per Slot
=============================================================

Aufgabenstellung
----------------

Anhand der Mnemonics und Registernamen soll für jede Instruktion der
zugehörige Slot bestimmt werden. Außerdem sollen die Register-Klassen
pro Slot identifiziert werden.

Hinweise aus der Aufgabe:

- Mnemonic-Präfix/-Suffix zeigt den Slot an
- Register-Präfix zeigt die Klasse: ``p`` → Pointer, ``r`` → Skalar,
  ``x``/``y`` → Vektor, ``dm``/``cm``/``bm`` → Akkumulator

Teilaufgabe 1 – Instruktions-Tabelle
--------------------------------------

======================================  ======  ==========================================
Instruction                             Slot    Beschreibung
======================================  ======  ==========================================
``vlda.conv.fp32.bf16 cml0, [p0, #0]``  A       Load mit bf16→fp32 Konvertierung
``movx r6, #1``                          X       Immediate in Skalar-Register
``vldb x1, [p1, #0]``                   B       Vector Load über Load Unit B
``vmov bmhl2, bmhh4``                   M       Move zwischen Akkumulator-Registern
``mova r0, #60``                         A       Immediate in Skalar-Register via Load A
``vadd.f dm0, dm0, dm1, r0``            V       BF16 Vektor-Addition
``ret lr``                               X       Return (Control Flow)
``mov p1, p4``                           M       Move zwischen Pointer-Registern
``vst.conv.bf16.fp32 cml0, [p2, #0]``   S       Store mit fp32→bf16 Konvertierung
======================================  ======  ==========================================

**Wie erkennt man den Slot?**

- **A**: Präfix ``vlda`` oder ``mova``
- **B**: Präfix ``vldb``
- **S**: Präfix ``vst``
- **X**: Suffix ``x`` (``movx``) oder Control-Flow (``ret``)
- **M**: ``vmov`` (Vektor-/Akkumulator-Move) oder ``mov`` ohne Suffix
- **V**: Präfix ``v`` + arithmetische Op (``vadd``)

``mova`` sieht auf den ersten Blick komisch aus – die Load Unit A kann
aber auch Immediates in Skalar-Register schreiben, nicht nur Speicher
laden.

Teilaufgabe 2 – Register-Klassen pro Slot
-------------------------------------------

======  ================================  ==========================
Slot    Register-Klassen (dst / src)      Beispiel-Register
======  ================================  ==========================
V       Akkumulator (dst+src), Skalar     ``dm0``, ``dm1``, ``r0``
A       Akkumulator (dst), Pointer,       ``cml0``, ``p0``, ``r0``
        Skalar (dst)
B       Vektor (dst), Pointer (src)       ``x1``, ``p1``
S       Akkumulator (src), Pointer (src)  ``cml0``, ``p2``
X       Skalar (dst), Link (src)          ``r6``, ``lr``
M       Akkumulator (dst+src),            ``bmhl2``, ``bmhh4``,
        Pointer (dst+src)                 ``p1``, ``p4``
======  ================================  ==========================

Die Slots sind klar spezialisiert:

- **V** rechnet (``vadd.f``) und arbeitet auf Akkumulator-Registern
  (``dm``). Der vierte Operand ``r0`` ist ein Skalar-Register –
  vermutlich ein Konfigurationswert für die Operation.
- **A** und **B** laden Daten aus dem Speicher. A lädt in
  Akkumulator-Register (``cm``), B in Vektor-Register (``x``/``y``).
  Beide nutzen Pointer-Register für die Adressierung.
- **S** schreibt zurück in den Speicher, liest also aus Akkumulator
  und Pointer.
- **X** ist für Skalar-Operationen und Control Flow (``ret``).
- **M** bewegt Daten zwischen Registern, sowohl Akkumulator als auch
  Pointer.
