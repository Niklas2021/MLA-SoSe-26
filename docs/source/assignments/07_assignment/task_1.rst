Task 1: Vector-Add Kernel
==========================

Aufgabenstellung
----------------

``src/vadd.cpp`` enthält eine Funktion ``vadd``, die zwei 64-Element
BF16-Vektoren elementweise addiert und das Ergebnis in einen
BF16-Ausgabevektor schreibt.

1. Den ``TODO``-Block in ``vadd.cpp`` ausfüllen (elementweise Addition)
2. Zu Assembly kompilieren: ``make asm_vadd`` → ``build/vadd.s``
3. ``verify()`` in ``driver.py`` implementieren und Kernel ausführen:
   ``make run_vadd``

Kernel-Implementierung
-----------------------

.. literalinclude:: ../../../../assignments/07_assignment/src/vadd.cpp
   :language: cpp

Der Kernel nutzt die AIE-API:

- ``aie::load_v<64>()`` lädt 64 BF16-Elemente in ein Vektor-Register
- ``aie::add(v_in0, v_in1)`` addiert elementweise
- ``aie::store_v()`` schreibt das Ergebnis zurück

Driver und Verifikation
------------------------

.. literalinclude:: ../../../../assignments/07_assignment/src/driver.py
   :language: python

Die ``verify()``-Funktion berechnet die Referenz auf der CPU:

.. code-block:: python

   expected = (in0.float() + in1.float()).bfloat16()

Vergleich mit ``torch.allclose(..., atol=1e-2, rtol=1e-2)``.
Für ``custom_vadd`` (Task 5) ist die Referenz ``A + B + B``.

Output:

.. code-block:: python
    [PASS] vadd verification passed.

**Frage:** Welcher Mnemonic wird für die BF16 elementweise Addition verwendet?

Antwort: BF16 Additions-Mnemonic
----------------------------------

Der Mnemonic für die BF16 elementweise Addition ist ``vadd.f``
(zu sehen in ``build/vadd.s``). Das ``v``-Präfix zeigt den Vector-Slot
an, ``.f`` steht für Floating-Point. Die vollständige Instruktion sieht
so aus:

.. code-block:: asm

   vadd.f dm0, dm0, dm1, r0

Die Addition arbeitet auf Akkumulator-Registern (``dm``), nicht direkt
auf den Vektor-Registern (``x``/``y``). Die Eingabedaten werden vorher
per ``vlda.conv.fp32.bf16`` in FP32-Akkumulator-Register konvertiert,
die Addition läuft in FP32, und beim Store (``vst.conv.bf16.fp32``)
wird zurück nach BF16 konvertiert.
