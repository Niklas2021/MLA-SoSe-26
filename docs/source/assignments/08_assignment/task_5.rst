Task 5: Implementation
=======================

Aufgabenstellung
----------------

Den Tensor-Kernel in ``src/matmul.s`` implementieren (außer dem finalen
``ret lr`` keine Control-Flow-Instruktion) und mit ``make run_matmul``
verifizieren.

Lösung
------

Der Kernel rechnet beide ``p``-Bahnen nacheinander; pro Bahn loopt er über
``r=0..7`` (per ``.rept`` entrollt — eine Assembler-Direktive, also kein
Control-Flow) und akkumuliert in ``dm0`` (q=0) und ``dm1`` (q=1). Pro ``r``
laufen die beiden ``q``-Ketten parallel auf getrennten Registern, und
``vshuffle`` (M) sowie ``vmul`` (V) werden im selben VLIW-Wort co-issued.

Die finale Version ist zusätzlich **software-pipelined** (``II = 16``): die
Loads der nächsten ``r``-Iteration werden in die Latenz-Wartezyklen der aktuellen
vorgezogen, sodass der Body keinen Front-Load-Stall mehr hat (16 statt 25
Zyklen/r). Damit Roh-Lade- und BFP16-Register sich nicht überlappen, liegen die
BFP16-Operanden auf den Shuffle-Outputs (``ex6=x6``, ``ex8=x8``); ``vconv ex0``
(in0) steht hinter den vier Shuffles, sodass die q0/q1-Konvertierungsketten nur
um 2 (statt 3) versetzt sind und die beiden ``vmac`` dicht beieinander liegen;
die letzte ``r``-Iteration ist „gepeelt" (kein Prefetch), damit ``p0``/``p2``
sauber durch beide Pässe laufen. Details siehe Task 6.

.. literalinclude:: ../../../../assignments/08_assignment/src/matmul.s
   :language: asm

Verifikation
------------

``make run_matmul`` baut die ``xclbin``, lädt die Instruktionen und ruft
``driver.py`` auf. Die ``verify()``-Funktion vergleicht gegen die FP32-Referenz
(``atol=0.5``, ``rtol=0.2``):

.. literalinclude:: ../../../../assignments/08_assignment/out/run_matmul.log
   :language: text

Der maximale absolute Fehler (0.357) liegt klar unter der Toleranz; die
Abweichung kommt aus der ``bfp16``-Numerik (gemeinsamer Block-Exponent).

Entwicklungsweg
---------------

Der Kernel entstand iterativ über die On-Device-Verifikation:

1. **Volle Version mit Skalar-Pointer-Arithmetik** -> Output komplett null.
   Ursache: der Round-Trip Pointer -> Skalar -> Pointer (``mov r0,p0``; ``add``;
   ``mov p0,r0``) korrumpiert die 20-bit-Pointer.
2. **Isolationstest** (nur ``out[0][0]`` mit direkten Pointern) -> Quadrant
   (0,0) korrekt (Fehler 0.285) => Rechnung in Ordnung, Bug war die
   Pointer-Arithmetik.
3. **Reine Pointer-Walks + ``mov p,p``-Resets** -> Quadranten q=0 korrekt,
   q=1 falsch. Ursache: im q=1-Block fehlte der ``vconv``-Füller, sodass das
   ``vmul`` einen Zyklus zu früh ``x7`` (Shuffle-Latenz 2) las.
4. **Eng geschedulte Version** (q0/q1 parallel, korrekte Latenzen) -> alle vier
   Quadranten korrekt, ``[PASS]``. 25 Zyklen/r, ≈ 438 Zyklen gesamt.
5. **Pipelined Version** (Loads der nächsten Iteration vorgezogen, ``II = 16``)
   -> 16 Zyklen/r, ≈ 300 Zyklen gesamt, weiterhin ``[PASS]`` (siehe Task 6 für
   die Vorher/Nachher-Zahlen).
