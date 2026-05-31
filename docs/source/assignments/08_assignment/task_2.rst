Task 2: Instructions and Latencies
===================================

Aufgabenstellung
----------------

Die Tabelle mit den Instruktionen ausfüllen, die wir für den Tensor-Kernel
brauchen — jeweils mit VLIW-Slot und Latenz.

Lösung
------

Die Latenzen stammen aus dem ``aie2p``-Scheduling-Modell des Peano-Compilers
(``llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td``, Feld ``/*dst*/``) und sind
gegen das generierte Referenz-Assembly (``matmul_bfp16.s`` aus Assignment 07)
verifiziert. **Latenz = Anzahl Zyklen vom Issue bis das Ergebnis bereitsteht**
(d. h. der Konsument muss ``Latenz`` Zyklen später stehen).

======================================  ======  ========  =====================================
Instruction                             Slot    Latenz    Funktion
======================================  ======  ========  =====================================
``vlda.conv.fp32.bf16``                 A       7         Load BF16 + Konvertierung -> FP32-Akku
``vldb``                                B       7         Vektor-Load (BF16) in x-Register
``vshuffle``                            M       2         Transpose ``kn -> nk``
``vconv.bfp16ebs8.fp32``                M       4         Konvertierung FP32-Akku -> BFP16 (ex)
``vmul.f``                              V       6         Multiplikation (BF16 bzw. 8x8x8 bfp16)
``vmac.f``                              V       6         8x8x8 bfp16 Multiply-Accumulate
``vst.conv.bf16.fp32``                  S       ~5        Store FP32-Akku -> BF16 (MemoryCycles 5, kein dst-Reg)
``vbcst.16``                            M       2         Skalar in alle 16-bit-Lanes broadcasten (Bypass -> 1)
``vmov``                                M       1         Vektor-Register-Move
``mova`` / ``movx``                     A / X   1         Skalar-Immediate-Move (Mode-Register)
``movxm``                               XM      1         32-bit-Immediate-Move
``mov`` (Pointer)                       M       1         Pointer-Register-Move (z. B. Reset)
======================================  ======  ========  =====================================

Wichtige Eigenschaften für das Scheduling:

- Der **Akkumulator-Eingang** von ``vmac.f`` hat über einen Bypass effektiv
  Latenz 1 — aufeinanderfolgende MACs in *denselben* Akkumulator laufen
  back-to-back (Ziel: ein ``vmac`` pro Zyklus).
- ``vconv`` und ``vshuffle`` liegen beide auf dem **M-Slot** und können nicht
  im selben Zyklus ausgegeben werden — das ist der spätere Flaschenhals
  (siehe Task 6).
- ``vmac``/``vmul`` (V) und ``vconv``/``vshuffle`` (M) liegen auf verschiedenen
  Slots und können co-issued werden.
- Einige Latenzen werden durch **Bypässe** verkürzt (Felder ``MV_Bypass`` /
  ``VEC_Bypass`` in der ``.td``): ``vbcst.16`` hat nominal Latenz 2, zu einem
  direkt folgenden ``vmov`` (Move-Bypass) aber effektiv 1 — deshalb steht das
  ``vmov`` im Prolog schon einen Zyklus später. ``vst.conv`` hat keine
  dst-Register-Latenz, sondern eine Speicher-Commit-Zeit (``MemoryCycles 5``);
  beide liegen im Prolog/Epilog und damit nicht auf dem kritischen Pfad.
