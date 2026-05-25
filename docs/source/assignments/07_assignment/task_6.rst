Task 6: MAC Kernel (optional)
=============================

Aufgabenstellung
----------------

Der Kernel ``src/matmul.cpp`` implementiert eine 8 x 8 x 8
BF16-Matrixmultiplikation mit FP32-Akkumulator. Die beiden Makefile-
Targets erzeugen Assembly fuer den normalen BF16-Modus und fuer den
BFP16-Modus:

.. code-block:: bash

   make asm_matmul
   make asm_matmul_bfp16

Build-Output
------------

Normaler BF16-Modus:

.. literalinclude:: ../../../../assignments/07_assignment/out/task6/asm_matmul.log
   :language: text

BFP16-Modus:

.. literalinclude:: ../../../../assignments/07_assignment/out/task6/asm_matmul_bfp16.log
   :language: text

Instruktionsanzahl
------------------

Die Assembly-Dateien enthalten folgende Anzahl an VLIW-Zeilen im
Funktionskoerper:

.. list-table::
   :header-rows: 1

   * - Mode
     - Assembly
     - VLIW instructions / cycles
   * - Normal BF16
     - ``matmul_normal.s``
     - 43
   * - BFP16 flag
     - ``matmul_bfp16.s``
     - 30

Ohne NOPs gezählt:

.. list-table::
   :header-rows: 1

   * - Mode
     - Non-NOP operations
   * - Normal BF16
     - 62
   * - BFP16 flag
     - 22

Effekt des BFP16-Flags
----------------------

``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`` aendert das Lowering von
``aie::mmul``. Normal wird die Operation in viele BF16/FP32-Schritte
zerlegt; im BFP16-Modus tauchen stattdessen BFP16-Konvertierungen und
eine kompaktere MAC-Sequenz auf:

.. code-block:: asm

   vconv.bfp16ebs8.fp32 ex0, dm0
   vconv.bfp16ebs8.fp32 ex2, dm1
   vmac.f dm0, dm0, ex0, ex2, r0

Dadurch ist die erzeugte Assembly deutlich kuerzer.

Performance-Implikation
-----------------------

Der BFP16-Modus sollte schneller sein, weil weniger Instruktionen
ausgefuehrt werden. Der Nachteil ist die andere Numerik: BFP16 nutzt
gemeinsame Exponenten pro Block und ist deshalb nicht exakt dasselbe wie
die normale BF16/FP32-Variante.
