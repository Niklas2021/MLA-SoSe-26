Task Optional: CUTILEIR und ``assume_div_by``
=============================================

Aufgabenstellung
----------------

Das Copy-Kernel aus Task 4 wird mit der Umgebungsvariablen
``CUDA_TILE_LOGS=CUTILEIR`` gestartet. Im generierten IR sind
``assume_div_by``-Hints zu suchen und deren Zweck zu erklären.

Implementiertes Skript
-----------------------

.. literalinclude:: ../../../../assignments/02_assignment/src/task_optional.py
   :language: python

Programmausgabe (vollständiger IR)
-----------------------------------

.. literalinclude:: ../../../../assignments/02_assignment/out/task_optional/task_optional.log
   :language: text

Warum setzt der Compiler diese Hints?
--------------------------------------

Der Compiler erzeugt intern einen ``make_tensor_view``-Aufruf, der die
logische 2D-Sicht auf das Array im GPU-Speicher aufbaut:

.. code-block:: text

   make_tensor_view(base_ptr=$2, shape=($4, $6), dynamic_strides=($8))

Die vier Argumente entsprechen direkt den vier ``assume_div_by``-Hints
(jeweils für ``src`` und ``dst``):

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Argument
     - Variable im IR
     - Divisor
     - Was wird garantiert
   * - ``base_ptr``
     - ``src_0`` / ``dst_0``
     - 16
     - Startadresse ist 16-Byte-aligned (für 128-Bit-Vektorloads)
   * - ``shape[0]``
     - ``src_1`` / ``dst_1``
     - 16
     - Zeilenzahl ist Vielfaches von 16 → kein Restblock, kein Masking nötig
   * - ``shape[1]``
     - ``src_2`` / ``dst_2``
     - 16
     - Spaltenzahl ist Vielfaches von 16 → kein Restblock, kein Masking nötig
   * - ``dynamic_strides``
     - ``src_3`` / ``dst_3``
     - 8
     - Stride in Elementen divisibel durch 8 (= 16 Byte) → jede Zeile beginnt 16-Byte-aligned

Tile-Dimensionen in cuTile sind immer Potenzen von 2.
Da ``bh = 64`` und ``bw = 64`` als ``ct.Constant[int]`` zur Compile-Zeit
feststehen, kann der Compiler beim Aufbau der Tensor-View für alle
vier Argumente Teilbarkeit durch die jeweiligen Divisoren beweisen:

- CuPy alloziert GPU-Speicher standardmäßig mit mindestens 256-Byte-Alignment,
  womit ``base_ptr`` automatisch durch 16 teilbar ist.
- ``M`` und ``N`` sind Vielfache der Tile-Größe (64 = 4 × 16),
  womit ``shape[0]`` und ``shape[1]`` durch 16 teilbar sind.
- Der Zeilenstride eines contiguous CuPy-Arrays ist ``N`` Elemente,
  also ebenfalls durch 8 teilbar (N ≥ 16, Potenz von 2).

Mit diesen Garantien kann das Backend:

1. **128-Bit-Vektorinstruktionen** (``LDG.128`` / ``STG.128``) emittieren –
   diese setzen 16-Byte-Ausrichtung der Startadresse voraus und laden
   8 fp16-Elemente in einem einzigen Speicherzugriff.
2. **Boundary-Masking weglassen** – da shape[0] und shape[1] durch 16
   teilbar sind, entstehen keine Restblöcke am Rand, für die
   predizierte Lade-/Speicherbefehle nötig wären.
3. **Adressarithmetik vereinfachen** – bekannte Teilbarkeit erlaubt dem
   Compiler, Multiplikationen und Divisionen durch Potenzen von 2
   als Bit-Shifts zu codieren.
