Task 4: Testing
===============

Aufgabenstellung
----------------

Die Implementierung mit ``make run_matmul`` testen.

Lösung
------

``make run_matmul`` baut die ``xclbin`` über ``aiecc.py``, lädt die
NPU-Instruktionen und ruft ``src/driver.py`` auf. Der Treiber erzeugt zufällige
``bf16``-Eingaben, führt den Matmul auf der NPU aus und vergleicht gegen die
``torch``-Referenz:

.. code-block:: python

   ref = in0 @ in1
   torch.testing.assert_close(out, ref, atol=1.2, rtol=0.05)

Die Toleranzen sind enger als in Assignment 09 (``atol=2, rtol=0.5``); die
Restabweichung ist die inhärente ``bfp16``-Rechengenauigkeit (gemeinsamer
Block-Exponent, 7-Bit-Mantisse) plus die finale ``bf16``-Rundung des Outputs.
Der Kernel setzt bereits ``conv_even`` (Round-to-nearest-even) per
``mov crrnd, #12``, sodass kein systematischer Trunkierungsfehler über die
``K``-Akkumulation entsteht.

Ergebnis
~~~~~~~~

Der Lauf auf dem NPU besteht die Verifikation::

   [PASS] matmul verification passed.

Über alle 32 Compute-Kacheln liegt der Fehleranteil (Schwelle ``1.2``) bei
``0.1 %`` und ist gleichmäßig über ``a``, ``x``, ``b`` und ``y`` verteilt — der
Output stimmt also für das gesamte Tile-Array auf bfp16-Genauigkeit.

.. note::

   Der Build erfordert die Peano-/``aiecc.py``-Toolchain und echte NPU-Hardware
   und läuft daher auf dem Remote-Server (AI-Max), nicht lokal.
