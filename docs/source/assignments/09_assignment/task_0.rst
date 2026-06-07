Task 0: Setup
=============

Aufgabenstellung
----------------

Den XDNA-Tensor-Kernel in das ``src/``-Verzeichnis kopieren und die
``verify()``-Funktion in den Driver übernehmen. Den maximalen absoluten Fehler
auf ``2`` und den maximalen relativen Fehler auf ``0.5`` setzen.

Lösung
------

Die Aufgabe ist erfüllt:

- Der Tensor-Kernel aus Assignment 08 (``src/matmul.s``) wurde unverändert
  übernommen; zusätzlich liegt der Zero-Init-Kernel ``src/zero.s`` bereit, der
  die Output-Kachel vor der ``c``-Akkumulation auf Null setzt.
- Die ``verify()``-Funktion ist in ``src/driver.py`` kopiert und vergleicht die
  NPU-Ausgabe gegen die FP32-Referenz mit den geforderten Toleranzen:

.. code-block:: python

   expected = in0.float() @ in1.float()
   assert torch.allclose(out.float(), expected, atol=2, rtol=0.5)

Die gegenüber Assignment 08 (``atol=0.5``, ``rtol=0.2``) gelockerten Toleranzen
tragen der größeren Kontraktionslänge Rechnung: über ``K=1024`` akkumulieren
sich mehr ``bfp16``-Rundungsfehler (gemeinsamer Block-Exponent, 7-bit Mantisse)
als über das ``K=64`` aus Assignment 08.
