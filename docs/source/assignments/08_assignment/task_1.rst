Task 1: Verify Function
========================

Aufgabenstellung
----------------

Die ``verify()``-Funktion für die Matrixmultiplikation in ``src/driver.py``
implementieren. Vor der Tensor-Initialisierung ``torch.manual_seed(42)``
einfügen. Beim Vergleich der Tensoren den maximalen absoluten Fehler auf
``0.5`` und den maximalen relativen Fehler auf ``0.2`` setzen.

Lösung
------

Da das Output-Scratchpad bei der NPU-Initialisierung mit Null vorbelegt ist,
gilt ``out == in0 @ in1``. Die Referenz wird auf der CPU in FP32 berechnet:

.. code-block:: python

   expected = in0.float() @ in1.float()
   assert torch.allclose(out.float(), expected, atol=0.5, rtol=0.2)

Die großzügigen Toleranzen (``atol=0.5``, ``rtol=0.2``) sind nötig, weil die
NPU intern mit ``bfp16`` (gemeinsamer Exponent pro 8er-Block, 7-bit Mantisse)
rechnet, die Referenz dagegen mit FP32 — die Numerik weicht daher leicht ab.

``torch.manual_seed(42)`` steht in ``run()`` direkt vor der Initialisierung der
Eingabe-Tensoren, sodass der Lauf reproduzierbar ist.

Driver und Verifikation
------------------------

.. literalinclude:: ../../../../assignments/08_assignment/src/driver.py
   :language: python

Die ``verify()``-Funktion gibt zusätzlich den maximalen und mittleren
absoluten Fehler aus; ``benchmark()`` misst den End-to-End-Durchsatz (siehe
Task 6).
