Task 3: Register Blocking
==========================

Aufgabenstellung
----------------

Ein Register-Blocking für den Tensor-Kernel wählen, Eingabe- und
Ausgabe-Tensoren den Registern zuordnen und die Entscheidung begründen.
Die Eingaben müssen von BF16 nach BFP16 (``bfp16ebs8``) konvertiert werden.

Lösung
------

Wir wählen ein **p-outer, q-geblocktes** Blocking: zwei Durchläufe (``p=0`` und
``p=1``); pro Durchlauf bleiben **zwei Output-Akkumulatoren** (``out[p][0]`` und
``out[p][1]``) über alle ``r=0..7`` aktiv, und ``in0[p][r]`` wird einmal geladen
und für beide ``q`` wiederverwendet.

============  ============================================================
Tensor        Register
============  ============================================================
``out``       ``dm0`` (q=0), ``dm1`` (q=1) — leben über alle ``r``
``in0``       Load ``vlda.conv`` -> ``dm2`` (FP32) -> ``vconv`` -> ``ex0``
``in1``       Load ``vldb`` -> ``x2,x4`` / ``x3,x5`` (BF16),
              Transpose -> ``x6,x7`` / ``x8,x9``,
              ``vmul`` -> ``dm3`` / ``dm4`` (FP32),
              ``vconv`` -> ``ex6`` / ``ex8`` (BFP16)
============  ============================================================

Begründung:

- **Nur 5 Akkumulatoren (DM0–DM4):** zwei sind als Output dauerhaft belegt,
  also bleiben drei (``dm2``, ``dm3``, ``dm4``) als Staging für die FP32-Eingaben
  (in0, in1[q=0], in1[q=1]).
- **Maximaler Reuse / minimale M-Slot-Last:** ``in1`` ist teuer (Transpose +
  Konvertierung = 3 M-Ops/Kachel), ``in0`` billig (1 M-Op). Indem wir pro ``r``
  beide ``q``-Kacheln zusammen bearbeiten, konvertieren wir jede ``in1``-Kachel
  nur **einmal** und teilen ``in0[p][r]`` zwischen beiden MACs.
- **Konvertierungskette BF16 -> BFP16:** BFP16 kann nicht direkt transponiert
  werden (gemeinsamer Block-Exponent würde brechen), daher: ``in1`` als BF16
  laden, im BF16-Vektorbereich transponieren (``vshuffle``), per ``vmul`` mit
  einem **Einser-Vektor** nach FP32 heben, dann ``vconv`` nach BFP16. ``in0``
  (Layout ``mk`` schon korrekt) wird direkt per ``vlda.conv`` geladen.
- **Register-Wiederverwendung:** Die BFP16-Register ``ex6``/``ex8`` belegen
  physisch ``x6``/``x8`` (die Shuffle-Lo-Outputs von ``in1[q=0]``/``[q=1]``), die
  nach dem ``vmul`` frei sind. Der Einser-Vektor liegt in ``y5 = (x10,x11)``.

*Hinweis aus der Aufgabe:* Dasselbe Register darf für mehrere Tensoren
wiederverwendet werden, wenn sich ihre Lebenszeiten nicht überlappen — genau
das nutzen wir (``ex6 = x6`` usw.).
