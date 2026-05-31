Task 6: Performance
====================

Aufgabenstellung
----------------

1. Die Instruktionen zählen. Welche Performance sollte der Kernel erreichen?
2. Argumentieren, ob die Instruktionszahl minimal ist, bzw. beschreiben, welche
   Optimierungen sie weiter reduzieren könnten.

Lösung
------

Instruktionszählung
~~~~~~~~~~~~~~~~~~~~~

Der Kernel rechnet 4 Output-Kacheln × 8 ``r`` = **32** native ``8x8x8``-MACs
(``vmac.f``/``vmul.f``), die theoretisch minimale Anzahl für ``16x16x64``.
Ebenso minimal sind die Konvertierungen: dank des 2×2-Reuse wird jede
Eingabekachel genau einmal nach BFP16 konvertiert.

Pro ``r`` fallen an: 2 Loads (A) + 4 Loads (B) + 7 M-Ops (1 ``vconv`` in0,
4 ``vshuffle``, 2 ``vconv`` in1) + 4 V-Ops (2 ``vmul``, 2 ``vmac``) = **17
Nutz-Ops**. Der **M-Slot mit 7 Ops/r** ist die am stärksten belastete Einheit.

Wir haben zwei Versionen umgesetzt:

==================================  ==========  ==========  ==========
Version                             Zyklen/r    Zyklen ges  GFLOPS
==================================  ==========  ==========  ==========
First try (eng, nicht pipelined)    25          ≈ 438       ≈ 135
Pipelined (``II = 16``)             16          ≈ 300       ≈ 197
==================================  ==========  ==========  ==========

(Gesamtzyklen = Prolog 8 + 2 Pässe + ``ret``/Delay 6; ``f = 1.8 GHz``,
``GFLOPS = 2·16·16·64·f / Zyklen``.)

Die pipelined Version ist **1,46× schneller** und erreicht **10,7 %** des
theoretischen Peaks (1843,2 GFLOPS = ``8x8x8 ·2 ·1.8 GHz``, ein ``vmac`` pro
Zyklus).

*Hinweis zur Messung:* Der ``[bench]``-Wert von ``driver.py`` (~0,3 GFLOPS,
~110 µs/call) misst **End-to-End** und ist fast vollständig Host-/DMA-Startkosten
— die reine Rechnung dauert nur ≈ 0,2 µs (0,2 % der Aufrufzeit). Maßgeblich für
die Compute-Performance ist daher der analytische Zyklenwert; der End-to-End-Wert
unterscheidet die Versionen nicht sichtbar (Overhead identisch).

Angewandte Optimierung: Software-Pipelining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die NOPs des ``first try`` sind „Latenz-Schatten" der Abhängigkeitskette
``load (7) → shuffle (2) → vmul (6) → vconv (4) → vmac`` (≈ 19 Zyklen). Innerhalb
einer ``r``-Iteration sind sie unvermeidbar; über Iterationsgrenzen hinweg
füllbar. Die pipelined Version zieht daher die **Loads von ``r+1`` in die
Wartezyklen von ``r``** vor — der Front-Load-Stall (4 NOPs/r) entfällt, der Body
schrumpft von 25 auf 16 Zyklen. Voraussetzung war eine Register-Umlegung (BFP16
auf die Shuffle-Outputs ``ex6``/``ex8``, damit die Roh-Lade-Register früh frei
werden), das Vorziehen von ``vconv ex0`` hinter die vier Shuffles (q1 lagt dann
nur um 2 statt 3, die zwei ``vmac`` rücken zusammen) und das „Peelen" der letzten
``r``-Iteration (kein Prefetch über das Kachelende hinaus).

Grenzen / weitere Optimierungen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Register-Druck als harte Grenze:** Bei 12 Vektor- und 5 Akku-Registern
  (2 fest für die Outputs) müssen sich Lade- und BFP16-Register überlappen. Das
  erzwingt eine iterationsübergreifende Abhängigkeit (``r+1``s Shuffle darf erst
  nach ``r``s ``vmac`` schreiben), sodass ``II`` nicht bis zum M-Slot-Limit
  (7 Zyklen/r) sinken kann — ``II = 16`` ist praktisch das Erreichbare ohne
  anderes Blocking.

- **M-Slot ist der eigentliche Flaschenhals**, getrieben vom erzwungenen
  ``kn → nk``-Transpose von ``in1`` (4 ``vshuffle`` + Konvertierungen pro ``r``).
  Käme ``in1`` schon im ``nk``-Layout über den DMA (``rqnk``), entfiele der
  gesamte Shuffle/``vmul``-Pfad → M-Slot ≈ halbiert → Richtung 1 ``vmac``/Zyklus
  = Peak. Das liegt aber im (fixen) MLIR-Datenlayout, nicht im Kernel.

- **``out``-Load sparen:** ``out`` ist null-initialisiert; eine ``vmul``-
  Initialisierung für ``r=0`` (statt ``out`` zu laden) spart 8 Loads — kleiner
  Gewinn.

Fazit: Die MAC- und Konvertierungszahl ist minimal; die NOP-Zahl wurde per
Pipelining deutlich gesenkt (438 → 300 Zyklen, 1,46×). Das verbleibende Gap zum
Peak ist **register- und transpose-gebunden**, nicht durch überflüssige
Instruktionen.
