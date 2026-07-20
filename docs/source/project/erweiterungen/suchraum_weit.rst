.. _erweiterter-suchraum:

Erweiterter Suchraum
^^^^^^^^^^^^^^^^^^^^

Motivation aus der Randanalyse
""""""""""""""""""""""""""""""

Ein Suchraum ist zu klein, wenn die beste Config an seinem Rand liegt — dann
zeigt der Gewinner in eine Richtung, in die das Gitter nicht weiterreicht. Auf
der GB10 sitzen 33 der 80 Gewinner-Koordinaten (5 Tile-Achsen über 16 Shapes,
41 %) auf einem Gitterrand. Vor allem am unteren: ``k_prim`` erreicht sein
Maximum 128 nur ein einziges Mal und steht 7 von 16 Mal auf dem kleinsten Wert
32, ``m_prim`` wählt nie die 256, ``n_prim`` nie die kleinste 64.

Das deutet auf ein Gitter, das am unteren Ende abgeschnitten ist. Die Erklärung
ist naheliegend: der Suchraum wurde auf der GB10 entworfen und passt zu ihr.
Kleinere Tiles als ``m_prim=64`` oder ``k_prim=32`` waren nie vorgesehen, obwohl
die Messungen sie an mehreren Stellen anfragen.

Umsetzung
"""""""""

``SearchSpace.wide()`` nimmt die kleineren Tiles dazu — ``m_prim`` und ``n_prim``
zusätzlich mit 32, ``k_prim`` zusätzlich mit 16:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — die erweiterten Tile-Größen
   :start-at: WIDE_MN_PRIM_CHOICES = [32, 64, 128, 256]
   :end-at: WIDE_K_PRIM_CHOICES = [16, 32, 64, 128]

Der Wert 32 bleibt mit ``MMA_ALIGN=16`` sauber teilbar, verletzt also keinen der
Prune-Filter. Der Raum wächst von 486 auf 1152 Kandidaten, nach dem Pruning von
342 auf 954.

Der weite Raum ist bewusst nicht der Default. Alle bisherigen Messungen und die
komplette Hybrid-Auswertung beziehen sich auf den engen Raum; ein stiller Wechsel
würde die Vergleichbarkeit zerstören. Er ist über ``python autotune.py hybrid
--wide`` erreichbar. Der Cache hält beide auseinander: er speichert die
``space_size`` mit, und ein im engen Raum getunter Eintrag (486) bedient keine
``--wide``-Anfrage (1152), weil er die kleinen Tiles nie gemessen hat — dieselbe
Logik wie bei ``topk`` gegen ``hybrid``.

Ergebnis
""""""""

Auf der GB10 ist der weite Raum netto ein Nullsummenspiel bei doppeltem
Messaufwand. In 7 von 16 Shapes wählt er exakt dieselbe Config wie der enge Raum.
Klammert man die zwei Shapes aus, bei denen die kleinen Tiles wirklich etwas
ändern, liegt das geometrische Mittel bei 100.2 % — die einzelnen Abweichungen
nach unten auf 97 % sind der :ref:`Messrahmen-Effekt <messrahmen>`, weil der
``--wide``-Lauf länger dauert und dadurch etwas niedriger misst.

Der eine große Gewinn lässt sich sauber erklären. Bei ``a06_krumm`` steigt die
Leistung von 20.4 auf 29.9 TFLOPS (+46 %), und der weite Raum wählt dort
``k_prim=16`` statt 64. Die Reduktionsachse dieser Shape ist ``p = 48``: mit
``k_prim`` aus ``{32, 64, 128}`` musste auf 64 hochgepaddet werden, also 33 %
Verschnitt auf der K-Achse, während ``48 = 3·16`` genau aufgeht. Denselben Effekt
in klein gibt es bei ``a06_small_k`` (ebenfalls ``k_prim=16``, +2.5 %).

Der Gewinn kommt also nicht daher, dass kleine Tiles grundsätzlich besser wären,
sondern daher, dass sie die Shape teilen. Und ob sie das tun, sieht man der Shape
an, ohne zu messen.

Auf der RTX 3070 sollte der weite Raum mehr bringen, weil das kleine L2 (4 MB
gegen 25 MB) kleinere Tiles bevorzugt. Eine belastbare Zahl gibt es dafür aber
nicht: der ``--wide``-Lauf auf der 3070 hat keinen sauberen Gegenpart im engen
Raum auf derselben Karte, und der alte Sweep, gegen den man vergleichen müsste,
ist für ``batch=1`` unbrauchbar (siehe :ref:`Datenbasis <datenbasis>`). Die
Richtung ist plausibel, die Größe des Effekts bleibt hier offen.

Adaptiver Suchraum
""""""""""""""""""

Aus dem GB10-Ergebnis folgt, wie man es besser macht: den Raum nicht global
verdoppeln, sondern die kleinen Tiles pro Achse nur dort dazunehmen, wo die Shape
sie braucht. Der a06_krumm-Gewinn kam ja nicht von kleinen Tiles an sich, sondern
davon, dass ``k_prim=16`` die Reduktionsachse ``p=48`` exakt teilt. Und ob eine
Achse durch den Standard-Boden teilbar ist, sieht man ihr an, ohne zu messen.

``adaptive_space`` setzt genau das um: ein Tile unter dem Boden (``m_prim`` oder
``n_prim`` auf 32, ``k_prim`` auf 16) kommt nur dazu, wenn die Achse nicht schon
durch den Boden (64 bzw. 32) teilbar ist.

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — Suchraum aus der Shape ableiten
   :pyobject: adaptive_space

Das beantwortet auch, wie weit man Tile-Größen aus der Hardware berechnen kann.
Die **Obergrenze** kommt aus Hardware-Konstanten und steckt schon im Pruning: das
SMEM- und Register-Budget der Karte begrenzt, wie groß ein Tile sein darf. Der
**Boden** (64 für M/N, 32 für K) ist ebenfalls hardware-informiert — kleiner wird
die ``mma``-Kachel ineffizient. Nur wie weit man im Einzelfall unter den Boden
geht, entscheidet nicht die Hardware, sondern die Teilbarkeit der konkreten Shape.
Die Tile-Wahl ist also eine Rechnung aus beidem, Hardware und Shape, und braucht
keine Messung.

Man könnte meinen, der L2-Cache gehöre hier auch hinein — kleiner Cache, kleinere
Tiles. Das trifft aber den falschen Knopf. Die Tile-Größe lebt pro CTA in
Registern und Shared Memory, nicht im L2; der L2 begrenzt die **Gruppe**
(``m_l2``, ``n_l2``), also wie viele Tiles sich die nachgeladenen A/B-Streifen
teilen. Diese Schranke steht auch im Traffic-Modell — der Gruppen-Working-Set
``(m_l2·m_prim + n_l2·n_prim)·K`` gegen ``dev.l2_cache`` —, sie bindet auf der
GB10 mit ihren 25 MB nur nie. Auf einer Karte mit kleinem L2 würde sie greifen und
die Gruppe verkleinern; die Tile-Größe bliebe davon unberührt.

Dass es funktioniert, lässt sich vollständig aus den schon vorhandenen Messungen
zeigen, ohne die GPUs erneut zu belegen. Für jede der 16 Shapes ist der adaptive
Raum entweder exakt der enge oder exakt der volle weite Raum — beide sind
gemessen. Bei den 14 teilbaren Shapes ist er der enge, der adaptive Tuner
verhält sich dort also identisch zum engen Lauf. Nur ``krumm`` und ``a06_krumm``
sind auf allen drei Achsen unteilbar und bekommen den weiten Raum.

Das Ergebnis des adaptiven Tuners ist damit für jede Shape ein bereits gemessener
Lauf:

======================  =====================  ===========================
Shape                   Raum                   Ergebnis gegenüber eng
======================  =====================  ===========================
14 teilbare Shapes      eng (= Standard)       unverändert
``krumm``               weit                   39.6 → 39.4 (im Messrauschen)
``a06_krumm``           weit                   20.4 → 29.9 TFLOPS (+46 %)
======================  =====================  ===========================

Der adaptive Raum enumeriert über alle 16 Shapes zusammen rund die Hälfte der
Kandidaten, die ein global weiter Raum kosten würde (9108 gegen 18432), holt aber
den einen echten Gewinn (a06_krumm, +46 %) vollständig mit. Aufrufbar über
``python autotune.py hybrid --adaptive``.
