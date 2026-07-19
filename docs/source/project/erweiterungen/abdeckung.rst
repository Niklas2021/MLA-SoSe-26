Einsum-Abdeckung: Layout-Guard und Kanonisierung
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die stille Lücke
""""""""""""""""

``parse_einsum`` hat deutlich mehr akzeptiert, als die Kernel rechnen können,
und geprüft hat das nichts. ``cmk,cnk->cmn`` — also B in NT-Layout — parst
sauber durch, der Kernel liest dann aber das falsche Layout. Auffällig wird das
erst beim ``allclose`` am Ende, und dort ist es von einem echten Rechenfehler
im Kernel nicht zu unterscheiden. Andere Strings sind vorher am Shape-Unpack
abgestürzt.

Das ist die unangenehmere Sorte Fehler: nicht der, der crasht, sondern der, der
plausibel aussieht und falsch rechnet.

Zwei verschiedene Mechanismen
"""""""""""""""""""""""""""""

Beim Umsortieren von Achsen muss man zwei Fälle auseinanderhalten, und diese
Unterscheidung trägt das ganze Kapitel.

**Batch-Achsen umsortieren ist gratis.** Eine fehlende Batch-Achse ergänzen oder
mehrere zu einer fusionieren ist ein reiner ``view`` — es ändert nur, wie wir
den Speicher interpretieren, nicht den Speicher selbst.

**M/N/K umsortieren ist eine Transposition.** Kein Vertauschen von Indizes
ändert etwas daran, welche Achse physisch stride-1 liegt. Wer hier umsortieren
will, muss die Daten anfassen.

Bewusst ``view()`` und nicht ``reshape()``: ``reshape`` würde bei einem nicht
zusammenhängenden Tensor still eine Kopie anlegen. Eine unsichtbare Kopie mitten
im Benchmark wäre schlimmer als ein Fehler — sie würde einfach die Messung
verfälschen. ``view`` wirft stattdessen.

.. literalinclude:: ../../project/src/autotuner/layout.py
   :language: python
   :pyobject: plan_layout

Der flex-Kernel
"""""""""""""""

Für die Fälle, die eine echte Transposition brauchen, gibt es
``matmul_flex_a`` mit drei Transponier-Flags als ``ct.Constant``. Transponiert
wird auf Tile-Ebene über ``ct.permute`` nach dem Laden — derselbe Trick, den
``matmul_ring_a`` schon benutzt, und damit ohne Kopie im Speicher.

Das ist ein eigener Kernel geworden statt zusätzlicher Flags in
``matmul_variant_a``. Der ist auf beiden Karten erprobt und liefert die Zahlen
für die gesamte Evaluation; ein Fehler im neuen, deutlich selteneren Pfad sollte
ihn nicht mitreißen können.

Offen war dabei, ob cuTile über eine ``ct.Constant`` verzweigen kann — in
unseren dreizehn Kerneln gab es dafür keinen Präzedenzfall. Es funktioniert:
cuTile löst die Verzweigung beim Spezialisieren auf. Verifiziert auf GB10 und
RTX 3070.

Abdeckungstabelle
"""""""""""""""""

``check_coverage.py`` fährt alle Formen aus der ``COVERAGE``-Liste durch, lokal
als reine Layout-Prüfung und mit ``--run`` als echte Messung.

.. image:: ../../project/praesentation/figures/fig_coverage.png
   :alt: Abdeckung der einsum-Formen auf beiden Karten
   :width: 95%

Von ursprünglich zwei lauffähigen Familien — der A05-GEMM-Form und der
A06-Ring-Form — sind es jetzt zehn. Abgelehnt wird weiterhin ``mck,ckn->mcn``:
dort liegt die Batch-Dimension nicht außen, und das lässt sich nicht per
``view`` reparieren, weil es eine echte Transposition der Daten wäre.

Dass eine Form abgelehnt wird, ist dabei das gewünschte Verhalten. Vorher wäre
sie durchgelaufen und hätte falsch gerechnet.

.. literalinclude:: ../../project/results_dgx_v2/coverage_run.log
   :language: text
