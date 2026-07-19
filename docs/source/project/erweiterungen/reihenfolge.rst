Loop-Reihenfolgen als Suchachse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Der ungenutzte Freiheitsgrad
""""""""""""""""""""""""""""

``Optimizer.permute_dims`` und ``fuse_dims`` stammen aus Assignment 05 und
können beliebige Dim-Reihenfolgen abbilden. Der Tuner hat davon aber nur zwei
fest verdrahtete Ordnungen benutzt, die Varianten A und B. Die Kernel haben die
Reihenfolge auch gar nicht aus der Config gelesen, sondern ``pid`` hart
dekodiert. Im Datenmodell war der Freiheitsgrad also da, im Suchraum nicht.

Vorab-Evidenz aus vorhandenen Daten
"""""""""""""""""""""""""""""""""""

Ob sich ein Kernel-Umbau lohnt, lässt sich vorher aus den schon gemessenen
Sweeps abschätzen.

Bei quadratischen Tiles mit ``m_prim == n_prim`` unterscheidet sich
``(m_l2=a, n_l2=b)`` von ``(m_l2=b, n_l2=a)`` nur durch die Swizzle-Richtung —
gleiche Gruppengröße, um 90° gedreht. Wäre die Richtung egal, müssten beide
Messungen gleich ausfallen.

Über die 15 Spiegelpaare in ``tune_a05.csv`` liegt der Median bei 1.41, im
Maximum bei 1.90. Die Richtung macht also einen Unterschied, obwohl der Tuner
sie zu dem Zeitpunkt nicht steuern konnte.

Umsetzung
"""""""""

``order`` kommt als siebter Knopf dazu und kodiert zwei Bits: welche Achse die
schnellste ``bid``-Komponente ist, und welche Gruppen-Achse außen zuerst läuft.
``order=0`` verhält sich bitgleich wie vorher, damit die älteren Messungen
gültig bleiben.

Der Knopf gilt nur für Variante A. Bei B sind ``m_l2`` und ``n_l2`` SEQ-Loops
im CTA, da gibt es kein Swizzling über die ``bid``.

Der Suchraum wächst entsprechend:

=================  ==========  ==============
Raum               enumeriert  nach Pruning
=================  ==========  ==============
Standard           486         342
``--ordered``      1215        855
=================  ==========  ==============

Ergebnis
""""""""

Der deutlichste Einzelfall ist ``a05``. Die Config ``A 128/128/64, m_l2=2,
n_l2=8``, die der ``--ordered``-Lauf auswählt, steht im Voll-Sweep bei 44.67
TFLOPS — der kennt nur ``order=0``. Mit ``order=2`` misst dieselbe Kachelung
67.86, also 52 % mehr bei identischen Tiles.

Das passt zum Mechanismus: ``m_l2=2`` ergibt eine nur zwei Blöcke hohe Gruppe.
Die alte Außenreihenfolge läuft erst alle N-Gruppen durch, bevor sie in der
M-Achse weiterrückt, und bei so einer flachen Gruppe fällt der A-Anteil dabei
immer wieder aus dem L2.

Am Endergebnis ändert das trotzdem wenig. Über alle Shapes bringt ``--ordered``
1.7 %, was im Rauschen des :ref:`Messrahmens <messrahmen>` liegt.
Der große Einzelgewinn entsteht nur, weil die Suche mit ``order`` überhaupt
erst eine Kachelung wählen darf, die sonst nie gewonnen hätte.

Isoliert gemessen
'''''''''''''''''

``measure_order.py`` hält die Kachelung fest (``128/128/64``, 8×8-Gruppe) und
misst nur die vier Reihenfolgen gegeneinander, im Round-Robin gegen thermische
Drift.

.. Bildpfade sind relativ zu docs/source/, weil alles ueber project.rst gerendert wird.
.. image:: ../../project/praesentation/figures/fig_order_effect.png
   :alt: Gewinn durch die beste Loop-Reihenfolge, GB10 gegen RTX 3070
   :width: 95%

Im Mittel bringt die beste Reihenfolge auf der GB10 1.026×, auf der RTX 3070
1.012×. Die Mediane liegen bei 0.7 % und 0.5 %, die Maxima bei 12.6 %
(``a06_krumm``, GB10) und 5.8 % (``a06_wide``, 3070). Ein Muster über die
Shapes ist nicht zu erkennen.

Zwei Einschränkungen
''''''''''''''''''''

Wir hatten erwartet, dass der Knopf auf der 3070 mehr bringt, weil das kleinere
L2 (4 MB gegen 25 MB) empfindlicher auf die Traversierung reagieren sollte.
Gemessen bringt er dort weniger, die Erwartung war also falsch.

Außerdem misst das Experiment die 8×8-Gruppe, also ausgerechnet den
quadratischen Fall, in dem die Swizzle-Richtung am wenigsten ausmacht. Der
Einzelfall oben zeigt den Effekt gerade bei einer flachen Gruppe.
``measure_order`` unterschätzt den Knopf damit; die 1.026× sind eher eine
untere Schranke.

Kosten
''''''

Der flexible Kernel kompiliert knapp doppelt so langsam wie der alte mit hart
dekodiertem ``pid``: 0.62 s statt 0.325 s pro Messung. Bei rund 22 gemessenen
Konfigurationen sind das etwa 6 s mehr pro Shape. Deshalb ist ``--ordered``
optional und nicht der Default.
