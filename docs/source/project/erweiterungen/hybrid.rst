Messgesteuerte Suche (Hybrid)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ausgangsproblem
"""""""""""""""

Die Suche war bis dahin einstufig: Modell ranken, die besten sieben messen,
fertig. Damit entscheidet das Modell allein, welchen Teil des Raums wir
überhaupt anfassen — obwohl wir aus dem :ref:`Ranking-Kapitel <ranking>`
wissen, dass es als Ranker kaum etwas taugt (Spearman um 0).

Aus dem Feedback nach der Präsentation kam der Vorschlag, zwischendurch echt zu
messen und aus den Messungen heraus weiterzusuchen, statt sich auf die
Modellreihenfolge zu verlassen.

Verfahren
"""""""""

Der Hybrid misst zuerst die Modell-Top-7 und steigt dann von der gemessen besten
Config aus achsenweise ab: pro Achse alle Nachbarwerte probieren, die beste
übernehmen, weiter zur nächsten Achse. Das läuft, bis ein kompletter Durchlauf
nichts mehr verbessert.

Die Achsen werden nach gemessener Wichtigkeit abgearbeitet. ``m_prim`` und
``n_prim`` laufen dabei gemeinsam, weil sie über das Registerbudget
zusammenhängen — einzeln optimiert läuft der Abstieg in 256×256 und bleibt dort
stecken.

.. literalinclude:: ../../project/src/autotuner/strategies.py
   :language: python
   :pyobject: hybrid

.. literalinclude:: ../../project/src/autotuner/strategies.py
   :language: python
   :pyobject: descend

Die Callback-Architektur
""""""""""""""""""""""""

Die Strategien messen nicht selbst, sondern bekommen eine ``measure``-Funktion
übergeben. Auf der GPU reicht ``autotune.py`` eine Funktion herein, die den
Kernel startet; offline reicht ``simulate_search.py`` eine herein, die in der
CSV nachschlägt.

Dadurch läuft in beiden Fällen derselbe Strategie-Code. Eine Simulation, die
ihre eigene Kopie des Suchverfahrens hätte, würde über die Zeit vom echten Tuner
abdriften, ohne dass es jemandem auffällt — so kann sie das nicht. Die
Offline-Auswertung unten ist deshalb überhaupt erst belastbar.

Ergebnisse
""""""""""

``simulate_search.py`` fährt alle Strategien gegen die Vollmessungen aus
``results_dgx_v2``. Jede „Messung" ist ein Lookup in der CSV, das kostet keine
GPU-Zeit.

.. image:: ../../project/praesentation/figures/fig_hybrid_vs_sweep.png
   :alt: Hybrid gegen Voll-Sweep, Ausbeute und Messaufwand
   :width: 95%

=========================  ===========  ==============
Strategie                  Messungen    vom Optimum
=========================  ===========  ==============
Default (8×8)              0            81.4 %
Modell-Top-7               7            96.6 %
Hybrid                     22           99.7 %
Vollmessung                256          100.0 %
=========================  ===========  ==============

Auf der GB10 selbst kommt der Hybrid auf 99.1 % — etwas schlechter als in der
Simulation, weil dort echte Messstreuung dazukommt.

Zwei Negativbefunde
'''''''''''''''''''

Der Abstieg allein reicht nicht. Startet man ihn nur von der besten
Modell-Config aus statt von sieben, fällt die Ausbeute auf **93.6 %** — und das
bei 18 Messungen, also mehr als die 7 des reinen Modellwegs, der 96.6 %
erreicht. Ein Greedy-Abstieg hängt am Startpunkt, und ein einzelner Startpunkt
ist zu wenig.

Mehr Startpunkte helfen umgekehrt auch kaum. Mit drei zusätzlichen Seeds aus dem
Ranking kommt der Hybrid auf 99.8 % statt 99.7 %, kostet dafür aber zwei
Messungen mehr. Das Verhältnis lohnt nicht.

Die schwächste Shape bleibt ``krumm`` mit 98.1 %. Dort sind die Achsen nicht
unabhängig voneinander: die Padding-Verluste hängen von ``m_prim`` und
``n_prim`` gemeinsam ab, und genau solche Kopplungen sieht ein achsenweiser
Abstieg nicht.

Zusammen ergibt das ein klares Bild: das Modell taugt nicht als Ranker, aber gut
als Lieferant von Startpunkten. Der Abstieg korrigiert, wo das Modell
danebenliegt. Keins von beiden reicht allein.

Monotonie über extra_seeds
""""""""""""""""""""""""""

Ein Problem, das erst auf der Hardware auftrat: ein *größerer* Suchraum kann den
Hybrid verschlechtern. Beim Wechsel auf ``--wide`` brach ``a06`` von 61.7 auf
49.2 TFLOPS ein.

Der Grund ist nicht, dass die bessere Config verschwindet — sie ist weiterhin im
Raum. Aber ein größerer Raum ändert auch das Ranking und damit die sieben
Startpunkte, und von den neuen Startpunkten aus landet der Abstieg woanders.

Die Lösung steckt in ``extra_seeds``: ein Cache-Treffer aus einem früheren,
engeren Lauf taugt vielleicht nicht mehr als Antwort, aber sehr wohl als
zusätzlicher Startpunkt. Damit kann ein größerer Raum per Konstruktion nicht
schlechter werden als der kleinere — schlimmstenfalls gewinnt wieder der alte
Startpunkt. Kostet eine Messung.

Danach liegt ``a06`` bei −0.3 % statt −20 %. Der Selbsttest in
``strategies.py`` baut den Fall nach, damit die Eigenschaft nicht still wieder
verlorengeht.
