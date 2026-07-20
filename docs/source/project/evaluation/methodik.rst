Messmethodik
^^^^^^^^^^^^

Messaufbau
""""""""""

Gemessen wird mit ``triton.testing.do_bench`` (``warmup=50``, ``rep=200``). Vor
jeder Messung läuft ein Vergleich gegen ``torch.einsum``; wenn der fehlschlägt,
bekommt die Config ``ok=0`` und fällt aus allen Auswertungen raus. Die TFLOPS
rechnen wir auf der Original-Shape aus, nicht auf der gepaddeten — sonst würde
sich eine Config mit viel Padding ihre zusätzlichen FLOPs selbst gutschreiben.

Auf der GPU läuft nur die Messung, das Ergebnis geht als CSV raus. Ausgewertet
wird alles offline aus diesen CSVs, damit die Zahlen ohne Karte nachvollziehbar
bleiben.

=========================  ==============================  =============================
Skript                     Artefakt                        Inhalt
=========================  ==============================  =============================
``tune.py``                ``tune_*.csv``, ``study.log``    Voll-Sweep je Shape
``autotune.py``            ``autotune_hybrid*.csv``         Hybrid-Strategie
``check_coverage.py``      ``coverage_run.*``               einsum-Formen
``measure_order.py``       ``order_isolated.*``             Loop-Reihenfolge isoliert
``baseline_probe.py``      ``baseline_probe_winners.*``     Baseline-Konfigurationen
=========================  ==============================  =============================

.. _messrahmen:

Der Messrahmen-Effekt
"""""""""""""""""""""

Beim Auswerten ist uns ein Widerspruch aufgefallen. Der Hybrid misst rund 22
Konfigurationen, der Voll-Sweep 342 beziehungsweise 171, und der Sweep enthält
alles, was der Hybrid auch probiert. Der Sweep kann also nicht schlechter sein.
Trotzdem lag der Hybrid bei 8 von 16 Shapes darüber, auf ``a05`` um 4.6 %.

Wenn man statt der jeweils besten Werte dieselbe Konfiguration in beiden Läufen
nachschlägt, wird klar warum:

==============  ==========  =====================  ==========
Shape           Hybrid      Sweep, gleiche Config  Verhältnis
==============  ==========  =====================  ==========
``a05``         66.46       63.56                  104.6 %
``a06_square``  70.60       67.53                  104.6 %
``square_1b``   62.12       60.17                  103.2 %
``large_k``     45.38       44.06                  103.0 %
``a06_krumm``   20.44       20.66                  98.9 %
==============  ==========  =====================  ==========

Über alle 16 Shapes kommt dieselbe Config im Hybrid-Lauf auf 101.4 % ihres
Sweep-Werts, maximal auf 104.6 %, und in 11 von 16 Fällen liegt sie darüber.
Gleiche GPU, gleicher Tag, gleiche Config — unterschiedlich ist nur der Lauf.

Das liegt an der Dauer. Der Hybrid ist nach etwa 7 s fertig, ein Sweep braucht
100 bis 250 s. Die GB10 ist ein integrierter Chip mit gemeinsamem LPDDR und
teilt sich Power-Budget und Kühlung mit der CPU. Ein kurzer Lauf misst also eine
kältere Karte als ein langer.

Für die Evaluation heißt das: TFLOPS aus verschieden langen Läufen kann man
nicht direkt vergleichen. Wir benutzen stattdessen die Auswahlgüte — die vom
Tuner gewählte Config im Messrahmen des Sweeps gegen den Sweep-Besten. Dann
stammen beide Zahlen aus demselben Lauf und der Rahmen fällt raus.

Reproduzierbarkeit über Läufe hinweg
''''''''''''''''''''''''''''''''''''

Als Gegenprobe würde sich anbieten, zwei Sweeps derselben Methodik zu
vergleichen. ``result_dgx_v1`` und ``results_dgx_v2`` haben 4104
Konfigurationen gemeinsam, die in beiden Läufen durchgelaufen sind. Die stimmen
aber nicht besonders gut überein:

==============  ========  ==========
Abweichung      Paare     Anteil
==============  ========  ==========
unter 2 %       1614      39.3 %
2 – 5 %         1252      30.5 %
5 – 10 %        772       18.8 %
10 – 25 %       220       5.4 %
über 25 %       246       6.0 %
==============  ========  ==========

Als Gegenprobe taugt das nicht, weil zwischen den beiden Kampagnen die Korrektur
der Baseline liegt — es sind keine zwei Wiederholungen desselben Aufbaus. Die
Mediane pro Shape liegen zwischen 94.7 % (``a05``) und 100.2 %
(``a06_small_k``), die 6 % Ausreißer über 25 % betreffen einzelne Configs. Für
die Aussage oben spielt das keine Rolle, weil der Vergleich zwischen Hybrid und
Sweep komplett aus ``results_dgx_v2`` stammt.

.. _datenbasis:

Datenbasis und ihre Grenzen
"""""""""""""""""""""""""""

======================  =========  ==========================  ===============
Ordner                  Karte      Inhalt                      belastbar
======================  =========  ==========================  ===============
``result_dgx_v1``       GB10       Voll-Sweep, Handkernel      alte Baseline
``results_dgx_v2``      GB10       Voll-Sweep, Erweiterungen   ja
``result_3070``         RTX 3070   Voll-Sweep                  nur ``c ≥ 2``
``results_3070_v2``     RTX 3070   Erweiterungen, kein Sweep   ja
======================  =========  ==========================  ===============

Mit dem ersten 3070-Sweep stimmt etwas nicht. Alle fünf Shapes mit ``batch=1``
liegen dort um den Faktor 3.3 bis 4.8 zu niedrig, während alles mit ``c ≥ 2``
auf ±13 % mit dem späteren Lauf übereinstimmt. Genau diese Shapes brauchten im
alten Lauf 9 bis 14 s pro Konfiguration statt der sonst üblichen 0.5 bis 2 s.
Woran das lag, konnten wir nachträglich nicht mehr feststellen; die
WSL-Umgebung von damals gibt es nicht mehr.

Alles, was auf dem alten 3070-Sweep aufbaut, ist damit nicht belastbar. Das
betrifft vor allem den Cross-GPU-Hebel von 1.88× und die Randtreffer-Analyse.
Beide sind im :ref:`3070-Kapitel <rtx3070>` als zurückgezogen markiert.
