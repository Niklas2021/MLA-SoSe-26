Messgesteuerte Suche (Hybrid)
=============================

.. Quellen: project/src/autotuner/strategies.py, project/src/simulate_search.py
   Das ist die inhaltlich staerkste Erweiterung -- entsprechend ausfuehrlich.

Ausgangsproblem
---------------

.. Inhalt:
   - Bisher war die Suche einstufig: Modell ranken, Top-7 messen, fertig. Das
     Modell entscheidet damit allein, welcher Teil des Raums angefasst wird --
     obwohl wir wissen, dass es als Ranker nichts taugt (Spearman ~0).
   - Die Feedback-Idee: zwischendurch echt messen und daraus ganze Regionen
     ausschliessen.

Verfahren
---------

.. Inhalt:
   - Modell-Top-7 messen, dann von der gemessen besten Config aus achsenweise
     absteigen, bis sich in einem vollen Durchlauf nichts mehr aendert.
   - Achsenreihenfolge nach gemessener Wichtigkeit; m_prim/n_prim gemeinsam, weil
     sie ueber das Registerbudget zusammenhaengen (einzeln laeuft der Abstieg in
     256x256 und bleibt stecken).
   - literalinclude hybrid und descend.

Die Callback-Architektur
------------------------

.. Inhalt:
   - Design-Punkt, der eigens erwaehnt gehoert: die Strategien messen nicht selbst,
     sondern bekommen eine measure-Callback. Dadurch laeuft DERSELBE Code gegen die
     GPU (autotune.py) und gegen die CSVs (simulate_search.py) -- die Simulation
     kann nicht vom echten Tuner abdriften.
   - Das macht die Offline-Evaluation ueberhaupt erst belastbar.

Ergebnisse und Negativbefunde
-----------------------------

.. Inhalt:
   - Tabelle: Default (0 Messungen) / Modell-Top-7 (7) / Hybrid (22) / Vollmessung
     (256+) mit Optimum-Ausbeute. Zahlen aus simulate_search.py gegen
     results_dgx_v2 (99.7 %) und aus dem echten GB10-Lauf (99.1 %).
   - Zwei Negativbefunde, die dazugehoeren:
     * reiner Koordinatenabstieg ohne Modell ist SCHLECHTER als das bisherige
       Top-7 (95.0 % bei 14 Messungen) -- er haengt am Startpunkt.
     * Multistart bringt nichts (99.0 % bei 32 statt 22 Messungen).
   - Die Grenze: krumm bleibt bei 86.3 %, weil die Achsen nicht unabhaengig sind.
   - Schlussfolgerung: Modell und Messung sind komplementaer, keins allein reicht.

Monotonie über extra_seeds
--------------------------

.. Inhalt:
   - Das Problem, das erst auf der Hardware sichtbar wurde: ein groesserer
     Suchraum kann den Hybrid VERSCHLECHTERN, weil er auch die Seeds und damit den
     Abstiegspfad aendert (a06 brach von 61.7 auf 49.2 TFLOPS ein).
   - Die Loesung: ein zu schwacher Cache-Treffer taugt nicht als Antwort, aber als
     zusaetzlicher Startpunkt. Damit ist ein Upgrade per Konstruktion nie
     schlechter. Kostet eine Messung.
   - Danach: a06 bei -0.3 % statt -20 %. Der Selbsttest in strategies.py baut den
     Fall nach.
