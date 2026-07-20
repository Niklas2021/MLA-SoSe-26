Schlussfolgerungen
------------------

.. Kein Wiederholen der Ergebnisse -- hier steht, was wir daraus gelernt haben.

Was der Tuner leistet
^^^^^^^^^^^^^^^^^^^^^

Der Ausgangspunkt war die Erfahrung aus Assignment 05 und 06: die Config einer
Tensor-Kontraktion lässt sich von Hand herleiten, aber die Herleitung hängt an der
konkreten Shape und an der GPU und müsste für jede neue Kombination von vorn
gemacht werden. Genau diesen Schritt automatisiert der Tuner — er nimmt einen
Einsum-String samt Shapes und liefert eine cuTile-Config, deren Güte per Messung
bestätigt ist.

Das ist vollständig umgesetzt und läuft auf beiden Testkarten. Der Tuner deckt
zehn der elf geprüften Einsum-Formen ab, und auf der GB10, wo jede geprunte Config
einmal gemessen wurde, gab es dabei keinen einzigen Korrektheitsfehler.
Enumerator, Pruning, beide Kernel-Familien samt flexiblem Transpositions-Kernel,
die Hybrid-Suche und der Cache mit GPU-Schlüssel sind portabel und nirgends auf
eine der beiden Karten festgeschrieben.

Der gemessene Vorsprung gegenüber einer je Karte kompetent gewählten festen Config
ist mit 1.12× auf der GB10 und 1.32× auf der 3070 moderat; auf einer einzelnen,
gut passenden Shape kommt auch eine von Hand gewählte Config nah heran. Der Nutzen
liegt in der Breite. Der Tuner findet die passende Config für jede Shape und jede
Karte selbst, und wo eine feste Wahl schlecht passt, wird der Abstand groß: die
A05-Config holt auf ihrer Heimat-Shape 97 % des Optimums, auf ``a06`` nur 44 % und
auf die 3070 übertragen rund 62 %. Vor solchen Fehlgriffen schützt die automatische
Wahl, weil sie je Shape neu entscheidet.

Modell und Messung sind komplementär
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Die Suche kombiniert ein analytisches Kostenmodell mit echten Messungen, und die
übertragbarste Erkenntnis des Projekts ist, dass keiner der beiden Teile allein
genügt. Das Modell schätzt die Laufzeit aus Bandbreite, Occupancy und Roofline. Es
trifft damit den groben Trend, aber nicht die Feinordnung: je nach Modellvariante
liegt die Rangkorrelation zur gemessenen Laufzeit zwischen +0.02 und +0.50, zu
wenig, um den tatsächlichen Gewinner zu benennen. Als Vorfilter reicht es dagegen —
die sieben :ref:`modellbesten <ranking>` Configs holen im Mittel 96.6 % des
Optimums. Der Koordinatenabstieg allein hat das umgekehrte Problem: von einem
einzelnen Startpunkt aus bleibt er an lokalen Optima hängen und kommt nur auf
93.6 %, schlechter als die sieben Modellkandidaten und mit mehr Messungen. Der
Hybrid setzt beide an ihre jeweilige Stärke — das Modell liefert die Startpunkte,
der Abstieg verfeinert von der gemessen besten aus — und erreicht so 99.1 % des
Voll-Sweep-Optimums bei rund 22 statt 342 Messungen.

Was wir über die Hardware gelernt haben
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drei Beobachtungen über die Hardware haben die Erwartung aus der Vorlesung
korrigiert. Die erste betrifft das große L2 der GB10. Das Grouping-Argument — eine
große L2-Gruppe senkt den Operanden-Traffic und gewinnt deshalb — gilt nur,
solange der Working-Set nicht ohnehin in den Cache passt. Mit 25 MB passt er auf
der GB10 fast immer; das Argument greift dort kaum, und die Karte ist compute-
statt bandbreitenlimitiert. Auf der :ref:`3070 <rtx3070>` mit ihren 4 MB liegt
eine einzelne feste Config je Shape viel weiter vom jeweiligen Optimum entfernt
(75.9 % gegen 89.5 % auf der GB10), weshalb Per-Shape-Tuning dort mehr bringt
(1.32× gegen 1.12×). Dass das kleine L2 die Ursache ist, ist plausibel, mangels
3070-Voll-Sweep aber nicht abschließend belegt.

Die zweite Beobachtung: sobald eine Shape nicht glatt in die Gruppenausdehnung
teilt, entscheidet der Padding-Verschnitt an den Rändern und nicht die Arithmetic
Intensity des eingeschwungenen Zustands. Das Verhältnis der Padding-Überhänge sagt
den gemessenen Abstand zweier fester Configs fast exakt vorher (für ``a06`` 1.78×
vorhergesagt, 1.77× gemessen). Das war das überraschendste Einzelergebnis, weil die
übliche Herleitung genau diesen Randeffekt übergeht.

Die dritte Beobachtung ist methodisch: dieselbe Config misst je nach Dauer des
umgebenden Laufs bis zu 4.6 % unterschiedlich, weil sich die integrierte GB10
Power-Budget und Kühlung mit der CPU teilt. Absolute TFLOPS aus verschieden langen
Läufen sind damit nicht direkt vergleichbar; belastbar wird der Vergleich erst über
die Auswahlgüte im festen :ref:`Messrahmen <messrahmen>`.

Negativergebnisse
^^^^^^^^^^^^^^^^^

Mehrere naheliegende Ideen haben sich beim Nachmessen nicht bestätigt. Zusätzliche
Startpunkte bringen dem Hybrid fast nichts: drei weitere Seeds heben ihn von
99.7 % auf 99.8 % und kosten zwei Messungen mehr. Der Reihenfolge-Knopf gewinnt in
einem konstruierten Einzelfall 52 % auf identischen Tiles, über alle Shapes
gemittelt aber nur 1.7 % und damit im Rauschen des Messrahmens. Der
:ref:`weite Suchraum <erweiterter-suchraum>` ist auf der GB10 netto ein
Nullsummenspiel bei doppeltem Messaufwand; sein einziger echter Gewinn
(``a06_krumm``, +46 %) entsteht allein daraus, dass ``k_prim=16`` die
Reduktionsachse ``p=48`` exakt teilt — was der Shape ohne Messung anzusehen ist.
Eine Vorhersage hat sich sogar ins Gegenteil verkehrt: der Reihenfolge-Knopf sollte
auf der 3070 stärker wirken, weil ihr kleineres L2 empfindlicher auf die
Traversierung reagieren müsste, gemessen bringt er dort aber weniger als auf der
GB10 (1.012× gegen 1.026×). In jedem dieser Fälle lässt sich der Grund benennen,
und ein verstandener Negativbefund ist für die weitere Arbeit nützlicher als ein
knapp gewonnener Prozentpunkt.
