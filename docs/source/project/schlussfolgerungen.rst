Schlussfolgerungen
------------------

.. Kein Wiederholen der Ergebnisse -- hier steht, was wir daraus gelernt haben.

Was der Tuner leistet
^^^^^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Die Kernaussage in einem Absatz, mit der belastbaren Zahl (1.12x GB10,
     1.32x 3070 gegen eine faire Baseline) und dem, was sie wert ist.
   - Der eigentliche Mehrwert praeziser gefasst als "schneller als alles": ein
     allgemeiner Mechanismus, der ohne Handarbeit ueber verschiedene Kontraktionen
     und Shapes brauchbare Leistung liefert -- und dort deutlich gewinnt, wo eine
     feste Wahl schlecht passt.

Modell und Messung sind komplementär
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Das ist die uebertragbarste Erkenntnis des Projekts und verdient einen
     eigenen Abschnitt: das analytische Modell taugt nicht als Ranker
     (Spearman ~0), aber als Startpunktlieferant. Der Abstieg taugt nicht als
     Startpunktsucher, aber zum Verfeinern. Keins allein reicht, zusammen
     erreichen sie 99 % bei 22 statt 342 Messungen.
   - Warum das kein Zufall ist: das Modell kennt die Physik grob, die Messung kennt
     die Realitaet punktuell.

Die Baseline entscheidet über die Aussage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Zweite uebertragbare Lehre: derselbe Datensatz ergibt 1.27x oder 1.12x, je
     nachdem wogegen man vergleicht. Die Wahl der Baseline war die folgenreichste
     methodische Entscheidung des Projekts -- folgenreicher als jede
     Code-Optimierung.
   - Dass die Frage aus dem Team selbst kam und zur Korrektur einer bereits
     praesentierten Zahl gefuehrt hat, gehoert dazu.

Was wir über die Hardware gelernt haben
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Das 25-MB-L2 der GB10 verschiebt die Story deutlich: die L2-Reuse-Regel aus
     der Vorlesung greift dort nicht, die Knappheit ist Compute/Occupancy.
   - Padding-Quantisierung schlaegt Arithmetic Intensity, sobald Shapes nicht
     glatt teilbar sind. Das war das ueberraschendste Einzelergebnis.
   - Messmethodik ist Teil des Ergebnisses: derselbe Kernel misst je nach
     Lastdauer 4.6 % anders.

Negativergebnisse
^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Bewusst als eigener Abschnitt: reiner Koordinatenabstieg schlechter als das
     Modell-Top-7, Multistart bringt nichts, der Ordnungs-Knopf bringt am Ende nur
     Rauschen, der weite Suchraum auf der GB10 nichts. Bei jedem laesst sich sagen
     warum -- und das ist wertvoller als ein weiterer Prozentpunkt.
   - Auch die widerlegte Vorhersage (Ordnungs-Knopf sollte auf der 3070 mehr
     bringen) hier festhalten.
