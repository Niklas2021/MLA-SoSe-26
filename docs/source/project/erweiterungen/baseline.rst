Was ist eine faire Baseline?
============================

.. Quellen: project/src/baselines_study.py, project/src/baseline_probe.py,
   problems.py (BASELINE_CONFIGS)
   Dieses Kapitel korrigiert eine Kernaussage des Projekts -- entsprechend
   sorgfaeltig und ohne Beschoenigung schreiben.

Das Problem
-----------

.. Inhalt:
   - Bis hierher war die Vergleichsbasis DEFAULT_CONFIG, die aus A05 uebernommene
     Hand-Config 128/128/64, 8x8. Das verzerrt systematisch zugunsten des Tuners,
     am staerksten beim Cross-GPU-Vergleich.
   - Die Frage kam aus dem Team selbst ("war das eventuell unfair fuer unseren
     Ansatz?") -- das gehoert so erzaehlt, es zeigt die Selbstkorrektur.

Warum die A05-Config als Baseline versagt
-----------------------------------------

.. Inhalt:
   - Sie ist nicht "die GB10-Config", sondern die Config fuer EINE Shape: auf ihrer
     Heimat-Shape a05 holt sie 97.2 % des Optimums, auf a06 nur 44 %.
   - Der Grund ist die GRUPPEN-AUSDEHNUNG, nicht die Tile-Groesse:
     m_l2*m_prim = 8*128 = 1024, also werden M und N auf Vielfache von 1024
     hochgepaddet. Bei a06 (x=1536, y=1152) ist das gepaddete Volumen das
     2.37-fache des echten.
   - Die quantitative Gegenprobe ausschreiben: teilt man die Padding-Ueberhaenge
     beider Baselines, sagt das den gemessenen Abstand fast exakt vorher
     (a06 vorhergesagt 1.78x, gemessen 1.77x). Und in die Gegenrichtung: wo nichts
     gepaddet wird, ist die neue Baseline leicht SCHLECHTER (0.95-1.00x), genau wie
     es das Arithmetic-Intensity-Argument verlangt. Beide Richtungen stimmen.

Warum 128/128/64 trotzdem eine vernünftige Wahl war
---------------------------------------------------

.. Inhalt:
   - Die Herleitung ist lehrbuchmaessig sauber und sollte nachgezeichnet werden:
     quadratische Tiles minimieren (M+N)*K bei gegebenem Akkumulator M*N;
     128x128 fp32 = 16384 Register = 64 KB von 256 KB pro SM, also vier Bloecke;
     Operanden-SMEM 64 KB passt ins 100-KB-Opt-in; 8x8 maximiert den L2-Reuse nach
     dem Triton-Grouping-Argument.
   - Der Fehler liegt nicht in der Kette, sondern in dem, was sie ignoriert: sie
     optimiert den eingeschwungenen Zustand und uebersieht die QUANTISIERUNG am
     Rand. Das ist die Pointe des Kapitels.

Die neue Baseline
-----------------

.. Inhalt:
   - BASELINE_CONFIGS: eine feste Config pro GPU (GB10 64/256/64 8x2,
     3070 64/128/64 8x2). Ohne Messung begruendbar: gleicher Akkumulator
     (64*256 = 128*128 = 16384 Register), aber halbe Gruppenausdehnung. Man
     tauscht ~25 % Arithmetic Intensity gegen weniger Padding-Quantisierung.
   - baselines_study.py leitet sie reproduzierbar aus den Vollmessungen her,
     inklusive leave-one-out (89.5 % gegen 90.9 % oracle -> die Wahl ist robust,
     kein Nachwissen-Artefakt).
   - baseline_probe.py fuer Karten ohne Vollmessung: 13x billiger, findet eine
     Config die 89.6 % statt 90.9 % holt. Der Bias ist bekannt und beziffert.

Auswirkung auf die Aussagen des Projekts
----------------------------------------

.. Inhalt:
   - Der ehrliche Tuner-Gewinn auf der GB10 ist 1.12x, nicht 1.27x.
   - Die beiden Fragen sauber trennen:
     1. Was kostet es, beim GPU-Wechsel nicht neu zu tunen? -> 2.56x auf der 3070.
        Das rechtfertigt den GPU-Modell-Key im Cache.
     2. Was bringt Per-Shape-Tuning gegen eine kompetent gewaehlte feste Config
        derselben Karte? -> 1.12x (GB10), 1.32x (3070).
   - Das Gegenargument, das dem Tuner zusteht: an die "kompetent gewaehlte feste
     Config" kommt man nur ueber einen vollen Multi-Shape-Sweep auf der Zielkarte
     -- genau das, was der Tuner ersetzt. Der wahre Wert liegt zwischen 1.12x und
     1.27x.
