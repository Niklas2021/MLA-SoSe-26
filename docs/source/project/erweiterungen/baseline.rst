Eine faire Baseline
^^^^^^^^^^^^^^^^^^^

Das Problem
"""""""""""

Bis hierher war die Vergleichsbasis für den „Tuner-Gewinn" immer
``DEFAULT_CONFIG`` — die aus A05 übernommene Hand-Config ``128/128/64, 8×8``.
Beim Nachrechnen fiel auf, dass das die Ergebnisse systematisch zugunsten des
Tuners verzerrt, am stärksten beim Cross-GPU-Vergleich, wo wir den
Optimierungshebel als „auf der 3070 stärker, Ø 1.88×" berichtet hatten.

Die Frage kam aus dem Team selbst: war diese Baseline vielleicht unfair für den
eigenen Ansatz? Sie war es, und dieses Kapitel korrigiert die Aussage.

Warum die A05-Config als Baseline versagt
"""""""""""""""""""""""""""""""""""""""""

Die Config ``128/128/64, 8×8`` ist nicht „die GB10-Config", sondern die Config
für *eine* Shape. Auf ihrer Heimat-Shape ``a05`` holt sie 97.2 % des Optimums,
auf ``a06`` nur 44 %.

Der Grund ist die Gruppen-Ausdehnung, nicht die Tile-Größe. Mit
``m_l2·m_prim = 8·128 = 1024`` werden M und N auf Vielfache von 1024
hochgepaddet. Bei ``a06`` (x=1536, y=1152) ist das gepaddete Volumen dadurch das
2.37-fache des echten — fast die Hälfte der Rechenzeit fließt in Nullen.

Dass das Padding die Ursache ist, lässt sich nachrechnen: teilt man die
Padding-Überhänge beider Baselines, sagt das den gemessenen Abstand fast exakt
vorher.

==============  ============  ============  ============  ========
Shape           Padding Def   Padding fest  vorhergesagt  gemessen
==============  ============  ============  ============  ========
``a06``         2.37×         1.33×         1.78×         1.77×
``a06_tall``    2.00×         1.00×         2.00×         1.92×
``krumm``       1.43×         1.07×         1.33×         1.25×
==============  ============  ============  ============  ========

Und in die Gegenrichtung: auf glatt teilbaren Shapes, wo beide Baselines nichts
padden, ist die neue feste Config sogar leicht *schlechter* (0.95–1.00×) — genau
wie es das Arithmetic-Intensity-Argument verlangt. Beide Richtungen stimmen.

Warum 128/128/64 trotzdem eine vernünftige Wahl war
"""""""""""""""""""""""""""""""""""""""""""""""""""

Die Herleitung der A05-Config ist für sich genommen sauber. Quadratische Tiles
minimieren den Operanden-Traffic ``(M+N)·K`` bei gegebenem Akkumulator ``M·N``;
``128×128`` fp32 sind 16384 Register für den Akkumulator, also klar unter dem
Deckel von 32768 aus dem :ref:`Pruning <pruning>`; das Operanden-SMEM liegt mit
``(128·64 + 64·128)·2·2`` bei 64 KB und passt ins rund 100 KB große Opt-in; und
``8×8`` maximiert den L2-Reuse nach dem Grouping-Argument.

Der Fehler liegt nicht in dieser Kette, sondern in dem, was sie übergeht: sie
optimiert den eingeschwungenen Zustand und ignoriert die Quantisierung an den
Rändern. Genau die entscheidet, sobald die Shape kein Vielfaches der
Gruppenausdehnung ist — und dann kostet die große Gruppe mehr, als der bessere
Reuse einbringt.

Die neue Baseline
"""""""""""""""""

``BASELINE_CONFIGS`` in ``problems.py`` hält eine feste Config pro GPU: für die
GB10 ``64/256/64, 8×2``, für die 3070 ``64/128/64, 8×2``. Die Wahl ist auch ohne
Messung begründbar: der Akkumulator ist mit ``64·256 = 16384`` genau so groß wie
bei ``128×128``, kostet also dieselben Register — aber die Gruppenausdehnung
halbiert sich auf 512×512. Man tauscht etwa 25 % Arithmetic Intensity gegen
deutlich weniger Padding-Quantisierung, und über einen Shape-Mix zahlt sich das
aus.

``baselines_study.py`` leitet die Config reproduzierbar aus den Vollmessungen ab.
Ein Leave-one-out (die feste Config auf 15 Shapes wählen, auf der 16. bewerten)
liegt bei 89.5 % gegen 90.9 % des Oracle — die Wahl ist also robust und kein
Artefakt des Nachwissens.

Für Karten ohne Vollmessung gibt es ``baseline_probe.py``: es misst nur einen
kleinen Kandidatenpool (13 Configs, 208 Messungen) statt des vollen Sweeps, rund
13-mal billiger. Auf der GB10, wo sich das prüfen lässt, findet die Sonde eine
Config mit 89.6 % — praktisch der Leave-one-out-Wert. Auf der 3070 liefert
dieselbe Sonde 75.9 %.

Auswirkung auf die Aussagen des Projekts
""""""""""""""""""""""""""""""""""""""""

Der ehrliche Tuner-Gewinn auf der GB10 ist damit **1.12×**, nicht 1.27×. Der
frühere Cross-GPU-Hebel von 1.88× misst zu rund zwei Dritteln, wie schlecht eine
fremde Config passt — eine legitime und sogar wichtige Aussage, aber eine andere
als „Per-Shape-Tuning bringt 1.88×". Sauber getrennt lauten die beiden Fragen:

1. **Was kostet es, beim GPU-Wechsel nicht neu zu tunen?** Die A05-Config auf der
   3070 holt nur 39.1 % des Optimums, das Neu-Tunen bringt dort also 2.56×. Das
   rechtfertigt den GPU-Modell-Anteil im Cache-Key.
2. **Was bringt Per-Shape-Tuning gegen eine kompetent gewählte feste Config
   derselben Karte?** 1.12× auf der GB10, 1.32× auf der 3070.

Ein Gegenargument steht dem Tuner dabei zu: an die „kompetent gewählte feste
Config" kommt man nur über einen vollen Multi-Shape-Sweep auf der Zielkarte —
also genau über das, was der Tuner ersetzt. Der reale Wert des Tunings liegt
damit zwischen 1.12× (gegen die im Nachhinein beste feste Config) und 1.27×
(gegen eine plausibel von einer anderen Shape übernommene Config).
