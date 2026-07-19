.. _ranking:

Ranking: die Kostenmodelle
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Quelle: project/src/autotuner/search.py (rank, estimate_dram_bytes,
   estimate_grid, occupancy_util, estimate_blocks_per_sm)
   Auch dieses Kapitel soll detailliert werden -- drei Modelle, jeweils Idee,
   Formel, und was die Messung dazu gesagt hat.

Zweck des Rankings
""""""""""""""""""

.. Inhalt:
   - Klarstellen: das Ranking ist NICHT primaer zum Zeitsparen da. Der Compile
     kostet nur ~0.4 s, wir koennten alle 342 messen. Der eigentliche Zweck war die
     Forschungsfrage "zieht unser Modell die beste Config nach oben?".
   - Erst spaeter wird es zum Vorfilter fuer den praktischen Tuner.

Modell 1: Bandbreite (bw)
"""""""""""""""""""""""""

.. Inhalt:
   - Idee: FLOPs sind fuer alle Kandidaten gleich, also entscheidet der
     DRAM-Traffic. Groessere Gruppe = weniger Nachladen.
   - Formel aus estimate_dram_bytes: A einmal pro Gruppen-Spalte, B einmal pro
     Gruppen-Zeile, plus C. Codeausschnitt.
   - Der L2-bewusste Zweig: passt der Working-Set einer Gruppe ins L2, zaehlt nur
     das Kaltladen. Das ist der portable Umschalter zwischen den Karten.

Modell 2: bw mit Occupancy (bw_occ)
"""""""""""""""""""""""""""""""""""

.. Inhalt:
   - Motivation: Variante B mit grosser Gruppe hat nur 32 CTAs bei 48 SMs,
     kriegt im reinen bw-Modell aber dieselbe Vorhersage wie A.
   - Wave-Quantisierung ueber estimate_blocks_per_sm und occupancy_util.

Modell 3: Roofline
""""""""""""""""""

.. Inhalt:
   - max(memory_ms, compute_ms) als automatischer, hardwaregetriebener
     Regime-Selektor. compute_ms = padded-FLOPs / (Tensor-Peak * util).
   - Der einzige nicht auslesbare Wert ist tensor_flop_per_sm_cycle
     (Architektur-Schaetzung) -- ehrlich benennen, was das beeinflusst
     (nur den Umschaltpunkt, nicht die Reihenfolge im Regime).
   - Die Tie-Break-Geschichte: im compute-Regime haengt compute_ms nicht von
     m_l2/n_l2 ab, alle Configs mit gleicher Prim-Form bekommen denselben Score.
     Der alte Tie-Break -grid schob dann kleine Tiles nach oben. Fix: ueber den
     worst-case-Traffic aufloesen. Das ist eine lehrreiche Fehlersuche und sollte
     ausgeschrieben werden.

Bewertung der Modelle
"""""""""""""""""""""

.. Inhalt:
   - Ergebnistabelle (Spearman und Top-7-Ausbeute) fuer bw / v2 / roofline.
     Zahlen aus der Roadmap M4 bzw. neu aus analyze_tune.py rechnen.
   - Das Kernresultat als eigener Absatz: die Roofline korreliert am besten,
     ist aber der SCHLECHTERE Top-k-Vorfilter -- bessere Korrelation heisst nicht
     besserer Filter. v2 bleibt Default.
   - fig_ranking_models.png und fig_topk_curve.png einbinden.
