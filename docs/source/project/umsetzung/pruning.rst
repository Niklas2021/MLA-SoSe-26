Statisches Pruning
^^^^^^^^^^^^^^^^^^

.. Quelle: project/src/autotuner/search.py (prune, prune_reason, estimate_*)
   Das ist eines der Kapitel, das der Auftraggeber explizit detailliert haben will.
   Jeder Filter bekommt: Was er prueft, die Formel, woher die Hardware-Zahl kommt,
   was er bei A05 konkret wegwirft, und wie sicher/heuristisch er ist.

Warum überhaupt filtern
"""""""""""""""""""""""

.. Inhalt:
   - Ziel: alles wegwerfen, was ohne Kompilieren als unsinnig erkennbar ist.
   - Wichtig: prune gibt (kept, rejected) mit Grund zurueck -- nachvollziehbar,
     was warum wegfaellt. Codeausschnitt.
   - code_prune.png aus den Praesentations-Figuren passt hier.

Filter 1: MMA-Alignment
"""""""""""""""""""""""

.. Inhalt: Vielfache von 16 fuer die fp16-Tensor-Cores. Guard-Charakter -- unsere
   Kandidatenwerte erfuellen ihn alle, er schuetzt vor kuenftigen Knopfwerten.

Filter 2: Shared-Memory-Budget
""""""""""""""""""""""""""""""

.. Inhalt:
   - Die harte Schranke. Formel: (M_PRIM*K_PRIM + K_PRIM*N_PRIM) * 2 Byte * stages.
   - Woher das Budget kommt: MaxSharedMemoryPerBlockOptin - ReservedSharedMemory
     = 101376 - 1024 ~ 100 KB auf der GB10. Auf die A02-Seite zu den Device
     Properties verlinken.
   - Was er wegwirft: bei A05 126 der 486 Kandidaten (alle 256x256-Tiles).
   - Ehrlichkeit: wir wissen nicht sicher, ob cuTile das Opt-in-Limit nutzt oder
     beim 48-KB-Default bleibt. Deshalb ist buffer_stages/smem_limit parametrisiert
     und der eigentliche Schutz ist das try/except ums Kompilieren.

Filter 3: Akkumulator-Register
""""""""""""""""""""""""""""""

.. Inhalt:
   - M_PRIM*N_PRIM fp32 gegen reg_fraction * regs_per_block.
   - Was er wegwirft: 18 weitere Kandidaten bei A05.
   - Der interessante Teil: dieser Filter wird spaeter als "v2" nochmal separat
     mit reg_fraction=0.4 auf das Ranking angewendet, weil das Modell sonst
     ausgerechnet die Register-Fresser nach oben sortiert. Verweis auf Ranking.

Filter 4: Padding-Verschwendung
"""""""""""""""""""""""""""""""

.. Inhalt:
   - gepaddetes Volumen / Original-Volumen gegen max_padding (Default 8.0).
   - Greift bei A05 nicht (4096 ist glatt teilbar), dafuer bei krummen Shapes.
   - Hier den Bogen spannen: derselbe Quotient erklaert spaeter quantitativ,
     warum die A05-Default-Config als Baseline versagt (Vorwaertsverweis).

Was das Pruning NICHT kann
""""""""""""""""""""""""""

.. Inhalt:
   - 486 -> 342 ist weniger, als man hofft, und das ist strukturell: das
     pro-Block-SMEM haengt nur an den Prim-Groessen, nicht an m_l2/n_l2 oder der
     Variante. Diese beiden Achsen sind statisch gar nicht beschneidbar.
   - Die L2-Reuse-Regel aus der Vorlesung greift auf der GB10 nicht: der
     Working-Set der groessten Gruppe ist ~256 KB gegen 25 MB L2. Deshalb
     verschiebt sich die Entscheidung ueber m_l2/n_l2 komplett auf die Messung.
   - dedup_mn_symmetry als optionale, verlustbehaftete Reduktion (342 -> 186)
     erwaehnen und begruenden, warum sie nicht im Default-Pfad ist.
