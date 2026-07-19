Mess-Harness, Tuner und Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Quellen: project/src/tune.py, project/src/autotune.py,
   project/src/autotuner/cache.py, project/src/analyze_tune.py

Vollmessung als Ground Truth
""""""""""""""""""""""""""""

.. Inhalt:
   - tune.py: enumeriert, prunt, kompiliert jeden Kandidaten im try/except, prueft
     gegen torch.einsum (allclose rtol=1e-2, atol=1e-1) und misst mit do_bench.
   - Wichtig: Compile-Fehler und inkorrekte Configs werden protokolliert, nicht
     verschwiegen. Die note-Spalte in der CSV.
   - do_bench-Settings (warmup=50, rep=200) und warum.
   - Die externe Referenz torch.einsum in fp16 pro Shape ("was, wenn man einfach
     die Library nimmt").

Die Shape-Menge
"""""""""""""""

.. Inhalt:
   - Tabelle der 16 Shapes aus problems.py mit Regime-Label. Warum gerade diese:
     quadratisch, rechteckig, klein-K, gross-K, unteilbar, viele Batches -- je
     einmal in beiden Familien. fig_regimes.png passt hier.

Praktischer Tuner und Cache
"""""""""""""""""""""""""""

.. Inhalt:
   - autotune.py: erst Cache, sonst suchen, dann cachen.
   - Der Cache-Key (einsum | shapes | GPU-Modell) und die Begruendung fuer das
     GPU-Modell im Key.
   - Amortisation: eine Kontraktion laeuft in ~8 ms, das Tuning kostet einmalig
     ein paar Sekunden. Warum sich das in der Praxis fast immer lohnt (feste
     Layer-Dimensionen, Millionen Aufrufe) und wann nicht (stark dynamische
     Shapes -> Bucketing).

Reproduzierbare Auswertung
""""""""""""""""""""""""""

.. Inhalt:
   - analyze_tune.py laeuft lokal aus den CSVs, ohne GPU. Das Prinzip
     "Messung auf dem Server, Auswertung offline" durchgehend erwaehnen -- es
     zieht sich durch alle spaeteren Skripte.
