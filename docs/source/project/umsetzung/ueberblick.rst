Architektur und Datenfluss
==========================

.. Inhalt:
   - Modul-Landkarte als Tabelle: Datei -> Zustaendigkeit -> laeuft ohne GPU?
     (config/optimizer/generate = aus A05 uebernommen, unveraendert;
      einsum_parser, layout, search, strategies = reines Python;
      kernels = braucht cuTile; tune/autotune/... = Skripte)
   - Die zentrale Design-Entscheidung erklaeren: alles ausser kernels.py ist
     GPU-frei. Deshalb laeuft der komplette Suchraum-Teil lokal testbar, und die
     Selbsttests in jedem Modul laufen ohne Karte.
   - Zweite Design-Entscheidung: Candidate als Datencontainer zwischen Suche und
     Kernel-Start. Was drin steht (Config, Knoepfe, Original- und Padded-Groessen,
     Layout-Plan) und warum.
   - Datenfluss-Absatz: einsum + shapes -> parse_einsum -> plan_layout ->
     enumerate_candidates -> prune -> rank -> Strategie -> run_candidate -> Cache.

Übernommene Bausteine aus Assignment 05
---------------------------------------

.. Inhalt:
   - Config, Optimizer, generate_config wurden unveraendert uebernommen.
     Kurz sagen, was sie tun, und auf die A05-Seiten verlinken statt zu wiederholen.
   - Wichtig fuer die Bewertung: split_dim/permute_dims/make_executable sind die
     Werkzeuge, mit denen der Enumerator die Configs baut -- der Tuner erfindet
     keine neue Config-Semantik.
   - Anmerkung: fuse_dims blieb ungenutzt (die Batch-Fusion passiert auf
     Torch-Ebene per view). Ehrlich erwaehnen.
