Einsum-Abdeckung: Layout-Guard und Kanonisierung
================================================

.. Quellen: project/src/autotuner/layout.py, project/src/check_coverage.py,
   die Erweiterung von project/src/autotuner/kernels.py (matmul_flex_a)

Die stille Lücke
----------------

.. Inhalt:
   - Der Befund, mit dem das Kapitel aufmachen sollte: parse_einsum akzeptierte
     deutlich mehr, als die Kernel rechnen koennen, und nichts prueft das.
     "cmk,cnk->cmn" (B als NT) parst sauber, der Kernel liest dann das falsche
     Layout -- auffaellig erst beim allclose, ununterscheidbar von einem echten
     Rechenfehler. Andere Strings crashten am Shape-Unpack.
   - Tabelle vorher/nachher ueber alle getesteten Strings.

Zwei verschiedene Mechanismen
-----------------------------

.. Inhalt:
   - Die zentrale Unterscheidung, die das Kapitel traegt: Batch-Achsen umsortieren
     ist GRATIS (reines view: fehlende Achse ergaenzen, mehrere fusionieren).
     M/N/K umsortieren ist eine TRANSPOSITION -- kein Vertauschen von Indizes
     aendert, was physisch stride-1 liegt.
   - Warum bewusst view() statt reshape(): reshape wuerde bei nicht
     zusammenhaengenden Tensoren still kopieren, und eine unsichtbare Kopie mitten
     im Benchmark waere schlimmer als ein Fehler.
   - literalinclude Layout und plan_layout.

Der flex-Kernel
---------------

.. Inhalt:
   - matmul_flex_a mit drei Transponier-Flags als ct.Constant. Der Tile-Transpose
     ist derselbe Trick, den matmul_ring_a schon nutzt (ct.permute nach dem Laden)
     -- keine Kopie im Speicher.
   - Die bewusste Entscheidung, einen EIGENEN Kernel zu bauen statt Flags in
     matmul_variant_a: der ist auf beiden Karten erprobt, ein Fehler im neuen Pfad
     sollte ihn nicht mitreissen. Das ist ein Engineering-Argument, das man
     ausschreiben sollte.
   - Die offene Annahme und ihre Aufloesung: fuer eine Verzweigung ueber eine
     ct.Constant gab es in unseren 13 Kerneln keinen Praezedenzfall. cuTile loest
     sie beim Spezialisieren auf -- auf GB10 und 3070 verifiziert.

Abdeckungstabelle
-----------------

.. Inhalt:
   - Die COVERAGE-Liste aus problems.py als Tabelle: Einsum, was noetig ist,
     Status. Von 2 auf 10 laufende Familien.
   - Was weiterhin abgelehnt wird und warum (Batch-Dim nicht aussen).
   - check_coverage.py als reproduzierbarer Nachweis, lokal und mit --run.
     Programmausgabe per literalinclude aus results_dgx_v2/coverage_run.log.
