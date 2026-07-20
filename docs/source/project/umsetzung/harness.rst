Mess-Harness, Tuner und Cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ab der Messung teilt sich der Code in zwei Wege. ``tune.py`` misst jeden
überlebenden Kandidaten einer Shape durch und dient als Ground Truth für die
Modell-Auswertung. ``autotune.py`` ist der praktische Tuner, der für eine Anfrage
nur die Top-k misst und das Ergebnis cacht. Beide benutzen dieselbe
Kernel-Instanziierung aus dem vorigen Kapitel.

Vollmessung als Ground Truth
""""""""""""""""""""""""""""

``tune.py`` enumeriert und prunt eine Shape und misst dann alle übrig gebliebenen
Configs. Jeder Kandidat wird einzeln kompiliert, gegen ``torch.einsum`` geprüft
und nur bei bestandener Prüfung mit ``do_bench`` gemessen:

.. literalinclude:: ../../project/src/tune.py
   :language: python
   :caption: tune.py — Messschleife über alle überlebenden Configs
   :start-at: for cand in kept:
   :end-at: results[sig(cand)] = row
   :dedent:

Das ganze ``try/except`` ist Absicht. Eine Config, die beim Kompilieren wirft
(zum Beispiel weil das Pruning sie fälschlich durchgelassen hat), bekommt eine
``note`` wie ``failed: OutOfMemoryError`` und ``ok=False``, statt den Lauf
abzubrechen. Eine Config, die kompiliert, aber am ``allclose``
(``rtol=1e-2``, ``atol=1e-1``) scheitert, wird als ``incorrect`` markiert. Beide
Fälle stehen so in der CSV — ein Fehler wird protokolliert, nicht verschwiegen.

``do_bench`` läuft mit ``warmup=50`` und ``rep=200`` (Millisekunden-Budgets):
50 ms werden verworfen, bis Takt und Caches eingeschwungen sind, danach wird über
200 ms gemittelt. Die TFLOPS rechnen wir aus der Zeit und den FLOPs der
**Original**-Shape, nicht der gepaddeten.

Zusätzlich misst ``tune.py`` pro Shape einmal ``torch.einsum`` in fp16 als externe
Referenz. So sieht man, was man bekommt, wenn man statt zu tunen einfach die
Library nimmt — der Vergleich taucht in der Evaluation wieder auf.

Die Shape-Menge
"""""""""""""""

Getunt wird nicht eine Shape, sondern ein Satz aus acht Regimen, der die
relevanten Achsen des Problems abdeckt: quadratisch, rechteckig in beide
Richtungen, kleines und großes K, unteilbare Größen und viele Batches.

.. image:: ../../project/praesentation/figures/fig_regimes.png
   :alt: Die acht Shape-Regime mit ihren Dimensionen
   :width: 95%

======================  =====  =====  =====  =====  ================================
Regime                  C      M      N      K      testet
======================  =====  =====  =====  =====  ================================
``a05`` (Referenz)      4      4096   4096   4096   Heimvorteil, dafür handgetunt
``square_1b``           1      4096   4096   4096   dasselbe ohne Batch
``tall``                1      8192   1024   4096   viele Zeilen, wenige Spalten
``wide``                1      1024   8192   4096   viele Spalten, wenige Zeilen
``small_k``             1      4096   4096   512    kleines K, eher bandbreitennah
``large_k``             1      1024   1024   8192   großes K, eher compute-nah
``krumm``               2      1500   3000   1000   unteilbar, Padding-Pfad
``batch16``             16     1024   1024   1024   viele kleine Batches
======================  =====  =====  =====  =====  ================================

Jedes dieser Regime existiert zweimal: einmal als GEMM (A05-Familie) und einmal
in Ring-Form (A06-Familie), macht 16 Shapes insgesamt. So sieht man für jede
Struktur, wie stark die optimale Config vom Regime abhängt.

Praktischer Tuner und Cache
"""""""""""""""""""""""""""

``autotune.py`` sieht für eine Anfrage zuerst im Cache nach, sucht nur bei einem
Fehltreffer und legt das Ergebnis danach ab. Der Cache-Key besteht aus drei
Teilen:

.. literalinclude:: ../../project/src/autotuner/cache.py
   :language: python
   :caption: cache.py — der Cache-Key
   :pyobject: make_key

Das GPU-Modell gehört in den Key, weil dieselbe Config auf einer anderen Karte
anders abschneidet — eine auf der GB10 getunte Config ist für die 3070 wertlos.
Einsum und Shapes stehen ohnehin drin, weil sich mit ihnen die ganze
Klassifikation und damit die beste Kachelung ändert.

Ob sich das Tuning lohnt, ist eine Frage der Amortisation. Die A05-Referenz läuft
in rund 8 ms pro Aufruf; das Tuning kostet einmalig ein paar Sekunden (Top-7 etwa
drei Sekunden, ein voller Sweep gut zweieinhalb Minuten). Sobald eine Kontraktion
mit festen Dimensionen millionenfach aufgerufen wird — der Normalfall bei festen
Layer-Größen in einem Netz — ist der einmalige Tuning-Aufwand nach wenigen
Aufrufen wieder drin. Nur bei stark wechselnden Shapes trägt der Cache nicht, dort
müsste man auf feste Größen bucketen.

Reproduzierbare Auswertung
""""""""""""""""""""""""""

``tune.py`` schreibt nur CSVs. Die eigentliche Auswertung — Modell gegen Messung,
Tuner-Modus, Ranking-Vergleich — macht ``analyze_tune.py`` offline aus genau
diesen CSVs, ohne GPU:

.. literalinclude:: ../../project/src/analyze_tune.py
   :language: python
   :caption: analyze_tune.py — Kopf
   :end-at: Tuner-Modus (Top-k)

Dieses Prinzip — auf der Karte nur messen, alles Weitere offline aus den
Ergebnisdateien — zieht sich durch alle späteren Skripte. Es hält die
GPU-Belegung kurz und macht jede Zahl in dieser Dokumentation ohne Zugriff auf
die Hardware nachvollziehbar.
