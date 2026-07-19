Messmethodik
============

Messaufbau
----------

.. Inhalt:
   - do_bench mit warmup/rep, Korrektheitspruefung gegen torch.einsum vor jeder
     Messung, TFLOPS auf der Original-Shape (nicht der gepaddeten).
   - Welche Skripte welche Artefakte erzeugen (Tabelle):
     tune.py -> tune_*.csv + study.log; autotune.py -> autotune_*.csv;
     check_coverage.py --run -> coverage_run.*; measure_order.py -> order_isolated.*;
     baseline_probe.py -> baseline_probe_*.*
   - Prinzip: messen auf der Karte, auswerten offline aus den CSVs.

Der Messrahmen-Effekt
---------------------

.. Inhalt:
   - Das ist der wichtigste methodische Befund und gehoert VOR die Ergebnisse,
     weil er bestimmt, welche Vergleiche zulaessig sind.
   - Symptom: der Hybrid lag bei 9 von 16 Shapes scheinbar UEBER der Vollmessung,
     teils um 4.6 % -- strukturell unmoeglich, weil die Vollmessung eine Obermenge
     misst.
   - Der entscheidende Test: dieselbe Config in beiden Laeufen nachschlagen.
     Ergebnis 101.4 % im Mittel, bis 4.6 % Abweichung. Gleiche GPU, gleicher Tag.
   - Ursache: Lastdauer. Hybrid ~22 Configs in ~7 s, Sweep 342 Configs in
     100-250 s. Die GB10 teilt sich als integrierter LPDDR-Chip Power und Kuehlung
     mit der CPU.
   - Gegenprobe, die die Deutung stuetzt: ueber alle 4104 gemeinsamen Config-Paare
     stimmen die zwei Sweeps auf 100.0 % ueberein -- die Sweep-Methodik ist
     reproduzierbar, verzerrt ist nur der Vergleich kurzer gegen langen Lauf.
   - Konsequenz als Regel formulieren: TFLOPS aus Laeufen unterschiedlicher Dauer
     sind nicht vergleichbar. Die belastbare Metrik ist die AUSWAHLGUETE -- die
     gewaehlte Config im Messrahmen des Sweeps gegen den Sweep-Besten.

Datenbasis und ihre Grenzen
---------------------------

.. Inhalt:
   - Tabelle: welcher Ordner, welche Karte, welcher Lauf, was er enthaelt, wie
     belastbar.
   - Die Einschraenkung beim ersten 3070-Sweep offen benennen: alle fuenf Shapes
     mit batch=1 liegen 3.3-4.8x zu niedrig, alles mit c>=2 stimmt auf +-13 %
     ueberein; im alten Lauf brauchten genau diese Shapes 9-14 s pro Config statt
     0.5-2 s. Die Ursache liess sich nachtraeglich nicht bestimmen (WSL-Umgebung).
   - Was daraus folgt: Aussagen, die auf dem alten 3070-Sweep beruhen (der
     Cross-GPU-Hebel 1.88x, die Randtreffer-Analyse), sind nicht belastbar und
     werden im 3070-Kapitel entsprechend gekennzeichnet.
