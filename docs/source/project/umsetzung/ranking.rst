.. _ranking:

Ranking: die Kostenmodelle
^^^^^^^^^^^^^^^^^^^^^^^^^^

Zweck des Rankings
"""""""""""""""""""

Nach dem Prunen bleiben 342 Kandidaten. Das ist wenig genug, dass man auf der GB10
alle messen könnte — ein Compile kostet etwa 0.4 s. Das Ranking ist deshalb nicht
in erster Linie dazu da, Zeit zu sparen. Der eigentliche Zweck war zunächst die
Frage: zieht unser Kostenmodell die tatsächlich beste Config nach oben? Erst in einem zweiten Schritt wird das Ranking zum
Vorfilter für den praktischen Tuner, dort werden dann nur die Top-k gemessen statt
aller überlebenden Kandidaten.

``rank`` schätzt dazu pro Kandidat eine Laufzeit und sortiert aufsteigend. Es gibt
drei Modelle mit wachsender Komplexität: ``bw`` (reine Bandbreite), ``bw_occ``
(Bandbreite mit Occupancy-Strafe) und ``roofline`` (Memory- gegen Compute-Zeit).
Alle drei laufen über dieselbe Funktion und denselben Satz an Zwischenwerten:

.. code-block:: python
   :caption: project/src/autotuner/search.py — rank()

   def rank(candidates, dev, batch=1, model="bw"):
       bw = dev.peak_dram_bandwidth()
       peak_flops = dev.peak_tensor_flops()
       ranked = []
       for cand in candidates:
           gb = batch * cand.par_batch_extra
           grid = estimate_grid(cand) * gb
           occ = occupancy_factor(grid, dev)
           util = occupancy_util(cand, dev, gb)
           memory_ms = estimate_dram_bytes(cand) * gb / bw * 1e3
           memory_ms_l2 = estimate_dram_bytes(cand, dev) * gb / bw * 1e3
           padded_flops = 2 * cand.padded_m * cand.padded_n * cand.padded_k * gb * cand.seq_batch
           compute_ms = padded_flops / (peak_flops * util) * 1e3 if util > 0 else float("inf")
           roof_ms = max(memory_ms_l2, compute_ms)
           ranked.append((cand, {"grid": grid, "occupancy": occ, "util": util,
                                 "est_ms": memory_ms, "est_ms_occ": memory_ms / occ,
                                 "compute_ms": compute_ms, "roof_ms": roof_ms,
                                 "bound": "compute" if compute_ms >= memory_ms_l2 else "memory"}))
       ...

Jeder Kandidat trägt also ``est_ms`` (Bandbreiten-Schätzung), ``est_ms_occ``
(Bandbreite geteilt durch Occupancy), ``compute_ms`` und ``roof_ms`` gleichzeitig
mit sich — welches Feld am Ende sortiert, entscheidet nur der ``model``-Parameter.
So bleiben die drei Modelle direkt vergleichbar, ohne dass man dieselben Kandidaten
mehrmals durchrechnen muss.

Modell 1: Bandbreite (bw)
""""""""""""""""""""""""""

Die Grundidee: die FLOPs sind für alle Kandidaten identisch (dieselbe Kontraktion,
nur unterschiedlich gekachelt), also entscheidet allein der DRAM-Traffic. Eine
größere L2-Gruppe bedeutet weniger Nachladen von A und B, also weniger Traffic und
eine niedrigere geschätzte Zeit.

``estimate_dram_bytes`` rechnet das im Worst Case durch — kein L2-Treffer über
Gruppengrenzen hinweg. A wird einmal pro Gruppen-Spalte geladen, B einmal pro
Gruppen-Zeile, C einmal geschrieben:

.. code-block:: python
   :caption: project/src/autotuner/search.py — estimate_dram_bytes()

   def estimate_dram_bytes(cand, dev=None, dtype_bytes=2):
       # Worst case (kein L2): A einmal pro Gruppen-Spalte, B einmal pro Gruppen-Zeile.
       group_cols = ceildiv(cand.padded_n, cand.n_l2 * cand.n_prim)
       group_rows = ceildiv(cand.padded_m, cand.m_l2 * cand.m_prim)
       a_bytes = cand.padded_m * cand.padded_k * dtype_bytes * group_cols
       b_bytes = cand.padded_k * cand.padded_n * dtype_bytes * group_rows
       c_bytes = cand.padded_m * cand.padded_n * dtype_bytes
       worst = a_bytes + b_bytes + c_bytes
       if dev is None:
           return worst
       cold = (cand.padded_m * cand.padded_k + cand.padded_k * cand.padded_n +
               cand.padded_m * cand.padded_n) * dtype_bytes
       group_ws = (cand.m_l2 * cand.m_prim + cand.n_l2 * cand.n_prim) * cand.padded_k * dtype_bytes
       return cold if group_ws <= dev.l2_cache else worst

Ohne ``dev`` liefert die Funktion konsequent den Worst Case — das ist ``est_ms``,
die Grundlage von Modell ``bw``. Mit ``dev`` schaltet sie um: passt der Working-Set
einer Swizzle-Gruppe (``(m_l2·m_prim + n_l2·n_prim) · padded_k · dtype_bytes``) ins
L2 der jeweiligen Karte, zählt nur noch das einmalige Kaltladen aller drei Tensoren.
Dieser zweite Zweig ist der portable Umschalter zwischen den Karten: auf einer GPU
mit großem L2 (wie der GB10) fällt der Traffic praktisch auf den Kaltstart-Wert,
auf einer GPU mit kleinem L2 bleibt der Worst Case bestimmend, und die
Gruppengröße zählt wieder.

Kandidaten mit gleichem Traffic werden nach Grid-Größe absteigend geordnet, weil
mehr Blöcke die SMs tendenziell besser füllen:

.. code-block:: python
   :caption: project/src/autotuner/search.py:301-302

           key = "est_ms_occ" if model == "bw_occ" else "est_ms"
           ranked.sort(key=lambda x: (x[1][key], -x[1]["grid"]))

Modell 2: bw mit Occupancy (bw_occ)
""""""""""""""""""""""""""""""""""""

Reine Bandbreite übersieht einen Fall: Variante B mit einer großen L2-Gruppe kann
auf ein sehr kleines Grid kommen. 

``bw_occ`` korrigiert das, indem es die geschätzte Zeit durch die Occupancy teilt
(``est_ms_occ = memory_ms / occ``). Die Occupancy-Schätzung selbst
kommt in zwei Stufen: wie viele Blöcke pro SM überhaupt Platz finden
(begrenzt durch Register- und SMEM-Budget) und wie gut das resultierende Grid
die verfügbaren Wellen füllt:

.. code-block:: python
   :caption: project/src/autotuner/search.py — estimate_blocks_per_sm() / occupancy_util()

   def estimate_blocks_per_sm(cand, dev, reg_fraction=DEFAULT_REG_FRACTION):
       # grobe Occupancy-Schaetzung: wie viele Bloecke passen pro SM. Akku (M_PRIM*N_PRIM
       # fp32) ist der Haupt-Registerfresser, das Operanden-SMEM der zweite Deckel.
       acc = estimate_acc_registers(cand)
       reg_blocks = int((dev.regs_per_block * reg_fraction) // acc) if acc else 1
       smem = estimate_smem_bytes(cand, DEFAULT_BUFFER_STAGES)
       smem_blocks = int(dev.smem_per_sm // smem) if smem else 1
       return max(1, min(reg_blocks, smem_blocks))


   def occupancy_util(cand, dev, batch=1):
       # Wave-Quantisierung: Anteil der SM-Kapazitaet, der wirklich gefuellt wird. 1.0 =
       # volle Waves; <1 bei zu wenig Bloecken (Variante B) oder schlechtem Tail-Wave.
       grid = estimate_grid(cand) * batch
       capacity = dev.number_sm * estimate_blocks_per_sm(cand, dev)
       if grid <= 0 or capacity <= 0:
           return 0.0
       waves = math.ceil(grid / capacity)
       return grid / (waves * capacity)

``estimate_blocks_per_sm`` schaut, wie viele Blöcke einer Config gleichzeitig auf einen SM passen. Dafür nimmt sie dieselben zwei Werte wie beim Pruning 
– SMEM-Bedarf und Register-Bedarf des Akkumulators – nur diesmal nicht als Ja/Nein-Schwelle, sondern um eine Zahl auszurechnen: die kleinere der beiden
 möglichen Block-Zahlen gewinnt, weil die knappere Ressource zuerst ausgeht.

Modell 3: Roofline
""""""""""""""""""

Das dritte Modell wägt Memory- und Compute-Zeit gegeneinander ab, statt nur die
Speicherseite zu betrachten:

.. code-block:: python
   :caption: project/src/autotuner/search.py — rank(), roofline-Zweig

           padded_flops = 2 * cand.padded_m * cand.padded_n * cand.padded_k * gb * cand.seq_batch
           compute_ms = padded_flops / (peak_flops * util) * 1e3 if util > 0 else float("inf")
           roof_ms = max(memory_ms_l2, compute_ms)
           ...
       if model == "roofline":
           # Tie-Break: worst-case-Traffic (L2-Reuse), nicht grid
           ranked.sort(key=lambda x: (x[1]["roof_ms"], x[1]["est_ms"]))

``roof_ms = max(memory_ms_l2, compute_ms)`` ist der klassische Roofline-Ansatz:
je nachdem, welche Seite größer ist, entscheidet Speicherbandbreite oder
Rechenleistung über die Laufzeit. 

Der einzige Wert in dieser Rechnung, der sich nicht direkt aus Config-Größen
ableiten lässt, ist ``peak_tensor_flops`` — eine Architektur-Schätzung der
Tensor-Core-Spitzenleistung. Das beeinflusst nur, an welcher Traffic-Schwelle ein
Kandidat vom Memory- ins Compute-Regime kippt, nicht die Reihenfolge der
Kandidaten innerhalb eines Regimes — dort zählt weiterhin ``memory_ms_l2`` bzw.
``compute_ms`` unverändert.

