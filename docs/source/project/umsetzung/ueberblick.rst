Architektur und Datenfluss
^^^^^^^^^^^^^^^^^^^^^^^^^^

Was kann ein Tuner an einer festen Kontraktion mit festen Shapes überhaupt
variieren?

Nehmen wir den GEMM `cmk,ckn->cmn`. Der Einsum besagt: `c` ist ein Batch (steht in beiden
Eingabematrizen und im Ergebnis), `m` und `n` spannen das Ausgabegitter auf, und über `k`
wird summiert. Das Ergebnis ist also ein `M×N`-Gitter pro Batch `c`, jeder Eintrag
eine Summe über `K`.

Auf der GPU wird diese Arbeit von **CTAs** erledigt — Thread-Blocks, die kleinste
Einheit, die ein Streaming Multiprocessor (SM) am Stück ausführt. Man zerlegt die
drei Achsen also in Kacheln und verteilt die Kacheln auf CTAs. Aber es gibt mehr als
eine Zerlegung und die Wahl hat zwei Dimensionen:

**1. Wie groß ist die Kachel?** Ein CTA rechnet eine
`M_PRIM × N_PRIM`-Kachel, indem es über `K_PRIM`-breite Streifen
der Eingaben `mma`-Instruktionen ausführt (`mma` = Matrix-Multiply-Accumulate, die
Kern-Instruktion der Tensor-Cores: multipliziert zwei kleine Matrizen und akkumuliert auf
das Ergebnis. Größere Kacheln heißen mehr Wiederverwendung pro geladenem Byte,
aber auch mehr Register- und Shared-Memory-Druck. Das ist der klassische
Tiling-Trade-off.

**2. In welcher Reihenfolge werden die Kacheln abgelaufen?** Jede Dimension der Kontraktion bekommt einen Ausführungstyp:

- **PAR** — parallel über die Block-ID verteilt
- **SEQ** — eine sequenzielle Schleife *innerhalb* eines CTA
- **PRIM** — die Achse, die in die `mma`-Kachel selbst eingeht

Diese Typen haben eine feste Ordnung: `PAR | SEQ | PRIM`, und `K` darf nie PAR
sein (man kann eine Reduktion nicht ohne weiteres über unabhängige CTAs verteilen).
Diese Regel ist in `Config` und `Optimizer` definiert:

.. code-block:: python
   :caption: project/src/autotuner/config.py:12-15

   class ExecType(Enum):
       SEQ = "seq"
       PAR = "par"
       PRIM = "prim"

.. code-block:: python
   :caption: project/src/autotuner/optimizer.py:113-122

           # 1) kein K darf PAR sein
           for i in range(n):
               if dim_types[i] == DimType.K and exec_types[i] == ExecType.PAR:
                   raise ValueError(f"Dim {i} ist K mit PAR - nicht erlaubt")

           # 2+3) Reihenfolge muss PAR -> SEQ -> PRIM sein
           order = {ExecType.PAR: 0, ExecType.SEQ: 1, ExecType.PRIM: 2}
           for i in range(n - 1):
               if order[exec_types[i]] > order[exec_types[i + 1]]:
                   raise ValueError(f"falsche Reihenfolge bei Dim {i}: {exec_types[i]} vor {exec_types[i+1]}")

Die L2-Gruppe
""""""""""""""

Der wichtigste Reihenfolge-Effekt ist der L2-Cache-Reuse. Er kommt vom
Timing: 2 Ausgabeblöcke in derselben Zeile brauchen denselben A-Streifen. Laufen
sie kurz nacheinander, findet der zweite die Daten noch im L2 vor, liegen viele
andere Blöcke dazwischen, sind sie längst verdrängt.

Deshalb splitten wir `M` in drei Ebenen — `m_l2_outer`, `m_l2`, `m_prim` — und `N`
genauso. Die mittlere Ebene bildet die **L2-Gruppe**: `M_L2 × N_L2` benachbarte
Ausgabeblöcke, die zeitlich direkt hintereinander laufen. Innerhalb der Gruppe wird
jeder A-Streifen `N_L2`-mal und jeder B-Streifen `M_L2`-mal gebraucht, und weil das
dicht beieinander passiert, kommen die Wiederholungen aus dem L2 statt aus dem DRAM.

Es gibt 2 Varianten:

- **Variante A:** `m_l2/n_l2` sind PAR. Man verteilt die Gruppe über die Block-IDs
  und dekodiert sie per Swizzle aus `bid(0)`. Viele CTAs.
- **Variante B:** `m_l2/n_l2` sind SEQ-Loops im CTA. Ein CTA arbeitet die ganze
  Gruppe selbst ab. Weniger CTAs.

Beide rechnen dasselbe, belasten aber Occupancy und Scheduler unterschiedlich
(Occupancy = wie viele CTAs gleichzeitig auf den SMs Platz finden; Variante B hat
weniger CTAs und füllt die GPU schlechter, wenn das Grid ohnehin klein ist).

Damit ist der Spielraum komplett: Kachelgrößen (`m_prim, n_prim, k_prim`),
L2-Gruppe (`m_l2, n_l2`) und Variante (A/B) sind frei wählbar, und jede Kombination
rechnet dasselbe Ergebnis, nur unterschiedlich schnell. Diese sechs Knöpfe bilden
den Suchraum des Tuners.

Wie eine Config im Code aussieht
"""""""""""""""""""""""""""""""""

Konkret ist eine Config vier parallele Listen (eine pro Dimension): `dim_types`
(M/N/K/C), `exec_types` (PAR/SEQ/PRIM), `dim_sizes` und die `strides` pro Tensor.
`generate_config` baut zuerst die naive Ausgangs-Config, in der alle Achsen SEQ
sind:

.. code-block:: python
   :caption: project/src/autotuner/generate.py:46-47

       # alles auf SEQ setzen
       exec_types = [ExecType.SEQ] * len(all_dims)

Der `Optimizer` formt daraus die eigentliche Config. `split_dim` zerlegt eine Achse
in outer/inner und teilt die Strides korrekt auf; `make_executable`
markiert die jeweils letzte M-, N- und K-Achse als PRIM, setzt den Rest (K→SEQ,
sonst PAR) und sortiert alles in die `PAR | SEQ | PRIM`-Ordnung:

.. code-block:: python
   :caption: project/src/autotuner/optimizer.py:81-103

           # letzte M, N, K jeweils als PRIM markieren
           for needed_type in [DimType.M, DimType.N, DimType.K]:
               for i in reversed(range(n)):
                   if cfg.dim_types[i] == needed_type and exec_types[i] is None:
                       exec_types[i] = ExecType.PRIM
                       break

           # Rest: K -> SEQ, sonst PAR
           for i in range(n):
               if exec_types[i] is not None:
                   continue
               if cfg.dim_types[i] == DimType.K:
                   exec_types[i] = ExecType.SEQ
               else:
                   exec_types[i] = ExecType.PAR

           cfg.exec_types = exec_types

           # Reihenfolge PAR | SEQ | PRIM
           par_ids = [i for i in range(n) if exec_types[i] == ExecType.PAR]
           seq_ids = [i for i in range(n) if exec_types[i] == ExecType.SEQ]
           prim_ids = [i for i in range(n) if exec_types[i] == ExecType.PRIM]
           self.permute_dims(par_ids + seq_ids + prim_ids)

Das ganze Modul `search.py` ist reines Python ohne cuTile-Import — so kann
man Enumerieren, Prunen und Ranken komplett ohne GPU lokal testen.

Die Pipeline
^^^^^^^^^^^^

Der Tuner selbst ist eine feste Kette aus 6 Stufen:

1. **`parse_einsum` / `generate_config`** — aus Einsum + Shapes die Dimensionstypen,
   Größen und Strides ableiten (die „Ground Truth", für alle Kandidaten gleich)
2. **`enumerate_candidates`** — den Suchraum aufspannen (die 486 Kandidaten aus dem
   Freiheitsgrad oben)
3. **`prune`** — statisch filtern, ohne zu kompilieren (-> 342 Kandidaten)
4. **`rank`** — ein Kostenmodell sortiert die übrig gebliebenen Kandidaten
5. **`tune.py`** — die Top-k kompilieren, gegen `torch.einsum` prüfen,
   mit `do_bench` messen
6. **`cache.py` / `autotune.py`** — die beste Config cachen

Die ersten 4 Stufen sind reines Python ohne GPU und lassen sich lokal testen, erst ab Stufe 5 wird kompiliert und gemessen.
Alles "billige" passiert also vorab, GPU-Zeit fließt nur in Kandidaten, die den
Vorfilter überlebt haben.

Stufe 1 haben wir mit `generate_config` und dem `Optimizer` gerade schon gesehen.
Bleiben die Stufen 2 bis 5.
