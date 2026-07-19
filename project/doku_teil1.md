# cuTile Auto-Tuner — Problem, Idee und Umsetzung

Wie findet man für eine Tensor-Kontraktion automatisch eine schnelle cuTile-Config?
Darum ging es in unserem Projekt und dieser Post erklärt, welcher Spielraum
beim Ausführen einer Kontraktion überhaupt existiert, wie daraus ein Suchraum wird
und wie der Tuner ihn durcharbeitet. Die Messergebnisse (Ranking-Modelle, Vergleich
gegen torch, Cross-GPU) stehen in Teil 2.

---

## Warum ein Tuner?

In Assignment 05 und 06 haben wir zwei Kontraktionen von Hand getunt: den batched
Matmul `cmk,ckn->cmn` und die Tensor-Ring-Kontraktion `acspx,bspy->abcyx`.  Der Kernel-Code war dabei nie das Problem. Die Rechenvorschrift ist in beiden
Fällen trivial — 2 fp16-Tensoren rein, über die gemeinsame Achse summieren,
fp32 akkumulieren. Das Performance steckt woanders: in der Config, also darin, wie
man die Arbeit in Kacheln zerlegt und in welcher Reihenfolge man die Kacheln
abarbeitet. 

Diese Herleitung zweimal per Hand zu machen hat gereicht, um zu sehen, dass sie
nicht skaliert. Die optimale Config hängt an der Shape und an der GPU — Cache-Größe,
Registerbudget, Zahl der SMs — und müsste für jede neue Kombination neu gefunden
werden. Genau das automatisiert der Tuner: er bekommt einen Einsum-String und
Shapes und liefert eine gute cuTile-Config, per Messung bestätigt.

Was er nicht ist: ein allgemeiner Tensor-Compiler. Er deckt zwei Struktur-Familien
ab (GEMM-artig wie A05, Ring wie A06) und tunt die Configs, nicht den Kernel-Code.

---

## Der Freiheitsgrad

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

```./MLA-SoSe-26/project/src/autotuner/config.py:12-15
class ExecType(Enum):
    SEQ = "seq"
    PAR = "par"
    PRIM = "prim"
```

```./MLA-SoSe-26/project/src/autotuner/optimizer.py:113-122
        # 1) kein K darf PAR sein
        for i in range(n):
            if dim_types[i] == DimType.K and exec_types[i] == ExecType.PAR:
                raise ValueError(f"Dim {i} ist K mit PAR - nicht erlaubt")

        # 2+3) Reihenfolge muss PAR -> SEQ -> PRIM sein
        order = {ExecType.PAR: 0, ExecType.SEQ: 1, ExecType.PRIM: 2}
        for i in range(n - 1):
            if order[exec_types[i]] > order[exec_types[i + 1]]:
                raise ValueError(f"falsche Reihenfolge bei Dim {i}: {exec_types[i]} vor {exec_types[i+1]}")
```

### Die L2-Gruppe

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

### Wie eine Config im Code aussieht

Konkret ist eine Config vier parallele Listen (eine pro Dimension): `dim_types`
(M/N/K/C), `exec_types` (PAR/SEQ/PRIM), `dim_sizes` und die `strides` pro Tensor.
`generate_config` baut zuerst die naive Ausgangs-Config, in der alle Achsen SEQ
sind:

```./MLA-SoSe-26/project/src/autotuner/generate.py:46-47
    # alles auf SEQ setzen
    exec_types = [ExecType.SEQ] * len(all_dims)
```

Der `Optimizer` formt daraus die eigentliche Config. `split_dim` zerlegt eine Achse
in outer/inner und teilt die Strides korrekt auf; `make_executable`
markiert die jeweils letzte M-, N- und K-Achse als PRIM, setzt den Rest (K→SEQ,
sonst PAR) und sortiert alles in die `PAR | SEQ | PRIM`-Ordnung:

```./MLA-SoSe-26/project/src/autotuner/optimizer.py:81-103
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
```

Das ganze Modul `search.py` ist reines Python ohne cuTile-Import — so kann
man Enumerieren, Prunen und Ranken komplett ohne GPU lokal testen.

---

## Die Pipeline

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
Bleiben die Stufen 2 bis 5: Messung, Modell-Auswertung und Caching sind Teil 2.

---

## Stufe 2 — Der Suchraum

Für jeden der 6 Knöpfe gibt es eine Reihe hardware-sinnvoller Werte:

```./MLA-SoSe-26/project/src/autotuner/search.py:16-22
# die Knoepfe (aus dem Pitch)
M_PRIM_CHOICES = [64, 128, 256]
N_PRIM_CHOICES = [64, 128, 256]
K_PRIM_CHOICES = [32, 64, 128]
M_L2_CHOICES = [2, 4, 8]
N_L2_CHOICES = [2, 4, 8]
VARIANT_CHOICES = ["A", "B"]   # A = m_l2/n_l2 als PAR (swizzle), B = als SEQ-Loops
```

Macht `3 · 3 · 3 · 3 · 3 · 2 = 486` Kombinationen — mehr als die 81 aus dem
ursprünglichen Pitch. Die 81 zählten nur die Tile-Kombinatorik und ließen zwei
Achsen weg: das asymmetrische `M_L2 ≠ N_L2` und die zweite Ausführungsvariante.
Beide gehören zum Freiheitsgrad, also zählen sie mit.

`enumerate_candidates` iteriert über das Kreuzprodukt und baut für jede
Kombination via `build_one_config` einen `Candidate`. Kombinationen, die
`Optimizer.verify()` nicht bestehen, fallen mit einem `except` raus — noch ohne
jedes Pruning:

```./MLA-SoSe-26/project/src/autotuner/search.py:150-155
                            try:
                                candidates.append(build_one_config(
                                    einsum_props, variant,
                                    m_prim, n_prim, k_prim, m_l2, n_l2))
                            except (ValueError, NotImplementedError):
                                skipped += 1
```

### Die Hand-Config als Sanity-Check

Ein Suchraum taugt nur, wenn die gesuchte Lösung überhaupt darin liegt. Wir kennen
eine gute Lösung: die handoptimierte A05-Config (128/128/64, 8×8, Variante A). Der
`__main__`-Block in `search.py` prüft deshalb, dass genau diese Config im
enumerierten Set auftaucht und auch das Pruning übersteht. Würde der Tuner sie
unterwegs verlieren, könnte er sie auch nie finden.

### Krumme Shapes

`split_dim` verlangt exakte Teilbarkeit (`outer · inner` muss die alte Größe
ergeben), sonst wirft es. Eine Shape wie `M = 1500` geht also nicht direkt. Statt
solche Fälle abzulehnen, padden wir auf die nächste teilbare Größe hoch, der
Überhang wird später im Kernel über `PaddingMode.ZERO` genullt:

```./MLA-SoSe-26/project/src/autotuner/search.py:76-84
    # split_dim will exakte Teilbarkeit, also runden wir krumme Groessen hoch.
    # dim_sizes sind damit gepaddet, der Ueberhang wird im Kernel genullt.
    m_l2_outer = ceildiv(einsum_props.orig_m, m_prim * m_l2)
    n_l2_outer = ceildiv(einsum_props.orig_n, n_prim * n_l2)
    k_outer = ceildiv(einsum_props.orig_k, k_prim)

    padded_m = m_l2_outer * m_l2 * m_prim
    padded_n = n_l2_outer * n_l2 * n_prim
    padded_k = k_outer * k_prim
```

Die `dim_sizes` der Config sind damit die gepaddeten Größen — die TFLOPS rechnen
wir aber konsequent auf der Original-Shape, sonst würde man sich die Padding-Arbeit
als Leistung schönrechnen.

---

## Stufe 3 — Static Pruning

Bevor ein einziger Kernel kompiliert wird, wirft `prune` alles raus, was sich als unsinnig erkennen lässt. Es gibt 4 Filter:

```./MLA-SoSe-26/project/src/autotuner/search.py:184-193
def prune_reason(cand, dev, buffer_stages, reg_fraction, max_padding, smem_limit):
    if cand.m_prim % MMA_ALIGN or cand.n_prim % MMA_ALIGN or cand.k_prim % MMA_ALIGN:
        return "mma_align"
    if estimate_smem_bytes(cand, buffer_stages) > smem_limit:
        return "smem_exceeded"
    if estimate_acc_registers(cand) > dev.regs_per_block * reg_fraction:
        return "acc_registers"
    if padding_ratio(cand) > max_padding:
        return "padding_waste"
    return None
```

1. **MMA-Teilbarkeit:** die Prim-Größen müssen Vielfache von 16 sein, sonst passt
   die fp16-Tensor-Core-Kachel nicht. Im Standardraum erfüllen das alle Werte; der
   Filter fängt nur handgestrickte Suchräume ab.
2. **SMEM-Budget:** die beiden fp16-Operand-Tiles mal Double-Buffering, müssen ins
   nutzbare Shared Memory passen — auf der GB10 rund 100 KB. Das ist der Filter,
   der tatsächlich aussortiert.
   ```./MLA-SoSe-26/project/src/autotuner/search.py:167-171
   def estimate_smem_bytes(cand, buffer_stages):
       # die beiden fp16-Operand-Tiles mal Stages. Akku liegt in Registern.
       a_tile = cand.m_prim * cand.k_prim
       b_tile = cand.k_prim * cand.n_prim
       return (a_tile + b_tile) * 2 * buffer_stages
   ```
3. **Akku-Register:** der Akkumulator braucht `M_PRIM · N_PRIM` fp32-Werte in
   Registern. Mehr als die halbe Registerdatei (`65536 · 0.5 = 32768`) lassen wir
   nicht zu — das trifft vor allem die `256×256`-Kacheln.
4. **Padding:** wächst das gepaddete Volumen auf mehr als das Achtfache des
   Originals, fliegt der Kandidat raus.

Für die A05-Referenz (`4096³`) ergibt das 486 → 342: 126 Kandidaten fallen wegen
SMEM, 18 wegen der Akku-Register. MMA- und Padding-Filter greifen hier gar nicht,
weil 4096 glatt durch alle Knöpfe teilbar ist.

Interessanter ist, was das Pruning nicht kann. Alle vier Filter hängen
ausschließlich an den Prim-Größen — weder `m_l2/n_l2` noch die Variante tauchen in
ihnen auf. Die 2 Achsen, die den L2-Reuse steuern, kann statisches Pruning also
gar nicht anfassen. Man könnte hoffen, sie über die übliche Cache-Regel
einzuschränken („Gruppen-Working-Set muss ins L2 passen"), aber auf der GB10 mit
ihren 25 MB L2 passt selbst die größte überlebende Gruppe locker hinein. Die
Entscheidung über Gruppengröße und Variante bleibt damit komplett der Messung
überlassen,nicht weil unser Filter zu schwach wäre, sondern weil die Hardware an
dieser Stelle nichts verbietet.

Das Pruning ist eine Heuristik, kein Beweis: `buffer_stages`, `smem_limit`,
`reg_fraction` und `max_padding` sind Parameter mit optimistischen Defaults. Fällt
eine Config fälschlich durch, fängt sie das `try/except` um das Kompilieren im
Mess-Harness — sie scheitert dort sauber, statt still falsch zu rechnen.

---

## Stufe 4 — Ranking

Nach dem Prunen bleiben 342 Kandidaten. Das ist wenig genug, dass man auf der GB10 alle messen könnte- ein Compile kostet etwa 0.4 s. 

`rank` schätzt pro Kandidat eine Laufzeit und sortiert aufsteigend. Das einfachste
Modell (`bw`) rechnet den DRAM-Traffic im Worst Case (kein L2-Treffer über
Gruppengrenzen hinweg) durch die Peak-Bandbreite — eine größere L2-Gruppe bedeutet
weniger Nachladen, also weniger geschätzte Zeit. Kandidaten mit gleichem Traffic
werden nach Grid-Größe absteigend geordnet, weil mehr Blöcke die SMs besser füllen:

```./MLA-SoSe-26/project/src/autotuner/search.py:301-302
        key = "est_ms_occ" if model == "bw_occ" else "est_ms"
        ranked.sort(key=lambda x: (x[1][key], -x[1]["grid"]))
```

Daneben gibt es `bw_occ` (bestraft kleine Grids über die Occupancy) und ein
`roofline`-Modell, das Memory- und Compute-Zeit gegeneinander abwägt; `autotune.py`
benutzt in der Praxis den `bw`-Ranker auf einem vorab register-gefilterten Pool
(dort „v2" genannt). 

---

## Stufe 5 — Aus einer Config wird ein Kernel

Ab hier braucht es die GPU. Es gibt einen einzigen generischen cuTile-Kernel pro Variante, dessen Tile-Größen als `ct.Constant[int]` deklariert sind. 

Der Kern von Variante A: die Tile-Größen kommen als `ct.Constant`, und der
L2-Gruppen-Swizzle wird aus der Block-ID dekodiert:

```./MLA-SoSe-26/project/src/autotuner/kernels.py:13-35
@ct.kernel
def matmul_variant_a(A, B, C,
                     M_PRIM: ct.Constant[int],
                     N_PRIM: ct.Constant[int],
                     K_PRIM: ct.Constant[int],
                     M_L2: ct.Constant[int],
                     N_L2: ct.Constant[int],
                     num_m_l2_outer: ct.Constant[int],
                     num_n_l2_outer: ct.Constant[int],
                     num_k_outer: ct.Constant[int]):
    pid = ct.bid(0)
    n_l2_idx = pid % N_L2
    pid = pid // N_L2
    m_l2_idx = pid % M_L2
    pid = pid // M_L2
    n_l2_outer_idx = pid % num_n_l2_outer
    pid = pid // num_n_l2_outer
    m_l2_outer_idx = pid % num_m_l2_outer
    pid = pid // num_m_l2_outer
    c_idx = pid
    m_block = m_l2_outer_idx * M_L2 + m_l2_idx
    n_block = n_l2_outer_idx * N_L2 + n_l2_idx
```

In der Dekodier-Reihenfolge steckt die L2-Gruppe von oben: `n_l2` und `m_l2` sind
die innersten Faktoren der Block-ID, benachbarte IDs laufen also durch dieselbe
Gruppe, bevor die Gruppe wechselt. Danach die eigentliche Rechnung — über die
K-Streifen laden, akkumulieren, am Ende einmal zurückschreiben:

```./MLA-SoSe-26/project/src/autotuner/kernels.py:37-48
    acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)
    for k_it in range(num_k_outer):
        a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                         shape=(1, M_PRIM, K_PRIM), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                         shape=(1, K_PRIM, N_PRIM), padding_mode=ct.PaddingMode.ZERO)
        a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
        b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
        acc = ct.mma(a_tile, b_tile, acc)
    out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
    ct.store(C, index=(c_idx, m_block, n_block), tile=out)
```

Variante B (`matmul_variant_b`) ist derselbe Kernel mit `m_l2/n_l2` als zwei
SEQ-Loops im CTA statt als Swizzle; der Launcher startet entsprechend ein um
`m_l2 · n_l2` kleineres Grid. Jede kompilierte Config wird gegen `torch.einsum`
geprüft (`allclose`, rtol=1e-2, atol=1e-1), bevor ihre Zeit zählt.

---
