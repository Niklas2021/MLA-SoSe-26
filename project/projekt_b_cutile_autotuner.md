# Projekt B — cuTile Auto-Tuner für Tensor-Kontraktionen

In Assignment 05 und 06 haben wir die L2-optimale Aufteilung einer Kontraktion jeweils von Hand
hergeleitet. Das Ziel dieses Projekts ist, genau diese Handarbeit zu automatisieren: Aus einem
Einsum-String und den Shapes soll ein kleiner Such- und Benchmarking-Loop selbst eine gute
cuTile-Tiling-Config finden. Der Ablauf ist dabei immer derselbe — aus den Shapes eine Basic-Config
erzeugen, daraus einen eingeschränkten Suchraum an Kandidaten aufspannen, diesen billig vorfiltern,
die übrig gebliebenen Configs zu Kernels machen und mit `do_bench` messen, und am Ende die beste
(plus ein paar gute) Configs ausgeben.

Wir bauen komplett auf der Infrastruktur aus Assignment 05 auf. `Config`, `Optimizer` und
`generate_config` haben wir dafür unverändert nach `project/src/autotuner/` übernommen. Als Testfälle
dienen die beiden Kontraktionen, für die wir schon Hand-Lösungen und Messwerte haben: der batched
Matmul aus A05 (`cmk,ckn->cmn`) und die Tensor-Ring-Kontraktion aus A06 (`acspx,bspy->abcyx`).

Die konkreten Zahlen weiter unten (Tile-Größen, Schwellen) stammen größtenteils aus dem Pitch und
sind teils noch optimistisch. Wo ich beim Aufschreiben gestolpert bin, steht es als Notiz dabei,
damit wir das in M0/M1 prüfen statt blind zu übernehmen.

## Roadmap

| Stufe | Inhalt | Ergebnis |
|---|---|---|
| M0 | Setup & Projektgerüst | Umgebung läuft, A05-Code als Bibliothek, Baselines fix |
| M1 | Config-Suchraum (Enumerator, Pruning, Ranking) | Liste gültiger Kandidaten-Configs, noch ohne Kernel |
| M2 | Kernel-Instanziierung | Eine aus einer Config erzeugte cuTile-Variante läuft korrekt |
| M3 | Benchmark & Ranking | do_bench + Korrektheit + Top-k, Vergleich gegen Hand-L2 |
| M4 | Transfer auf A06 | Tensor-Ring als zweiter Test, optional Config-Cache |

Minimal wollen wir mit M0–M3 die A05-Lösung reproduzieren (mindestens 95 % der Hand-L2-Performance).
M4 ist die Erweiterung und prüft, ob das Ganze auch auf die schwierigere A06-Kontraktion übergeht.

## Stand (24.06.2026)

M0 bis M3 sind durch, inklusive einer Multi-Shape-Studie über acht Regime (`problems.py`). Kurzfassung:
der Tuner ist dem Handkernel gleichwertig, wo Handtuning passt, und bis zu 21 % besser, wo nicht (krumme
Shape) — und das in ~3 s pro Shape, wenn man nur die Modell-Top-7 misst. Das analytische Ranking taugt
dabei nur als grobe Vorauswahl, nicht als exakter Ranker (Spearman im Schnitt −0.26). Details in M3.
Offen ist noch M4 (Transfer auf A06).

Der Enumerator zählt für die A05-Shape 486 Kandidaten auf (nicht die 81 aus dem Pitch — die zählten
nur die Tile-Größen ohne asymmetrisches `m_l2≠n_l2` und ohne die zweite Exec-Variante). Der
Akzeptanztest geht durch: die handoptimierte A05-Config ist im Set enthalten. Krumme Shapes (z. B.
M=1234) werden korrekt hochgepaddet, und A06 fällt mit einer klaren Meldung als M4 raus.

Das Pruning bringt bei A05 nur 486 → 342 (126 wegen SMEM, 18 wegen Akku-Registern; Padding und MMA
greifen hier nicht, weil 4096 glatt teilbar ist). Das ist weniger als erhofft, und der Grund ist
strukturell: das pro-Block-SMEM hängt nur an den Prim-Größen, nicht an `m_l2/n_l2` oder der Variante.
Statisches Pruning kann diese beiden Achsen also gar nicht beschneiden — die bleiben für M1.4/M3 übrig.

Wichtig ist auch, was wir mit den Hardware-Daten *nicht* mehr prunen können: die L2-Reuse-Regel aus
der Vorlesung würde verlangen, dass der Working-Set einer Swizzle-Gruppe ins L2 passt — bei der
größten überlebenden Gruppe sind das aber nur rund 256 KB gegen 25 MB L2. Die Regel greift auf der
GB10 also schlicht nicht; das L2 ist zu groß, um die Gruppengröße einzuschränken. Genau deshalb
verschiebt sich die Entscheidung über `m_l2/n_l2` und Variante komplett auf die Messung.

Als optionale, *verlustbehaftete* Reduktion gibt es noch `dedup_mn_symmetry`: bei quadratischen
Problemen sind gespiegelte `(m_prim,m_l2)/(n_prim,n_l2)`-Paare grob austauschbar (342 → 186 bei A05).
Grob, weil M und N im Speicher nicht symmetrisch sind — deshalb nicht im Default-Pfad, nur als
schnelle Vorauswahl für einen ersten Mess-Durchlauf.

Beim Durchsehen der Umgebung sind ein paar Dinge aufgefallen, die in M1 einfließen. Die Ziel-GPU ist
eine **NVIDIA GB10** (DGX Spark, Compute Capability 12.1, 48 SMs, cuTile 1.4.0, CUDA 13). Wichtig für
uns: sie ist *integriert* und teilt sich den LPDDR-Speicher mit der CPU, und ihr L2-Cache ist mit
**25 MB** ungewöhnlich groß. Das ist keine diskrete H100/A100, und das verschiebt die L2-Reuse-Story
spürbar — Working-Sets passen viel eher komplett ins L2, die eigentliche Knappheit ist eher die
DRAM-Bandbreite. Das Kostenmodell in M1.4 muss also mit den echten 25 MB rechnen, nicht mit Annahmen
aus den üblichen Datacenter-Tutorials.

Zweite Sache: die Baselines in `baselines.py` (66.10 / 38.60 / 49.84 / 16.18 TFLOPS) sind aus den
A05/A06-Logs übernommen und mit `BASELINE_GPU = "NVIDIA GB10"` gelabelt. Die Logs selbst nennen die
GPU aber nicht explizit. Bevor wir in M3 die 95-%-Ziele ernst nehmen, sollten wir A05/A06 einmal auf
der GB10 nachmessen.

## M0 — Setup & Projektgerüst (erledigt)

- [x] Projektstruktur `project/src/` und `project/results/` angelegt (`cache/` kommt erst in M4).
- [x] `Config`/`Optimizer`/`generate_config` aus A05 nach `project/src/autotuner/` übernommen
      (`config.py`, `optimizer.py`, `generate.py`), Semantik unverändert, Import getestet.
- [x] Smoke-Test `project/src/smoke_test.py`: minimaler Kernel + `ct.launch` + `do_bench` laufen,
      Versionen und GPU-Properties landen in `results/smoke_test.log`.
- [x] Baselines in `baselines.py` eingefroren (siehe Stand-Abschnitt zur offenen Konsistenzfrage).

## M1 — Config-Suchraum

Hier geht es weiter. Die ganze Logik kommt in ein neues Modul `project/src/autotuner/search.py`, das
bewusst nur reines Python ist und kein cuTile importiert — so können wir es auch lokal ohne GPU testen.
Sinnvolle Reihenfolge: erst die Knöpfe und den Suchraum festlegen, dann den Enumerator, dann das
Pruning, zuletzt das Ranking.

Der erste echte Akzeptanztest für M1: Für die A05-Shape `(C=4, M=N=K=4096)` muss die handoptimierte
Config (m_prim = n_prim = 128, k_prim = 64, m_l2 = n_l2 = 8, Variante A) im enumerierten und geprunten
Set auftauchen. Wenn sie da rausfällt, kann der Tuner sie in M3 nie finden.

### M1.1 — Knöpfe definieren

Wir halten den Suchraum bewusst klein. Als Prim-Tiles nehmen wir `M_PRIM, N_PRIM ∈ {64, 128, 256}`
und `K_PRIM ∈ {32, 64, 128}`. Dazu kommt die L2-Gruppengröße `M_L2, N_L2 ∈ {2, 4, 8}` — das ist die
*zeitliche* Block-Gruppe, die den L2-Reuse erzeugt (das Gegenstück zu `group_size_m` im
Triton-Matmul-Tutorial), nicht eine räumliche Kachel. Als drittes brauchen wir das Exec-Muster, also
mindestens die zwei Varianten aus A05: Variante A mit `m_l2/n_l2` als PAR (Swizzling) und Variante B
mit `m_l2/n_l2` als SEQ-Loops.

- [x] Konstanten und eine kleine `SearchSpace`-Struktur in `search.py` angelegt.

Die im Pitch genannte Zahl „3·3·3·3 = 81" ist nur die Tile-Kombinatorik. Mit beiden Exec-Mustern und
asymmetrischem `M_L2 ≠ N_L2` sind es tatsächlich 486 (= 3⁵ · 2). Pruning muss da also noch ordentlich was wegnehmen.

### M1.2 — Enumerator

`enumerate_candidates(einsum, shapes)` erzeugt für jede Knopf-Kombination eine Config: erst
`generate_config` für die Basic-Config, dann mit dem `Optimizer` die M/N-Dims in (l2_outer, l2, prim)
und die K-Dim in (k_outer, k_prim) splitten, umsortieren und die Exec-Types setzen. Jeder Kandidat
muss am Ende durch `verify()` (K nie PAR, Reihenfolge PAR | SEQ | PRIM, PRIM enthält M, N und K).

- [x] `enumerate_candidates` geschrieben (liefert `Candidate`-Objekte mit Config + Knöpfen + Original-M/N/K).
- [x] Akzeptanztest: A05-Hand-Config ist im Ergebnis enthalten (Selbsttest in `search.py`).

Ein Stolperstein, den `split_dim` uns aufzwingt: es verlangt, dass `outer * inner` exakt die alte
Größe ergibt, und wirft sonst einen `ValueError`. Krumme Shapes wie M = 1234 gehen also nicht direkt.
Wir splitten deshalb auf der gepaddeten Größe `ceildiv(M, prim*l2) * prim*l2` und nullen den Überhang
später im Kernel über `PaddingMode.ZERO` — genau so macht es der A05-Launcher schon. Das heißt, die
`dim_sizes` der Config sind die *gepaddeten* Größen; die TFLOPS rechnen wir trotzdem auf der
Original-Shape.

### M1.3 — Static Pruning

`prune(candidates, device_props)` wirft alles raus, was sich schon ohne Kompilieren als unsinnig
erkennen lässt. Die MMA-Teilbarkeit (K_PRIM und M/N_PRIM als Vielfache von 16 für die fp16-Tensor-Cores)
erfüllen unsere Kandidatenwerte zwar alle, aber der SMEM-Check greift: nutzbares Shared Memory pro
Block auf der GB10 ist `MaxSharedMemoryPerBlockOptin − ReservedSharedMemoryPerBlock`, also
101376 − 1024 ≈ 100 KB. Pro Block brauchen wir grob `(M_PRIM*K_PRIM + K_PRIM*N_PRIM) * 2` Byte für die
fp16-Operanden plus `M_PRIM*N_PRIM * 4` Byte Akku, mit Double-Buffering nochmal Faktor zwei auf den
Operanden. Die `device_props` reichen wir aus `device_props.py` rein.

Umgesetzt sind vier Filter, vom Fundamentalsten zum Weichsten: MMA-Teilbarkeit (Guard), das
SMEM-Budget (der harte Filter), ein Register-Check für den Akku (`M_PRIM*N_PRIM` fp32 gegen die halbe
Registerdatei) und Padding-Verschwendung (gepaddetes Volumen gegen das Original). `prune` gibt
`(kept, rejected)` zurück, wobei `rejected` den Grund mitführt, damit nachvollziehbar bleibt was
warum wegfällt. Die Hardware-Werte kommen als `DeviceProperties` rein; für die Tests ohne GPU liegt
ein fertiges `GB10`-Objekt in `device_props.py` (der cupy-Import ist dort jetzt lazy).

- [x] `prune` mit allen vier Filtern geschrieben, kept/rejected + Gründe geloggt.
- [x] Akzeptanztest: A05-Hand-Config überlebt, alle 256×256-Tiles fallen weg, Padding/MMA-Filter
      mit Mini-Shapes bzw. ungeraden Knöpfen verifiziert.

Ehrlich gesagt kennen wir die cuTile-internen SMEM/Register-Limits nicht genau — es kann gut sein,
dass cuTile beim 48-KB-Default bleibt statt das Opt-in-Limit zu nutzen (Notiz dazu steht im
`project_diary.md`). Deshalb sind `buffer_stages`, `smem_limit`, `reg_fraction` und `max_padding`
Parameter mit optimistischen Defaults (Opt-in-SMEM, Double Buffering). Das Pruning bleibt eine
Heuristik, kein Beweis — der eigentliche Schutz ist das `try/except` ums Kompilieren in M2/M3.

### M1.4 — Ranking

Damit wir in M3 nicht alles messen müssen, ranken wir die geprunten Kandidaten vorab. Statt einer
reinen Arithmetic Intensity (die den L2-Reuse gar nicht abbildet, weil der Effekt an der zeitlichen
Scheduling-Reihenfolge der CTAs hängt) nehmen wir ein einfaches L2-Residency-Modell: Für eine
Swizzle-Gruppe `M_L2 × N_L2` wird A über `N_L2` und B über `M_L2` wiederverwendet, der Working-Set pro
k-Schritt ist etwa `M_L2*M_PRIM*K_PRIM + N_L2*N_PRIM*K_PRIM` (mal 2 Byte). Score ist der geschätzte
DRAM-Traffic, Tie-Break die Occupancy.

Wichtig: das Ranking ist *nicht* zum Zeitsparen da. Der Compile kostet nur ~0.4 s (Server-Messung), wir
könnten alle 342 problemlos durchmessen. Der eigentliche Zweck ist die Forschungsfrage: zieht unser
Modell die tatsächlich beste Config in seine Top-k bzw. Top-1? Das prüfen wir in M3 gegen die
Vollmessung (Ground Truth).

- [x] `rank` als bandbreitenbasiertes Kostenmodell geschrieben (DRAM-Traffic / Peak-Bandbreite, Grid
      als Tie-Break). Bandbreite kommt aus `device_props` (~270 GB/s auf der GB10).

Erster Befund aus dem Modell (A05): die gemessene 66.54-TFLOPS-Config (128×128, 8×8, A) steht auf
Rang #10 von 342 — das Modell bevorzugt größere Tiles (128×256 / 256×128), weil die weniger DRAM-Traffic
machen. Zwei Schwächen zeigt es sofort: es kann Variante A und B nicht unterscheiden (gleiche Tile-Form =
gleiche Bytes), und es übersieht Occupancy — Variante B mit großer Gruppe hat nur grid=32 CTAs bei 48 SMs
(< 1 Wave), kriegt aber dieselbe Vorhersage wie A. Ob das in der Realität durchschlägt, zeigt M3. Optional
bauen wir einen Occupancy-Term ein und evaluieren beide Modell-Varianten gegeneinander (Ablation).

## M2 — Kernel-Instanziierung

Aus einer Config muss ein lauffähiger cuTile-Kernel werden. Der Pitch spricht von „Codegen aus
Templates", aber String-Templates zu erzeugen und per `exec()` zu laden ist fragil. Die A05-Kernel
nehmen ihre Tile-Größen ohnehin schon als `ct.Constant[int]`, also reicht ein einziger generischer
Kernel, den der JIT pro Konstanten-Kombination spezialisiert (so wie Triton das mit `constexpr` macht).
Dass diese Spezialisierung wirklich pro Wert passiert, sollten wir gleich zu Beginn von M2 verifizieren
— davon hängt der ganze Ansatz ab.

Erster Schritt steht: der generische Kernel `matmul_variant_a` (Variante A, Swizzling) plus Launcher
`run_variant_a` in `project/src/autotuner/kernels.py`, mit den Tile-Größen als `ct.Constant`. Dazu das
Mess-Skript `project/src/measure_compile.py`, das auf der GB10 die offenen Fragen klärt: kompiliert der
Kernel überhaupt, stimmt das Ergebnis gegen torch, wie lange dauert ein Compile (hochgerechnet auf 342
bzw. 186 Configs), und spezialisiert `ct.Constant` wirklich pro Wert (zweite Config mit anderem M_PRIM).

> **➡ Auf dem Server auszuführen:** `python measure_compile.py` (aus `project/src/`). Das Ergebnis
> entscheidet, wie es weitergeht — liegt die Compile-Zeit im Minutenbereich für alle Configs, sparen wir
> uns M1.4 und messen einfach alles; ist sie hoch, bauen wir das Ranking aus. Log landet in
> `results/measure_compile.log`.

- [x] Generischen Kernel `matmul_variant_a` + `run_variant_a` geschrieben (Variante A).
- [x] Auf der GB10 verifiziert: kompiliert, korrekt, `ct.Constant`-Spezialisierung, Compile ~0.4 s,
      66.54 TFLOPS (= Hand-L2) — Output in `results/measure_compile.log`.
- [x] Variante B (`matmul_variant_b` + `run_variant_b`, m_l2/n_l2 als SEQ-Loops).
- [x] `run_candidate(cand, A, B)`: Dispatcher, der aus einem `Candidate` den passenden Kernel startet.
- [x] Smoke-Test: aus der A05-Config erzeugter Variante-A-Kernel == handgeschriebener (66.54 TFLOPS).

Scope-Grenze für jetzt: zwei Inputs, GEMM-artig, eine K-Dim und je eine M-/N-Dim. Alles Allgemeinere
heben wir uns für M4 auf.

## M3 — Vollmessung & Modell-Evaluation

Der Harness steht: `project/src/tune.py`. Er enumeriert und prunt (ohne dedup — wir messen fair alles),
kompiliert jeden Kandidaten in einem `try/except`, prüft Korrektheit gegen `torch.einsum`
(`allclose(rtol=1e-2, atol=1e-1)`) und misst mit `do_bench`. Die nach TFLOPS sortierte Liste ist unsere
Ground Truth. Compile-Fehler und inkorrekte Configs werden als solche protokolliert, nicht verschwiegen.

Das eigentliche Ergebnis ist die Modell-Evaluation: für beide Modelle (`bw` und `bw_occ`) berichtet der
Harness, auf welchem Modell-Rang die *gemessen* beste Config steht, recall@k für k ∈ {1,5,10,20}, und
die Spearman-Korrelation zwischen Modell-Schätzung und Messung. Damit beantworten wir direkt: zieht
unsere Eingrenzung die besten Configs nach oben, und hilft der Occupancy-Term?

> **➡ Auf dem Server auszuführen:** `python tune.py` (aus `project/src/`). Voll-Sweep über die ~342
> Kandidaten mit moderaten do_bench-Settings (warmup=50, rep=300, ~10–15 min). Output:
> `results/tune_a05.csv` (alle Configs, fürs Plotten) + `results/tune_a05.log` (Zusammenfassung).

- [x] Harness `tune.py`: Korrektheit + do_bench über alle Kandidaten, CSV/Log-Output.
- [x] Auf der GB10 gelaufen: 342 Kandidaten, alle korrekt, 0 Fehlschläge (`results/tune_a05.*`).
- [x] Auswertung reproduzierbar in `analyze_tune.py` (läuft lokal aus dem CSV).
- [x] Ablation: Prim-Form dominiert klar; L2-Gruppe und Variante wirken nur zweitrangig.

### Ergebnisse

Gemessen wurde auf der GB10 über acht Shapes (in `problems.py`), jeweils alle 342 geprunten Configs.
Insgesamt 22 min Sweep, und alle 8×342 Messungen liefen durch — keine ist am Compile gescheitert oder
falsch gewesen, auch nicht die unteilbare `krumm`-Shape, womit der Padding-Pfad auf der echten Hardware
bestätigt ist. Reproduzierbar ist die ganze Auswertung mit `analyze_tune.py` (läuft lokal aus den CSVs).

**Was die beste Config ist.** In 7 von 8 Fällen gewinnt `128/128/64` (Prim-Größen), nur bei großem K
will der Kernel ein größeres k_prim (`64/128/128`). Es gibt also eine ziemlich robuste Quasi-Universal-
Wahl; die L2-Gruppe und A-vs-B feilen oben nur noch ein paar Prozent. Die asymmetrischen 256-breiten
Tiles sind dagegen Gift (3–14 TFLOPS statt ~65) — die sprengen das Registerbudget.

**Lohnt sich der Tuner gegenüber dem Handkernel?** Der Handkernel ist hier unsere Default-Config
(`128/128/64`, 8×8). Auf den regulären GEMMs holt der Tuner praktisch nichts heraus (1.00×) — er
*bestätigt* die Handarbeit. Wo es vom Heimvorteil weggeht, schlägt er sie aber: +3 % bei großem K und
+21 % auf der krummen Shape, wo die feste 8×8-Gruppe einfach schlecht passt.

| Shape | Hand | Tuner (top-7) | abs. Best | Tuner/Hand |
|---|---|---|---|---|
| a05 / square_1b / tall / wide | 60–65 | = Hand | +1–3 % | 1.00× |
| small_k | 36.4 | = Hand | 36.9 | 1.00× |
| large_k | 44.3 | 45.5 | 47.5 | 1.03× |
| krumm | 33.6 | 40.7 | 42.4 | **1.21×** |
| batch16 | 46.0 | 46.0 | 46.0 | 1.00× |

Im Schnitt ist der Tuner 1.03× über dem Handkernel, nie schlechter, und erreicht 97.6 % des absoluten
Optimums.

**Das Ranking-Modell taugt nicht als Ranker — aber als Vorfilter.** Das reine Bandbreitenmodell ist über
alle Shapes negativ korreliert (Spearman im Schnitt −0.26), weil es ausgerechnet die 256-breiten
Register-Fresser nach oben sortiert. Auch die Hoffnung, dass es im bandbreitenlimitierten `small_k`
greift, geht nicht auf (−0.30) — das 25-MB-L2 schluckt den Traffic auch dort. Filtert man die
Register-Fresser raus (v2, Akku ≤ 0.4·Registerdatei), verschwindet die Anti-Korrelation, ein guter
Ranker wird es trotzdem nicht (Schnitt ~0).

Der Witz ist: als *exakter* Ranker scheitert es, aber als *Vorauswahl* reicht es. Misst man nur die
Modell-Top-7 und nimmt die schnellste davon, landet man in jedem der acht Regime bei ≥ 95 % des
Optimums (im Schnitt 97.6 %). Und das kostet ~3 s statt ~3 min Voll-Sweep, weil pro Config der Compile
(~0.4 s) dominiert und wir eben nur 7 statt 342 kompilieren. Die exakte Beste erwischt man so meist
nicht (die sitzt im Modell tief, und das Spitzenfeld liegt ohnehin innerhalb ~3 % im Messrauschen) —
das ist der ehrliche Preis.

Unterm Strich: die Eingrenzung (enumerate + prune) ist das Sichere und Wertvolle, das analytische
Ranking taugt auf dieser Hardware nur grob zur Vorauswahl, und die eigentliche Entscheidung muss man
messen. Genau das rechtfertigt den Tuner: gleichwertig dort, wo Handtuning passt, klar besser dort, wo
nicht — bei vernachlässigbarem Aufwand.

## M4 — Transfer auf A06 (Erweiterung)

Als zweiten Testfall nehmen wir die Tensor-Ring-Kontraktion `acspx,bspy->abcyx`. Das ist deutlich
schwerer, als der Pitch klingt: A06 hat zwei Reduktionsdimensionen (`s` und `p`) und mehrere M-, N- und
Batch-artige Dims. Der generische Single-K-Kernel aus M2 deckt das nicht ab. Entweder fusionieren wir
die beiden K-Dims mit `fuse_dims` zu einer (nur wenn sie adjazent/contiguous sind — prüfen), oder wir
brauchen einen zweiten Kernel-Typ mit verschachtelter K-Schleife. Hier wird also echtes
Strukturmuster-Handling nötig.

- [ ] A06-Kontraktion einspeisen und korrekt rechnen.
- [ ] Ziel ≥ 90 % der Hand-cuTile-Performance aus A06 (49.84 TFLOPS).
- [ ] Optional: Config-Cache, keyed nach `(einsum, shapes, GPU-Modell)`. Das GPU-Modell muss in den Key,
      weil die optimale L2-Gruppe von der L2-Größe abhängt.

## Offene Fragen

Ein paar Dinge sollten wir klären, bevor bzw. während wir implementieren:

- Kostenmodell oder einfach alles messen? Wenn der geprunte Raum klein bleibt, ist „alle messen"
  robuster als ein wackeliges Residency-Modell. Entscheiden wir nach der realen Suchraumgröße in M1.
- Spezialisiert `ct.Constant` wirklich pro Wert? Falls nicht, müssen wir doch zu String-Templates
  greifen. Früh in M2 testen.
- Padding sauber durchhalten: `dim_sizes` gepaddet, OOB über `PaddingMode.ZERO`, TFLOPS aber auf der
  Original-Shape.
- Hardware-Abhängigkeit: alle Prozentziele gelten relativ zu der GPU, auf der die Baselines gemessen
  wurden. Bei GPU-Wechsel neu messen.
- Scope-Disziplin: kein allgemeiner Tensor-Compiler. M0–M3 sind ein K-Dim mit zwei Inputs, A06 ist
  Stretch und darf scheitern, ohne das Kernprojekt zu kippen.
