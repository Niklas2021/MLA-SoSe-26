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

M0 ist fertig und geprüft, M1.1 und M1.2 (Knöpfe + Enumerator) stehen in
`project/src/autotuner/search.py` und laufen. Als Nächstes ist das Pruning (M1.3) dran.

Der Enumerator zählt für die A05-Shape 486 Kandidaten auf (nicht die 81 aus dem Pitch — die zählten
nur die Tile-Größen ohne asymmetrisches `m_l2≠n_l2` und ohne die zweite Exec-Variante). Der
Akzeptanztest geht durch: die handoptimierte A05-Config ist im Set enthalten. Krumme Shapes (z. B.
M=1234) werden korrekt hochgepaddet, und A06 fällt mit einer klaren Meldung als M4 raus.

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

- [ ] `prune` schreiben, Anzahl vorher/nachher loggen.

Ehrlich gesagt kennen wir die cuTile-internen SMEM/Register-Limits nicht genau — es kann gut sein,
dass cuTile beim 48-KB-Default bleibt statt das Opt-in-Limit zu nutzen (Notiz dazu steht im
`project_diary.md`). Das Pruning ist also eine Heuristik, kein Beweis. Deshalb kapseln wir später in
M2/M3 jeden Compile in ein `try/except` und protokollieren Fehlschläge als „verworfen", statt uns auf
den statischen Filter allein zu verlassen.

### M1.4 — Ranking

Damit wir in M3 nicht alles messen müssen, ranken wir die geprunten Kandidaten vorab. Statt einer
reinen Arithmetic Intensity (die den L2-Reuse gar nicht abbildet, weil der Effekt an der zeitlichen
Scheduling-Reihenfolge der CTAs hängt) nehmen wir ein einfaches L2-Residency-Modell: Für eine
Swizzle-Gruppe `M_L2 × N_L2` wird A über `N_L2` und B über `M_L2` wiederverwendet, der Working-Set pro
k-Schritt ist etwa `M_L2*M_PRIM*K_PRIM + N_L2*N_PRIM*K_PRIM` (mal 2 Byte). Score ist der geschätzte
DRAM-Traffic, Tie-Break die Occupancy.

- [ ] `rank` schreiben, Top-k an M3 übergeben.

Mit den 25 MB L2 der GB10 könnte das Modell allerdings kaum noch differenzieren, weil fast jeder
sinnvolle Working-Set hineinpasst. Falls das so ist, ist das selbst ein Ergebnis („auf der GB10 zählt
eher Bandbreite und Occupancy als L2-Reuse"), und wir gewichten entsprechend um. Im Zweifel halten wir
M1.4 minimal und lassen einfach die Messung in M3 entscheiden — bei nur ~40–60 Kandidaten ist es
durchaus vertretbar, alle zu messen, statt ein wackeliges Modell zu bauen.

## M2 — Kernel-Instanziierung

Aus einer Config muss ein lauffähiger cuTile-Kernel werden. Der Pitch spricht von „Codegen aus
Templates", aber String-Templates zu erzeugen und per `exec()` zu laden ist fragil. Die A05-Kernel
nehmen ihre Tile-Größen ohnehin schon als `ct.Constant[int]`, also reicht ein einziger generischer
Kernel, den der JIT pro Konstanten-Kombination spezialisiert (so wie Triton das mit `constexpr` macht).
Dass diese Spezialisierung wirklich pro Wert passiert, sollten wir gleich zu Beginn von M2 verifizieren
— davon hängt der ganze Ansatz ab.

- [ ] Generischen Kernel nach Vorlage `kernel_l2` / `kernel_l2_strict` aus `assignments/05_assignment/src/task4.py`,
      parametrisiert über `M_PRIM, N_PRIM, K_PRIM, M_L2, N_L2` und das Exec-Muster.
- [ ] `build_launch(config) -> (kernel, grid, args)`: Grid-Größe, Padding-Buffer (am Ende zurückslicen),
      pid-Zerlegung aus `dim_sizes`/`exec_types`.
- [ ] Smoke-Test: aus der A05-Config erzeugter Kernel liefert dasselbe wie der handgeschriebene.

Scope-Grenze für jetzt: zwei Inputs, GEMM-artig, eine K-Dim und je eine M-/N-Dim. Alles Allgemeinere
heben wir uns für M4 auf.

## M3 — Benchmark & Ranking

Die Top-k-Kandidaten messen, die beste Config küren und die Tuning-Kosten ehrlich ausweisen.
Korrektheit prüfen wir wie in A05 (task4c) gegen die `torch.einsum`-Referenz in fp32, dann nach fp16
gecastet, mit `allclose(rtol=1e-2, atol=1e-1)` und zusätzlich dem max-Fehler, auch für krumme Shapes.
Gemessen wird mit `triton.testing.do_bench` (A05 nutzt warmup = 200, rep = 2000), und aus
`2 * Produkt der Original-Dim-Größen` und der Zeit rechnen wir die TFLOPS.

- [ ] Korrektheitscheck gegen torch.einsum.
- [ ] Benchmark mit do_bench, TFLOPS auf der Original-Shape.
- [ ] Top-k ausgeben, Vergleich gegen Hand-L2 (Ziel ≥ 95 % von 66.10) und gegen die Baseline.
- [ ] Ablation: welcher Knopf wirkt am stärksten (Prim-Größe, L2-Gruppe, Exec-Muster)?

Zwei Punkte, die wir nicht unter den Tisch fallen lassen sollten. Erstens: die eigentlichen
Tuning-Kosten stecken in der Compile-Zeit mal der Anzahl Kandidaten, nicht in der do_bench-Laufzeit.
Das „Minuten statt Stunden"-Argument aus dem Pitch belegen wir nur, wenn wir Anzahl kompilierter
Kandidaten und die gesamte Wall-Clock mitloggen. Zweitens: realistischerweise reproduzieren wir die
schon handoptimierte A05-Lösung — sie zu *schlagen* ist nicht garantiert, weil unser Suchraum klein
ist und die Hand-Lösung nah am Optimum liegt. Das Minimalziel ist, die 95 % zu erreichen und zu zeigen,
dass es ohne Handarbeit gefunden wird.

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
