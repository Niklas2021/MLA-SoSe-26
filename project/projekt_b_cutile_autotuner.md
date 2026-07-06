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

## Stand (05.07.2026)

M0 bis M4 sind durch. M0–M3: Multi-Shape-Studie über acht GEMM-Regime (`problems.py`) — der Tuner ist dem
Handkernel gleichwertig, wo Handtuning passt, und bis zu 21 % besser, wo nicht (krumme Shape), in ~3 s pro
Shape über die Modell-Top-7. Das analytische Ranking taugt dort nur als grobe Vorauswahl, nicht als
exakter Ranker (Spearman im Schnitt −0.26). Details in M3.

M4 (Transfer auf A06) ist ebenfalls durch: die Tensor-Ring-Kontraktion `acspx,bspy->abcyx` läuft als
zweite Struktur-Familie über einen eigenen Ring-Kernel. Über acht Ring-Shapes rechnen alle 8×171 Configs
korrekt (0 Fehlschläge), und auf der Referenz-Shape schlägt der Tuner den A06-Handkernel mit 61.9 gegen
49.84 TFLOPS (+24 %), weit über dem 90-%-Ziel. Details in M4.

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

Will man näher ans Optimum, gibt es eine Mittelstufe: für ≥ 99 % muss man bis ~top-90 messen (worst
case `small_k`, sonst top-20–40), das sind ~45 s pro Shape statt ~3 s. Damit hat man drei Stufen —
top-7 / ~3 s / ≥ 95 %, top-90 / ~45 s / ≥ 99 %, Voll-Sweep / ~3 min / 100 %. Die letzten paar Prozent
liegen aber im Messrauschen, also lohnt der Aufwand selten; top-7 ist praktisch der sweet spot.

Und ob sich Tuning überhaupt lohnt, hängt an der Wiederverwendung: eine Kontraktion läuft in ~8 ms,
die 3 s Tuning sind also nur ein Einmalkosten-Posten, der sich erst über viele Aufrufe derselben Shape
amortisiert (bei ~2–3 % Gewinn über den Default braucht es Tausende Aufrufe, bei den Ausreißer-Shapes
wie `krumm` mit +26 % nur ein paar hundert). Für eine einmalige Rechnung also einfach rechnen lassen,
für Schleifen mit fester Shape tunen und die Config cachen (Key = Shape + GPU).

In der Praxis ist die Wiederverwendung der Normalfall: in einem Netz sind die Layer-Dimensionen fix,
jeder Trainings-Step macht dieselben Matmuls — über ein Training also millionenfach dieselbe Shape
(Inferenz genauso). Den Break-even von ~15.000 Aufrufen erreicht man damit in Sekunden, die einmaligen
3 s sind vernachlässigbar. Genau deshalb tunen Frameworks wie cuBLAS, Triton (`autotune`) oder
torch.compile pro Shape einmal und cachen. Nicht auf geht die Rechnung nur bei stark dynamischen Shapes
(z. B. ständig wechselnde Sequenzlängen), wo jede Shape selten vorkommt — dagegen hilft in der Praxis
Bucketing/Padding auf wenige feste Shapes.

Unterm Strich: die Eingrenzung (enumerate + prune) ist das Sichere und Wertvolle, das analytische
Ranking taugt auf dieser Hardware nur grob zur Vorauswahl, und die eigentliche Entscheidung muss man
messen. Genau das rechtfertigt den Tuner: gleichwertig dort, wo Handtuning passt, klar besser dort, wo
nicht — bei vernachlässigbarem Aufwand.

## M4 — Transfer auf A06 (Erweiterung)

Als zweiten Testfall nehmen wir die Tensor-Ring-Kontraktion `acspx,bspy->abcyx`. Das ist deutlich
schwerer, als der Pitch klingt: A06 hat zwei Reduktionsdimensionen (`s` und `p`) und mehrere M-, N- und
Batch-artige Dims. Der Kern des Problems ist aber nicht das doppelte K, sondern die **Batch-Topologie**:
A05 hat einen *geteilten* Batch (`c` indiziert A, B und C mit demselben Index), A06 hat *unabhängige*
Batches (`a,c` nur in A, `b` nur in B). Der M2-Kernel indiziert A und B mit demselben `c_idx` — das
kann A06 gar nicht ausdrücken; zwängt man es durch, kämen alle `a×b`-Kombinationen statt der Diagonale.
Deshalb ist ein *zweiter Kernel-Typ* nötig, kein Umbau des bestehenden.

Das passt zur Philosophie: der Tuner automatisiert die **Config-Suche**, nicht das Kernel-Schreiben —
schon M0–M3 hatten zwei Hand-Kernel (Variante A/B), und der Tuner sucht/misst nur Tiling-Configs. A06
ist eine *zweite Struktur-Familie*, kein Kernel pro Einsum: innerhalb einer Familie handhabt der Tuner
beliebige Shapes über die `ct.Constant`-Spezialisierung (wie in M3 über 8 Shapes gezeigt); erst eine
neue Topologie (andere Batch-/Reduktionsstruktur, mehr Inputs) braucht ein neues Template. Genauso
arbeiten cuBLAS/CUTLASS (endliche Template-Menge, Configs getunt) und Triton (`autotune` pro `@jit`).

Umgesetzt (Code fertig, Messung auf der GB10 steht noch aus):

- Klassifikation verallgemeinert: `parse_einsum` liefert jetzt `m_chars/n_chars/k_chars` plus die
  gewählten prim-Dims (die inneren, stride-1: `x`→prim_m, `y`→prim_n, `p`→prim_k) und den Rest als
  `extra_m_chars` (`a,c` → A-seitiger PAR-Batch), `extra_n_chars` (`b`) und `seq_k_chars` (`s` → SEQ-Loop).
  Der Single-M/N/K-Pfad (A05) bleibt bitgleich (486→342, Hand-Config überlebt).
- `generate_config` klassifiziert über die Listen statt über die einzelnen prim-Chars — sonst landet die
  Reduktionsdim `s` fälschlich als N/PAR und es würde nicht über `s` summiert.
- `build_one_config` baut die A06-Config über `make_executable` (das markiert die jeweils letzte M/N/K-Dim
  als PRIM und legt `extra`→PAR, `seq_k`→SEQ korrekt ab). Variante B (strict-Loops) deckt den Mehrdim-Fall
  nicht ab und wird für A06 übersprungen — bleibt also nur Variante A (Swizzle), wie in der Handlösung.
- Neuer Kernel `matmul_ring_a` + `run_ring_a` in `kernels.py`: parametrisierte Version des A06-Hand-Kernels
  (`kernel_lf`), Tile-Größen als `ct.Constant`, mit Per-Tile-`permute` (Layout A/B/C ist nicht mma-fertig)
  und äußerer SEQ-Schleife über `s`. `run_candidate` dispatcht per `cand.multi` auf den Ring-Kernel.
- A06 als Problem in `problems.py`; `flops_and_batch` verallgemeinert auf `2·∏(Output-Dims)·∏(K-Dims)`
  (die alte `batch·m·n·k`-Formel unterschlägt `a,c,b,s`). Enumerator: 243 Kandidaten (nur Variante A),
  Pruning → 171, ~696 GFLOP.

- [x] `python tune.py a06` auf der GB10: 8 Ring-Shapes, je 171 Configs, **0 fehlgeschlagen/inkorrekt**
      (auch `a06_krumm` mit Padding auf x, y und p) — der Ring-Kernel kompiliert und rechnet korrekt.
      Output: `results/tune_a06*.csv` + `results/study_a06.log`, 21 min Sweep.
- [x] Ziel ≥ 90 % der Hand-cuTile-Performance klar erreicht — sogar **übertroffen**: siehe Ergebnisse.
- [x] Optional: Config-Cache, keyed nach `(einsum, shapes, GPU-Modell)` (`autotuner/cache.py` + `autotune.py`).
      Das GPU-Modell muss in den Key, weil die optimale L2-Gruppe von der L2-Größe abhängt.

### Ergebnisse

Gemessen auf der GB10 über acht Ring-Shapes (`problems.py`), jeweils alle 171 geprunten Configs, alle
korrekt. Reproduzierbar mit `analyze_tune.py`.

**Der Transfer gelingt — und der Tuner schlägt den Handkernel.** Auf der Referenz-Shape (Original-A06)
holt der Tuner **61.9 TFLOPS** gegen die 49.84 der handoptimierten cuTile-Variante: das sind **1.24× bzw.
+24 %**, weit über dem 90-%-Ziel (44.86). Der Gewinn kommt aus `k_prim=32`: der Handkernel nahm `p=64`
als einen mma-Tile, der Tuner teilt die p-Reduktion in zwei 32er-Kacheln und ist damit ~14 % schneller
(selbst die hand-nahe 128/128/`k_prim=64`-Wahl liegt hier schon bei 54.5 TFLOPS).

| Shape | Default (8×8) | Tuner best | Tuner/Default |
|---|---|---|---|
| a06 (Referenz) | 27.0 | **61.9** | 2.29× |
| a06_square (x==y) | 59.4 | 69.4 | 1.17× |
| a06_tall (x≫y) | 31.6 | 69.2 | 2.19× |
| a06_wide (y≫x) | 30.6 | 67.2 | 2.19× |
| a06_small_k (s·p=256) | 18.4 | 22.6 | 1.23× |
| a06_large_k (s·p=16384) | 64.1 | 75.1 | 1.17× |
| a06_krumm (unteilbar) | 14.3 | 21.0 | 1.47× |
| a06_batch (a8 c4 b8) | 57.1 | 62.7 | 1.10× |

Der Tuner-Gewinn ist hier durchweg größer als bei A05 (1.10–2.29× statt ~1.00×), weil die aus A05
übernommene Default-Config (8×8-Gruppe, `k_prim=64`) auf die A06-Shapes schlecht passt — die feste
8×8-Gruppe paddet x/y stark und trifft das p-Tiling nicht. Genau das automatisiert der Tuner weg.

**Das Modell taugt hier sogar als Ranker, nicht nur als Vorfilter.** Anders als bei A05 korreliert das
register-gefilterte Modell (`v2`) auf den Ring-Shapes positiv (Spearman im Schnitt **+0.38**, vs. ~0 bei
A05); das reine Bandbreitenmodell bleibt bei ~0. Als Vorauswahl reicht es ohnehin: misst man nur die
Modell-Top-7 und nimmt die schnellste, landet man in jedem der acht Regime bei **95.1–100 % des Optimums**
(im Schnitt ~98 %, wie bei A05) — bei ~3 s statt ~2 min Voll-Sweep pro Shape.

Unterm Strich bestätigt A06 die A05-Befunde und geht darüber hinaus: die enumerate+prune-Maschinerie und
der eine zusätzliche Kernel-Typ transferieren sauber auf eine zweite Struktur-Familie, alle Configs
rechnen korrekt (inkl. Padding-Pfad), und der Tuner schlägt den *Handkernel* klar (49.84 → ~60 TFLOPS) —
weil er eine Achse (`k_prim`) mitdurchsucht, die von Hand nicht angefasst wurde. Der Handkernel ist aber,
wie die torch-Referenz unten zeigt, gar nicht die richtige Messlatte.

### Externe Referenz: torch.einsum (cuBLAS)

Neben dem Default misst `tune.py` jetzt `torch.einsum` in fp16 pro Shape — die ehrliche „was, wenn man
einfach die Library nimmt"-Baseline (GB10, `result_dgx/study.log`). Das Bild ist differenzierter als der
Assignment-Wert (16.18) vermuten ließ.

**GEMM (A05-Familie): cuBLAS gewinnt.** Über die acht GEMM-Shapes erreicht unser cuTile-Tuner im Schnitt
~77 % von torch (geom. Mittel), nur auf der hand-getunten a05-Referenz zieht er knapp vorbei (1.04×). Das
ist erwartbar und ehrlich: cuBLAS ist eine gereifte GEMM-Library; sie zu schlagen ist nicht das Ziel.

**Ring (A06-Familie): stark shape-abhängig, im Mittel leicht vorn (~1.17× geom.).** torch.einsum ist hier
*nicht* durchweg schwach — bei manchen Shapes findet es einen guten Pfad (→ bmm), bei anderen nicht. Der
Tuner gewinnt groß, wo torchs Pfad schlecht ist, und verliert, wo er gut ist:

| A06-Shape | Tuner/torch | | A06-Shape | Tuner/torch |
|---|---|---|---|---|
| a06_tall | **3.95×** | | a06 (Ref) | 0.99× |
| a06_large_k | 2.65× | | a06_batch | 0.81× |
| a06_square | 1.42× | | a06_small_k | 0.59× |
| a06_wide | 1.30× | | a06_krumm | 0.38× |

Zwei ehrliche Korrekturen: (1) `A06_TORCH_EINSUM = 16.18` aus dem Assignment ist veraltet/nicht
vergleichbar — dieselbe Referenz-Shape macht mit aktuellem torch.einsum-fp16 **60.22 TFLOPS**, gleichauf
mit unserem Tuner (59.83) und über dem Handkernel (49.84). (2) Der Mehrwert des Tuners ist damit präziser
gefasst: nicht „schneller als alles", sondern *ein allgemeiner Mechanismus, der ohne Handarbeit über
beliebige Kontraktionen brauchbare Leistung liefert und dort deutlich gewinnt, wo die Library keinen guten
Pfad hat* (Ring-Shapes wie tall, large_k).

**Config-Cache end-to-end bestätigt:** `autotune.py` hat auf der GB10 alle 16 Shapes über die v2-Top-7
getunt und gecacht (`cache/tuned_configs.json`, Key inkl. GPU-Modell), 2. Aufruf jeweils Cache-Hit. Die
Top-7-Picks liegen bei ~95–100 % der Vollmessung — die Offline-Analyse ist auf echter Hardware bestätigt.

### Ranking-Modell-Studie: Roofline mit Auto-Selektor

Das reine Bandbreitenmodell (`bw`) rankt auf beiden Familien schlecht (Spearman ⌀ ~0), und das ist
*falsche Physik*: die GB10 hat 25 MB L2, fast alles bleibt resident, die Knappheit ist Compute/Occupancy,
nicht Bandbreite. Deshalb ein **Roofline-Modell** (`roofline` in `rank`): `max(memory_ms, compute_ms)` mit
`compute_ms = padded-FLOPs / (Tensor-Peak · util)`, wobei `util` die Wave-Quantisierung + Register-/SMEM-
Occupancy ist. Der Memory-Term ist **L2-bewusst** — passt der Working-Set einer Swizzle-Gruppe ins L2,
zählt nur das Kaltladen. Damit ist `max()` ein **automatischer, hardwaregetriebener Regime-Selektor**: auf
der L2-großen GB10 dominiert compute, auf einer bandbreitenlimitierten GPU memory. Alles liest aus
`DeviceProperties`; der einzige nicht auslesbare Wert ist `tensor_flop_per_sm_cycle` (Architektur-Schätzung,
setzt nur den Umschaltpunkt, nicht die Reihenfolge im Regime).

Erster Versuch: Spearman stieg (bw +0.03 → roofline +0.48), aber die **Top-7-Ausbeute fiel** (98 % → 66 %).
Diagnose (kein Code-Bug): im compute-Regime hängt `compute_ms` *nicht* von `m_l2/n_l2` ab — Occupancy und
FLOPs sind gruppen-unabhängig. Alle Configs mit gleichem `(m_prim,n_prim,k_prim)` bekommen denselben Score,
und der alte Tie-Break `-grid` schob dann die kleinen 64×64-Tiles nach oben (mehr Blöcke), die real langsamer
sind. Die einzige Größe, die `m_l2/n_l2` sieht, ist der Traffic-Term — den wirft `max()` im compute-Regime
weg. **Fix:** den Gleichstand über den worst-case-Traffic auflösen (holt das L2-Reuse-Signal zurück).

Ein zweiter Schritt macht die Physik dann ehrlich: die Grid-/Traffic-/FLOP-Schätzer bekommen die
A06-Batch-Faktoren (`a·c·b` fürs Grid/Traffic, `s` für die Compute-Arbeit) mitgereicht. Vorher war die
Ring-Familie fälschlich `memory`-gelabelt und wurde deshalb *zufällig* nach Traffic gerankt (was gut aussah);
jetzt ist sie korrekt `compute`-limitiert wie A05 — und *genau dadurch* fällt ihre Roofline-Top-7 (weil im
compute-Regime dieselbe `m_l2/n_l2`-Blindheit greift). Das ist der ehrlichere, nicht der schönere Wert.

Endergebnis über alle 16 Shapes (`analyze_tune.py`):

| Modell | Spearman | Top-7-Ausbeute |
|---|---|---|
| bw (Bandbreite, wie M3) | +0.03 | 83 % |
| v2 (bw + Register-Filter) | +0.38 | **97.8 %** |
| roofline (Traffic-Tie-Break, korrektes Regime) | **+0.50** | 85.5 % |

Das Fazit ist differenziert und wird durch den Regime-Fix *geschärft*: die Roofline ist der **bessere globale
Ranker** (höchste Korrelation, jetzt korrekt compute-limitiert auf allen Shapes), aber genau *weil* sie das
Regime richtig trifft, ist sie ein **schlechterer Top-k-Vorfilter** — im compute-Regime entscheiden L2-Reuse
und Tile-Effizienz (zweite Ordnung), und da hilft der Compute-Term nicht. **v2 bleibt der Default** (nutzt
direkt Traffic + Register-Filter, unberührt vom Regime-Fix). Die Roofline ist als dokumentierte, portablere
und physikalisch selbstumschaltende Variante drin.

Portabilität ist *by construction* (alles über `device_props` parametrisiert) und inzwischen auf einer
zweiten Karte (RTX 3070, 4 MB L2, `result_3070/`) gemessen. Absolute TFLOPS sind zwischen den Karten nicht
vergleichbar; der aussagekräftige Vergleich ist der Optimierungshebel (Speedup Tuner/Default): er wirkt auf
beiden, auf der 3070 stärker (Ø 1.88× vs. 1.36×), und die gemessen beste Config ist in **16/16 Shapes je
GPU verschieden** — das rechtfertigt GPU-spezifisches Tuning und den GPU-Modell-Key im Cache.

### Config-Cache

Damit sich das Tuning amortisiert (eine Shape einmal tunen, dann wiederverwenden), gibt es `autotuner/cache.py`
(JSON, Key = `einsum|shapes|GPU-Modell` — das GPU-Modell muss rein, weil die optimale L2-Gruppe von der
L2-Größe abhängt) und den praktischen Tuner `autotune.py`: `autotune(einsum, shapes, dev)` sieht erst im Cache
nach und misst sonst die v2-Modell-Top-7, cacht die schnellste und gibt sie zurück. `candidate_from_config`
baut aus den gecachten Knöpfen wieder einen `Candidate` für `run_candidate`. So kostet die erste getunte Shape
~3 s, jede weitere Nutzung 0 s.

## Offene Fragen (Auflösung)

Die Planungsfragen von M0 sind inzwischen weitgehend beantwortet:

- **Kostenmodell oder alles messen?** → *Geklärt:* der geprunte Raum bleibt klein (171–342) und der
  Compile dominiert (~0.4 s), also messen wir für die Ground Truth alles. Das Modell ist nur Vorfilter —
  Top-7 messen reicht für ~98 % des Optimums (M3/M4). Welches Modell dabei am besten vorfiltert, ist die
  Ranking-Studie: v2 (bw + Register-Filter) gewinnt, die Roofline rankt global besser (siehe M4).
- **Spezialisiert `ct.Constant` pro Wert?** → *Geklärt (M2):* ja, verifiziert auf der GB10, keine
  String-Templates nötig (`results/measure_compile.log`).
- **Padding sauber durchhalten?** → *Geklärt:* `dim_sizes` gepaddet, OOB über `PaddingMode.ZERO`, TFLOPS
  auf der Original-Shape. Auf echter Hardware bestätigt für die unteilbaren `krumm`-Shapes beider Familien
  (A05 und A06, dort inkl. Padding auf der p-Achse).
- **Scope-Disziplin gehalten?** → *Ja:* kein allgemeiner Tensor-Compiler. A06 ist als *zweite Familie*
  aufgegangen (eigener Ring-Kernel), ohne das Kernprojekt zu berühren.

Inzwischen ebenfalls erledigt:

- **Config-Cache** → *gebaut:* `autotuner/cache.py` + `autotune.py` (Key inkl. GPU-Modell). Siehe M4.
- **A06-Regime-Label** → *gefixt:* die Batch-Faktoren (`a·c·b`, `s`) laufen jetzt durch die Schätzer, die
  Ring-Familie wird korrekt als `compute`-limitiert gelabelt.

Cross-GPU (inzwischen erledigt):

- **Auf GB10 und RTX 3070 gemessen** (`result_3070/`). Der Tuner läuft auf beiden korrekt; der
  Optimierungshebel (Speedup Tuner/Default) wirkt auf beiden, auf der 3070 stärker (Ø 1.88× vs. 1.36×),
  und die beste Config ist in 16/16 Shapes je GPU verschieden. Bei jedem GPU-Wechsel neu messen — genau
  das erledigt der Cache (Key inkl. GPU-Modell) automatisch.
