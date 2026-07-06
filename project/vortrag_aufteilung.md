# Vortrag: cuTile Auto-Tuner — Folienplan

Ziel des Vortrags: nicht die Projektidee nochmal pitchen, sondern zeigen **was konkret umgesetzt
wurde** (Pipeline, Kernel, Filter), **wie es final funktioniert** und **wie gut es in der Evaluation
abschneidet** (gegen Handkernel, gegen torch/cuBLAS, über Shapes und GPUs hinweg). Die Problemstellung
kommt nur als kurzer Recap.

## Rahmen

- **Autoritative Zahlenquelle:** `project/result_dgx/study.log` (GB10, do_bench warmup=50 rep=200) —
  als einzige Quelle enthält sie sowohl Default, Tuner-Best **und** `torch.einsum` für A05 **und** A06.
  Alle Balkendiagramme ziehen aus den `tune_*.csv` in `result_dgx/`. Ranking-Zahlen aus `analyze_tune.py`.
  Wo Werte zwischen Runs leicht schwanken (z. B. a06-Referenz 59.8/61.9/62.3), nehmen wir konsequent
  `study.log` und erwähnen die ~2 % Messstreuung.
- **Achtung Baseline-Begriffe:** „Default" = die naive 8×8-Config (`DEFAULT_CONFIG`, aus A05 übernommen,
  das was man *ohne* Tuner nähme). „Handkernel A06" = die von Hand getunte A06-Lösung (49.84 TFLOPS aus
  dem Assignment). Nicht verwechseln — der 2.29×-„Gewinn" auf A06 ist gegen die *mismatchte* 8×8-Default,
  nicht gegen den echten Handkernel (+24 %).
- **RTX 3070: Daten stehen noch aus.** Der Cross-GPU-Teil ist bewusst **eine** Folie (Folie 14) und nicht
  tragend — der Vortrag steht komplett auf den GB10-Ergebnissen. Kommt die 3070 rechtzeitig, füllt sie den
  Platzhalter; kommt sie nicht, fällt nur diese eine Folie weg.
- **Team-Testfälle:** A05 = batched Matmul `cmk,ckn->cmn`; A06 = Tensor-Ring `acspx,bspy->abcyx`.

## Foliendesign (Best Practice — gilt für alle Folien)

- **Eine Aussage pro Folie.** Die Kernaussage ist die Überschrift, der Rest belegt sie.
- **Bild-first, nicht Textwüste.** Die Grafik trägt die Folie; Text sind höchstens ~5 kurze Anker
  (wenige Wörter), keine ganzen Sätze. Ausnahme: ein wörtliches Zitat.
- **Details in die Speaker-Notes**, nicht auf die Folie. Der Abschnitt „Inhalt" unten ist v. a.
  Notiz-Material (das, was gesagt wird) — auf die Folie kommt nur „Zu sehen".
- Zahlen sparsam und groß; nie eine ganze Tabelle vorlesen. Farben/Legende konsistent (Tuner = Blau,
  Default = Grau, torch = Orange, Handkernel = dunkles Grau — wie in den Figures).
  Farben sind auf Rot-Grün-Schwäche geprüft (dataviz-Validator: Blau/Orange ΔE 96.7,
  Blau/Rot 75.7); die diverging-Grafik codiert „wer gewinnt" zusätzlich über Links/Rechts.

## Aufteilung

- **Person 1 — Folien 1–7 (Problem, Umsetzung, Architektur), ca. 7–9 min.**
- **Person 2 — Folien 8–16 (Evaluation, Ergebnisse, Einordnung), ca. 8–11 min.**
- A06 auf beide verteilt: Person 1 erklärt *wie* erweitert wurde (Folie 7), Person 2 *wie gut* das misst
  (Folie 11). Falls Person 2 zu voll ist, kann Person 1 Folie 13 (Ranking-Modell) übernehmen — passt
  technisch zur Umsetzung.

## Grafik-Übersicht (alles aus vorhandenen Daten erzeugbar)

| ID | Folie | Typ | Datenquelle |
|---|---|---|---|
| G1 | 3 | Pipeline-Flussdiagramm (schematisch) | — (selbst gezeichnet) |
| G2 | 2 | Tiling/L2-Reuse-Schema | — (selbst gezeichnet) |
| G3 | 6 | Suchraum-Trichter 486→342→7 + Prune-Gründe | `search.py` (Zahlen), study.log |
| G4 | 9 | A05-Balken je Regime: Default / Tuner / torch | `tune_*.csv` (A05) |
| G5 | 10 | Tuner/torch-Verhältnis A05 vs A06 (diverging um 1.0) | study.log |
| G6 | 11 | A06-Balken je Regime + Referenz-„Leiter" | `tune_a06*.csv`, study.log |
| G7 | 12 | Top-k-Kurve: % vom Optimum über Mess-Budget | `tune_*.csv` + `rank()` |
| G8 | 13 | Ranking-Modelle: Spearman vs Top-7-Ausbeute | analyze_tune.py |
| G9 | 14 | Cross-GPU Balken GB10 vs RTX 3070 (Platzhalter) | study.log (+ 3070 TBD) |
| C1 | 5 | Code-Snippet: `matmul_variant_a` (ct.Constant + Swizzle) | `kernels.py` |
| C2 | 7 | Code-Snippet: `matmul_ring_a` (Batch-Decode + permute) | `kernels.py` |
| C3 | 6 | Code-Snippet: `prune_reason` (4 Filter) | `search.py` |

---

# PERSON 1 — Problem, Umsetzung, Architektur

## Folie 1: Recap — das Problem & unser Ziel

**Kernaussage:** In A05/A06 haben wir die L2-optimale Aufteilung *von Hand* hergeleitet. Das
skaliert nicht — für jede neue Kontraktion, Shape oder GPU müsste man neu nachdenken. Das
automatisieren wir.

**Inhalt:**
- A05: L2-optimale Tiling-Config für `cmk,ckn->cmn` von Hand begründet. A06: dasselbe nochmal für die
  schwierigere Ring-Kontraktion.
- Kernbeobachtung (aus der Vorlesung): *die Performance steckt in der Config, nicht im Kernel-Code.*
- Ziel: **Eingabe** Einsum-String + Shapes → **Ausgabe** gute cuTile-Tiling-Config, automatisch gemessen.
- Scope-Ansage: kein allgemeiner Tensor-Compiler. Zwei Struktur-Familien (GEMM + Ring), Configs getunt.

**Zu sehen:** die zwei Testfälle nebeneinander (A05-GEMM-Einsum, A06-Ring-Einsum) mit ihren Shapes.

**Grafik:** klein/schematisch — „von Hand (Nachdenken pro Shape/GPU) → automatisch (Such-Loop)". Kann
ein simples 2-Spalten-Bild sein, keine Daten nötig.

## Folie 2: Was ist überhaupt eine Tiling-Config? (Vorlesungs-Recap)

**Kernaussage:** Eine Config = Tile-Größen + Ausführungsreihenfolge. Der L2-Reuse entsteht durch die
*zeitliche* Block-Gruppe (`m_l2 × n_l2`), nicht durch eine räumliche Kachel.

**Inhalt:**
- Prim-Tiles `M_PRIM / N_PRIM / K_PRIM`: die mma-Kachel, die ein CTA rechnet.
- Super-Tile / L2-Gruppe `M_L2 × N_L2`: benachbarte Blöcke, die zeitlich nah laufen → A-Zeile und
  B-Spalte bleiben im L2 (Gegenstück zu `group_size_m` im Triton-Matmul-Tutorial).
- Exec-Typen PAR / SEQ / PRIM und die Reihenfolge-Regel (PAR | SEQ | PRIM, K nie PAR).
- Zwei Varianten: **A** = `m_l2/n_l2` als PAR (Swizzle über die Block-ID), **B** = als SEQ-Loops im CTA.

**Zu sehen:** das M×N-Ausgabegitter, in Prim-Tiles gekachelt, eine `m_l2×n_l2`-Gruppe hervorgehoben mit
Pfeilen „A-Zeile wiederverwendet über N_L2 / B-Spalte über M_L2".

**Grafik G2:** selbst gezeichnetes L2-Reuse-Schema (die Vorlesungsfolie 33 als Vorlage). Statisch.

## Folie 3: Die Pipeline — wie der Tuner final arbeitet

**Kernaussage:** Ein fester Ablauf von Einsum bis gecachte Best-Config.

**Inhalt (die Module beim Namen nennen):**
- `generate_config` → Basic-Config aus Einsum + Shapes (Dim-Typen, Strides).
- `enumerate_candidates` → Suchraum aufspannen.
- `prune` → statisch filtern (ohne Kompilieren).
- `rank` → Kostenmodell sortiert die Kandidaten.
- `tune.py` → Top-k (oder alle) kompilieren, gegen `torch.einsum` prüfen, mit `do_bench` messen.
- `cache.py` / `autotune.py` → Best-Config cachen (Key = Einsum + Shapes + GPU-Modell).

**Zu sehen:** der 6-Stufen-Fluss als Diagramm, mit „reines Python (ohne GPU)" für enumerate/prune/rank
und „braucht GPU" für compile/bench markiert.

**Grafik G1:** horizontales Flussdiagramm
`Einsum+Shapes → generate → enumerate (486) → prune (342) → rank → compile+verify+bench (Top-7) → Best → Cache`.
Die Stufen-Zahlen (486/342/7) als Badges dranschreiben.

## Folie 4: Der Suchraum — die Knöpfe

**Kernaussage:** Bewusst klein gehaltener, hardware-sinnvoller Raum: 486 Kandidaten.

**Inhalt:**
- Knöpfe: `M_PRIM, N_PRIM ∈ {64,128,256}`, `K_PRIM ∈ {32,64,128}`, `M_L2, N_L2 ∈ {2,4,8}`, Variante `{A,B}`.
- `3·3·3·3·3·2 = 486` (nicht die „81" aus dem Pitch — die zählten nur Tile-Kombis, ohne asymmetrisches
  `m_l2 ≠ n_l2` und ohne die zweite Variante).
- Akzeptanztest, der den ganzen Ansatz absichert: **die handoptimierte A05-Config (128/128/64, 8×8, A)
  muss im enumerierten Set sein** — sonst könnte der Tuner sie nie finden. Sie ist drin. ✓
- Krumme Shapes: `split_dim` verlangt exakte Teilbarkeit → wir padden hoch (`ceildiv`) und nullen den
  Überhang im Kernel (`PaddingMode.ZERO`); TFLOPS rechnen wir auf der Original-Shape.

**Zu sehen:** die Knopf-Tabelle + die Rechnung `3·3·3·3·3·2 = 486` prominent + ein grüner Haken
„Hand-Config ∈ Suchraum".

**Grafik:** kleines Code-Snippet der `SearchSpace`-Choices (search.py Zeilen 17–22) daneben, sonst Text.

## Folie 5: Kernel-Umsetzung — ein generischer Kernel, kein String-Codegen

**Kernaussage:** Wir erzeugen **keine** Kernel-Strings per `exec()`. Ein einziger generischer cuTile-Kernel
pro Variante, Tile-Größen als `ct.Constant[int]` → der JIT spezialisiert pro Wert (wie Triton `constexpr`).

**Inhalt:**
- Die zentrale Design-Entscheidung: String-Templates wären fragil; `ct.Constant` reicht. Auf der GB10
  verifiziert, dass die Spezialisierung wirklich pro Wert passiert (`measure_compile.log`, ~0.4 s/Compile).
- Variante A: `m_l2/n_l2` über die Block-ID dekodiert (Swizzle) — der eigentliche Performance-Kandidat.
- Variante B: `m_l2/n_l2` als SEQ-Loops im CTA (weniger CTAs).
- Korrektheit immer gegen `torch.einsum` (`allclose`, rtol=1e-2/atol=1e-1).

**Zu sehen:** der Kernel-Code, mit den `ct.Constant`-Parametern und dem `ct.mma`-Loop hervorgehoben.

**Grafik C1:** Code-Snippet `matmul_variant_a` aus `kernels.py` (Z. 13–48), gekürzt auf: Signatur mit
`ct.Constant`, den Block-ID-Swizzle-Decode (Z. 23–35), den `ct.load(...PaddingMode.ZERO)` + `ct.mma`-Loop.
Zwei, drei Zeilen farbig markieren: `M_PRIM: ct.Constant[int]` und `acc = ct.mma(a_tile, b_tile, acc)`.

## Folie 6: Static Pruning — den Raum billig eingrenzen (und wie weit wirklich)

**Kernaussage:** Vier Filter werfen offensichtlich Unsinniges *vor* dem Kompilieren raus. Aber ehrlich:
statisch lässt sich weniger beschneiden als erhofft — der Rest muss gemessen werden.

**Inhalt — die 4 Filter (vom Härtesten zum Weichsten):**
1. **MMA-Teilbarkeit** (Guard): Prim-Größen Vielfache von 16 für die fp16-Tensor-Cores.
2. **SMEM-Budget** (der harte Filter): Operand-Tiles × Double-Buffering gegen nutzbares Shared Memory
   (GB10: 101376 − 1024 ≈ 100 KB).
3. **Akku-Register**: `M_PRIM·N_PRIM` fp32 gegen die halbe Registerdatei.
4. **Padding-Verschwendung**: gepaddetes Volumen gegen Original (Faktor > 8 raus).

**Wie weit es *tatsächlich* eingrenzt (die ehrliche Pointe):**
- A05: **486 → 342** (144 verworfen: **126 SMEM, 18 Akku-Register**; MMA und Padding greifen bei 4096
  nicht, weil glatt teilbar).
- Warum nicht mehr: das SMEM hängt **nur an den Prim-Größen**, nicht an `m_l2/n_l2` oder der Variante.
  Statisches Pruning kann diese beiden Achsen also gar nicht anfassen — sie bleiben für die Messung.
- **Die L2-Reuse-Regel aus der Vorlesung greift auf der GB10 nicht:** Working-Set der größten Gruppe
  ~256 KB gegen **25 MB L2**. Das L2 ist zu groß, um die Gruppengröße einzuschränken → die Entscheidung
  über `m_l2/n_l2` und Variante verschiebt sich komplett auf die Messung. (Genau das motiviert Folie 12/13.)

**Zu sehen:** der Trichter 486 → 342 → (später) 7, daneben die Aufschlüsselung der 144 Verworfenen.

**Grafik G3:** Trichter/Balken `486 → 342 → 7` (drei Stufen: enumeriert / geprunt / gemessen-Top-7), plus
ein kleiner gestapelter Balken der Prune-Gründe (126 SMEM + 18 Register). **Grafik C3:** das `prune_reason`
(search.py Z. 184–193) als 4-Zeilen-Code-Snippet daneben, damit die Filter konkret sind.

## Folie 7: Erweiterung auf A06 — eine zweite Struktur-Familie

**Kernaussage:** A06 ist nicht „nur doppeltes K". Das eigentliche Problem ist die **Batch-Topologie** —
und die verlangt einen *zweiten Kernel-Typ*, keinen Umbau.

**Inhalt:**
- A06-Ring `acspx,bspy->abcyx`: zwei Reduktionen (`s`, `p`) und mehrere Output-Dims.
- Der Knackpunkt: A05 hat einen **geteilten** Batch (`c` in A, B und C), A06 hat **unabhängige** Batches
  (`a,c` nur in A, `b` nur in B). Der A05-Kernel indiziert A und B mit demselben `c_idx` — A06 kann das
  gar nicht ausdrücken; zwänge man es durch, kämen alle `a×b`-Kombinationen statt der Diagonale.
- Deshalb: ein **neuer Kernel-Typ** (Ring-Kernel), kein Umbau des bestehenden. Passt zur Philosophie —
  der Tuner automatisiert die **Config-Suche**, nicht das Kernel-Schreiben (genau wie cuBLAS/CUTLASS eine
  endliche Template-Menge tunen, oder Triton `autotune` pro `@jit`).
- Umsetzung: `parse_einsum` verallgemeinert auf `m/n/k_chars` + `extra_m` (`a,c`→PAR-Batch), `extra_n`
  (`b`), `seq_k` (`s`→SEQ-Loop). **Der Single-M/N/K-Pfad (A05) bleibt bitgleich** (486→342, Hand-Config
  überlebt). Enumerator A06: 243 Kandidaten (nur Variante A) → Pruning → **171**.
- Der Ring-Kernel braucht einen Per-Tile-`permute` (Layout A/B/C ist nicht mma-fertig) und die äußere
  SEQ-Schleife über `s`.

**Zu sehen:** die A05-vs-A06-Batch-Topologie (geteiltes `c` vs. unabhängige `a,c`/`b`) + der Ring-Kernel.

**Grafik C2:** Code-Snippet `matmul_ring_a` (kernels.py Z. 125–170), gekürzt auf: den erweiterten
Block-ID-Decode (`a_idx/b_idx/c_idx`, Z. 139–146), die äußere `for s_it`-Schleife (Z. 152) und den
`ct.permute` (Z. 159, 168). Dazu ein Mini-Schema „shared batch (A05) vs. independent batches (A06)".

---

# PERSON 2 — Evaluation, Ergebnisse, Einordnung

## Folie 8: Benchmark-Setup & Baselines

**Kernaussage:** Wir messen fair gegen drei Referenzen und prüfen jede Config auf Korrektheit.

**Inhalt:**
- Gemessen wird pro Shape: **getunter Kernel** vs. **Default (8×8)** vs. **torch.einsum (cuBLAS)** —
  bei A06 zusätzlich gegen den **Handkernel** (49.84). fp16 rein, fp32 akkumuliert.
- Korrektheit: jede Config gegen `torch.einsum` (`allclose`). Ergebnis: **8×342 (A05) + 8×171 (A06),
  alle korrekt, 0 Fehlschläge** — inkl. der unteilbaren `krumm`-Shapes (Padding-Pfad auf echter HW bestätigt).
- Acht Shape-Regime pro Familie: `square / tall / wide / small_k / large_k / krumm / batch` (+ Referenz).
- GPU: **NVIDIA GB10 (DGX Spark)** — 48 SMs, **25 MB L2**, integrierter LPDDR (teilt sich Speicher mit
  CPU), CC 12.1, cuTile 1.4.0. (RTX 3070: folgt, Folie 14.)

**Zu sehen:** die Hardware-Faktenbox (48 SMs / 25 MB L2 / integriert) + die Regime-Tabelle mit Shapes +
die „alle korrekt, 0 Fehlschläge"-Zahl prominent.

**Grafik:** kein Plot nötig — Faktenbox + Regime-Tabelle reichen. Optional ein kleiner „0 / 4104
Fehlschläge"-Badge.

## Folie 9: A05-Ergebnisse — der Tuner bestätigt (und schlägt punktuell) die Handarbeit

**Kernaussage:** Auf regulären GEMMs holt der Tuner nichts heraus (1.00×) — er *bestätigt* das Handtuning.
Wo die feste 8×8-Gruppe schlecht passt, gewinnt er klar.

**Inhalt (Zahlen aus study.log):**
- Tuner ⌀ **1.03×** über Default, **nie schlechter**, erreicht **97.6 %** des absoluten Optimums.
- Regulär (square/tall/wide/batch): 1.00–1.02×. `large_k`: **+7 %** (will `k_prim=128`). `krumm`:
  **+58 %** (41.8 vs 26.6 — die 8×8-Gruppe paddet die unteilbare Shape schlecht).
- Robuste Quasi-Universal-Config: `128/128/64` gewinnt 7 von 8, nur `large_k` will `k_prim=128`. Die
  asymmetrischen 256-breiten Tiles sind Gift (Registerbudget) — 3–14 statt ~65 TFLOPS.

**Zu sehen:** Balken pro Regime, Default vs Tuner-Best (vs torch als dritte Referenz), krumm hervorgehoben.

**Grafik G4:** gruppiertes Balkendiagramm, x = 8 A05-Regime, y = TFLOPS, drei Balken je Gruppe
(Default / Tuner-Best / torch.einsum) aus `tune_*.csv` + study.log. Über jeder Gruppe klein der
Tuner/Default-Faktor. `krumm` (1.58×) farblich betonen.

## Folie 10: Die ehrliche Baseline — Tuner vs. cuBLAS / torch.einsum

**Kernaussage:** Auf reinem GEMM **gewinnt cuBLAS** — und das ist okay. Der Wert des Tuners ist nicht
„schneller als alles", sondern *ein allgemeiner Mechanismus, der ohne Handarbeit über beliebige
Kontraktionen brauchbare Leistung liefert und dort klar gewinnt, wo die Library keinen guten Pfad hat*.

**Inhalt:**
- **GEMM (A05): cuBLAS vorn.** Über die 8 GEMM-Shapes erreicht der Tuner im Schnitt ~77 % von torch
  (geom. Mittel); nur auf der hand-getunten a05-Referenz zieht er knapp vorbei (1.04×). Erwartbar —
  cuBLAS ist eine gereifte GEMM-Library, sie zu schlagen war nie das Ziel.
- **Ring (A06): stark shape-abhängig, im Mittel leicht vorn (~1.17× geom.).** torch findet bei manchen
  Ring-Shapes einen guten Pfad (→ bmm), bei anderen nicht. Der Tuner gewinnt groß, wo torchs Pfad
  schlecht ist: `a06_tall` **3.95×**, `a06_large_k` **2.65×**; verliert, wo er gut ist: `a06_krumm` 0.38×.
- Ehrliche Korrektur: der Assignment-Wert `A06_TORCH_EINSUM = 16.18` ist veraltet — dieselbe Referenz-Shape
  macht mit aktuellem torch-fp16 **60.22 TFLOPS**, gleichauf mit unserem Tuner (~60).

**Zu sehen:** wie sich Tuner/torch um die 1.0×-Linie verteilt — A05 alle knapp darunter, A06 breit gestreut.

**Grafik G5:** diverging Bar um 1.0×, x = Shapes (A05-Block + A06-Block getrennt), y = Tuner/torch-Verhältnis
(log-Skala hilft: 0.38× … 3.95×). Linie bei 1.0×. Daten direkt aus study.log (die „Tuner Nx"-Zeilen).

## Folie 11: A06-Ergebnisse — Transfer gelingt, Tuner schlägt den Handkernel

**Kernaussage:** Der Ansatz transferiert sauber auf die zweite Familie, alle Configs rechnen korrekt, und
der Tuner schlägt den **Handkernel** — weil er eine Achse mitdurchsucht, die von Hand nicht angefasst wurde.

**Inhalt (Referenz-Shape, study.log):**
- Ehrliche Leiter: **Default (8×8) 26.3 → Handkernel 49.84 → Tuner ~60 (59.8–61.9) ≈ torch 60.2.**
- Der 2.29×-„Gewinn" ist gegen die *mismatchte* 8×8-Default (aus A05 übernommen, passt auf A06 schlecht);
  gegen den echten Handkernel sind es **+24 %**.
- Woher der Gewinn: `k_prim=32`. Der Handkernel nahm `p=64` als *einen* mma-Tile, der Tuner teilt die
  p-Reduktion in zwei 32er-Kacheln → ~14 % schneller.
- Über alle 8 Ring-Shapes: Tuner/Default **1.10–2.47×** (größer als bei A05, weil die 8×8-Default auf
  A06 durchweg schlecht sitzt). Alle 171 Configs je Shape korrekt, inkl. Padding auf x, y **und** p.

**Zu sehen:** Balken je Ring-Regime (Default vs Tuner) + die 4-stufige „Leiter" für die Referenz-Shape.

**Grafik G6:** (a) gruppiertes Balkendiagramm x = 8 A06-Regime, Default vs Tuner-Best (`tune_a06*.csv`);
(b) daneben die Referenz-„Leiter" als 4 Balken: 26.3 / 49.84 / 60 / 60.2 mit Beschriftung
Default / Hand / Tuner / torch.

## Folie 12: Wie oft findet der Tuner das Optimum? — Top-k-Stufen

**Kernaussage:** Ground Truth ist die *volle* Messung. Frage: zieht das Modell die real beste Config in
seine Top-k? Antwort: als *exakter* Treffer meist nicht, als *Vorauswahl* fast immer — mit klaren Stufen.

**Inhalt (die drei Betriebspunkte):**
- **Top-7 / ~3 s / ≥ 95 %** (im Schnitt 97.6 % des Optimums) — der Sweet Spot. Nur 7 statt 342 kompilieren
  (Compile ~0.4 s dominiert die Zeit).
- **Top-~90 / ~45 s / ≥ 99 %** — Mittelstufe (worst case `small_k`, sonst top-20–40).
- **Voll-Sweep / ~3 min / 100 %** — Ground Truth.
- Ehrlich: die *exakt* Beste sitzt im Modell tief und liegt ohnehin innerhalb ~3 % im Messrauschen; top-7
  erwischt sie meist nicht — kostet aber praktisch nichts an Performance.

**Zu sehen:** eine Kurve „% vom Optimum" gegen das Mess-Budget (k bzw. Sekunden), mit den drei Stufen
markiert und einer 95 %- und 99 %-Linie.

**Grafik G7:** Stufen-/Liniendiagramm — x = Anzahl gemessener Kandidaten k (log, 1…342), y = erreichter
Anteil am Optimum (% vom abs. Best), Kurve = `max(TFLOPS der Modell-Top-k)/absBest`, über die Shapes
gemittelt (±Band). **Aus den CSVs + `rank()` direkt berechenbar.** Vertikale Marker bei k=7 (→95 %) und
k≈90 (→99 %); horizontale 95 %/99 %-Linien. Eine zweite x-Achse „≈ Sekunden" (k·0.4 s) ist ein Bonus.

## Folie 13: Das Ranking-Modell — besserer Ranker ≠ besserer Vorfilter

**Kernaussage:** Wir haben drei Kostenmodelle verglichen. Das Ergebnis ist nicht-trivial: das Modell mit
der besten Korrelation ist *nicht* der beste Top-k-Vorfilter.

**Inhalt:**
- **bw** (reine Bandbreite): rankt schlecht (Spearman ⌀ **+0.03**) — falsche Physik, denn die GB10 hat
  25 MB L2, fast alles bleibt resident, die Knappheit ist Compute/Occupancy, nicht Bandbreite.
- **v2** (bw + Register-Filter, wirft die 256-breiten Register-Fresser raus): Spearman **+0.38**,
  **97.8 %** Top-7-Ausbeute → **bleibt Default**.
- **roofline** (`max(memory_ms, compute_ms)`, L2-bewusst, schaltet das Regime *selbst* um): bester
  *globaler* Ranker (Spearman **+0.50**), aber **schlechterer Vorfilter** (85.5 % Top-7) — *gerade weil*
  er das Compute-Regime richtig trifft, und da entscheiden L2-Reuse/Tile-Effizienz zweiter Ordnung, die
  der Compute-Term nicht sieht.
- Take-away: Korrelation ≠ Vorfilter-Güte. Für die Praxis zählt die Top-7-Ausbeute → v2.

**Zu sehen:** die drei Modelle mit ihren zwei Metriken direkt gegenübergestellt.

**Grafik G8:** Scatter — x = Spearman (−0.1…+0.6), y = Top-7-Ausbeute (%), drei Punkte beschriftet
(bw / v2 / roofline). Zeigt visuell: roofline rechts (beste Korrelation) aber unter v2 (beste Ausbeute);
v2 oben. Alternativ ein 2er-Balkenpaar je Modell. Zahlen aus `analyze_tune.py`.

## Folie 14: Cross-GPU — ist Tuning GPU-abhängig? (Platzhalter RTX 3070)

**Kernaussage:** Der Tuner soll nicht auf eine GPU overfitten. Weil die optimale L2-Gruppe von der
L2-Größe abhängt, ist die GPU im Cache-Key — und wir erwarten *andere* Gewinner-Configs auf anderer HW.

**Inhalt:**
- Motivation: GB10 = 25 MB L2, integriert, eher DRAM-bandbreitenlimitiert; RTX 3070 = diskret, kleines
  L2, klassisch bandbreitenlimitiert. Erwartung: das Roofline-Modell schaltet auf `memory` um, und die
  besten `m_l2/n_l2`/Prim-Wahlen verschieben sich → Argument für **GPU-spezifisches** Autotuning.
- Portabilität ist *by construction* (alles liest aus `device_props`), aber bisher nur auf der GB10
  validiert. Der eigentliche Cross-GPU-Test steht mit der 3070 aus.
- **[RTX-3070-Ergebnisse hier einsetzen, sobald da:]** gewinnende Config je Shape, unterscheidet sie sich
  von der GB10? Findet der Tuner auch dort korrekte Configs? Wie weit ist Top-7 vom Optimum?

**Zu sehen:** GB10 vs RTX 3070 nebeneinander, gewinnende Config + TFLOPS je Shape.

**Grafik G9:** gruppiertes Balkendiagramm GB10 vs RTX 3070 je Shape (Tuner-Best) — **RTX-3070-Balken als
gestrichelter Platzhalter „TBD"** anlegen, damit die Folie ohne Daten steht und beim Nachliefern nur die
Werte rein müssen. Datenquelle GB10: study.log; RTX 3070: folgt.

## Folie 15: Praktische Einordnung — Cache & Amortisierung

**Kernaussage:** Tuning lohnt sich über Wiederverwendung. ~3 s Einmalkosten amortisieren sich in Sekunden,
sobald dieselbe Shape oft läuft — genau der Normalfall in Training/Inferenz.

**Inhalt:**
- Eine Kontraktion läuft in ~8 ms; 3 s Tuning sind ein Einmalposten. Break-even bei ~2–3 % Gewinn: wenige
  Tausend Aufrufe; bei Ausreißer-Shapes (`krumm`, +58 %) schon wenige Hundert.
- Im Netz sind Layer-Dims fix → jeder Trainings-Step dieselben Matmuls → millionenfach dieselbe Shape.
  Genau deshalb tunen cuBLAS, Triton (`autotune`), torch.compile pro Shape einmal und cachen.
- Unser Config-Cache: `autotuner/cache.py` + `autotune.py`, Key = **Einsum + Shapes + GPU-Modell** (GPU
  muss rein, weil die optimale L2-Gruppe an der L2-Größe hängt). End-to-end bestätigt: 16 Shapes getunt
  und gecacht, 2. Aufruf jeweils Cache-Hit, Top-7-Picks bei ~95–100 % der Vollmessung.
- Nicht auf geht die Rechnung nur bei stark dynamischen Shapes → in der Praxis Bucketing/Padding.

**Zu sehen:** die Amortisierung + der Cache-Ablauf (1. Aufruf misst & cacht, ab dem 2. Cache-Hit = 0 s).

**Grafik (optional):** kumulierte Zeit über N Aufrufe: „ungetunt" (linear) vs „getunt+gecacht"
(3 s Offset, dann flacher wegen der ~2–3 % schnelleren Kernel) — Break-even-Punkt markiert. Kann auch als
simples Cache-Ablaufschema statt Plot.

## Folie 16: Fazit

**Kernaussage:** Die Eingrenzung (enumerate + prune) ist das Sichere und Wertvolle; das analytische
Ranking taugt auf dieser Hardware nur grob als Vorauswahl — **die Entscheidung muss man messen**.

**Inhalt:**
- **A05:** Tuner reproduziert Handtuning (⌀ 1.00–1.03×), nie schlechter, gewinnt wo die feste Gruppe
  schlecht passt (`krumm` +58 %).
- **A06:** sauberer Transfer auf eine zweite Familie (ein zusätzlicher Kernel-Typ), alle Configs korrekt,
  schlägt den Handkernel (+24 %).
- **Ehrliche Einordnung:** cuBLAS schlagen wir auf GEMM nicht — der Wert ist *Allgemeinheit ohne
  Handarbeit* und klare Gewinne, wo die Library keinen guten Pfad hat (Ring `tall`/`large_k`).
- **Modell:** analytisches Ranking allein reicht nicht; v2 ist der beste Vorfilter (97.8 % @ top-7),
  roofline der bessere globale Ranker — messen bleibt entscheidend.
- **GPU-spezifisch + gecacht** ist der eigentliche Praxiswert (Cross-GPU by construction, auf 3070 noch
  zu bestätigen).

**Zu sehen:** die drei, vier Kernaussagen als Badges; kein neuer Plot (ggf. G4/G6 im Kleinformat).

---

## Timing-Empfehlung

- **15 min:** Person 1 Folien 1–7 (~7 min), Person 2 Folien 8–16 (~8 min). Folie 2 und 12 knapp halten.
- **20 min:** Person 1 ~9 min, Person 2 ~11 min; Folie 12/13 (Top-k + Ranking) ausführlicher.
- Puffer: Folie 14 (Cross-GPU) fällt weg, falls RTX 3070 nicht rechtzeitig da ist — dann Folie 10
  (cuBLAS) und 13 (Ranking) etwas ausbauen.
