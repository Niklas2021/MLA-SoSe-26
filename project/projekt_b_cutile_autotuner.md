# Projekt B — cuTile Auto-Tuner für Tensor-Kontraktionen

**Ziel:** Ein bewusst eingeschränkter Such- und Benchmarking-Loop, der für eine
gegebene Kontraktion (Einsum + Shapes) automatisch gute cuTile-Tiling-Configs
findet — statt sie wie in Assignment 05/06 von Hand herzuleiten.

**Pipeline:** `Einsum + Shapes → Basic Config → Kandidaten-Suchraum → Static Pruning
→ Ranking (Kostenmodell ODER messen) → Kernel-Instanziierung → verify + do_bench → Best Config (Top-k)`

**Basis:** Wir bauen auf `Config`/`Optimizer`/`generate_config` aus A05 auf
(`assignments/05_assignment/src/task1-3.py`) und nutzen die beiden bestehenden
Testfälle A05 (batched matmul, `kernel_l2` in `task4.py`) und A06 (Tensor-Ring) als Referenz.

> **Hinweis zur Genauigkeit:** Die folgenden Knöpfe/Schwellen stammen aus dem Pitch.
> Einige davon sind vermutlich zu optimistisch oder ungenau (siehe ⚠-Notizen und
> Abschnitt *Offene Fragen & Designentscheidungen* am Ende). Bewusst so dokumentiert,
> damit wir sie in M0/M1 verifizieren statt blind übernehmen.

---

## Roadmap

| Stufe | Inhalt | Eigenständiges Ergebnis |
|---|---|---|
| **M0** | Setup & Projektgerüst | Lauffähige Umgebung, A05-Code als importierbare Bibliothek, Baselines fixiert |
| **M1** | Config-Suchraum (Enumerator + Pruning + Ranking) | Liste gültiger, gerankter Kandidaten-Configs, noch ohne Kernel |
| **M2** | Kernel-Instanziierung (parametrisierter Kernel) | Eine aus Config erzeugte cuTile-Variante läuft korrekt (A05-Matmul) |
| **M3** | Benchmark & Ranking | `do_bench` + correctness + Top-k, Vergleich gegen Hand-L2, Tuning-Wall-Clock |
| **M4** | Transfer auf A06 | Tensor-Ring-Kontraktion als zweiter Test; optional Config-Cache |

**Minimal-Abschluss:** M0–M3 reproduzieren Assignment 05 (≥ 95 % der Hand-L2-Performance).
**Erweiterung:** M4 prüft Generalisierung auf Assignment 06 (deutlich anspruchsvoller, s.u.).

---

## M0 — Setup & Projektgerüst

*Saubere Ausgangsbasis schaffen, A05-Code wiederverwendbar machen, Referenzwerte einfrieren.*

- [ ] Projektstruktur: `project/src/` (Code), `project/results/` (Logs/Plots), `project/cache/` (optional Config-Cache).
- [ ] `task1-3.py` aus A05 als importierbares Modul übernehmen (z. B. `config.py`, `optimizer.py`),
      **ohne die Semantik zu ändern** — sie sind in A05 getestet.
- [ ] Smoke-Test der Umgebung: minimaler `ct.kernel` + `ct.launch` + `triton.testing.do_bench`
      lauffähig auf der Ziel-GPU; CUDA-/cuTile-Version + **GPU-Modell + L2-Größe** ins Log schreiben
      (L2-Größe brauchen wir fürs Kostenmodell, z. B. via `torch.cuda.get_device_properties`).
- [ ] Baselines fixieren: A05 Hand-L2 = 66.10 TFLOPS, Baseline = 38.60 TFLOPS, A06 Hand-cuTile = 49.84,
      `torch.einsum` = 16.18. Diese Zahlen stammen von **unserer** GPU — falls wir auf anderer Hardware
      tunen, **neu messen**, sonst sind die ≥95 %-Ziele bedeutungslos.

## M1 — Config-Suchraum (Enumerator + Pruning + Ranking)

*Suchraum hart einschränken, billig statisch vorfiltern, dann ranken — alles **vor** dem Kompilieren.*

### M1.1 Knöpfe / Suchachsen definieren
- [ ] Prim-Größen: `M_PRIM, N_PRIM ∈ {64,128,256}`, `K_PRIM ∈ {32,64,128}`.
- [ ] L2-Gruppengröße (Swizzle-Block): `M_L2, N_L2 ∈ {2,4,8}` — das ist die **zeitliche** Block-Gruppe,
      die L2-Reuse erzeugt (analog `group_size_m` im Triton-Matmul-Tutorial), **nicht** eine räumliche Kachel.
- [ ] Exec-/Reihenfolge-Muster: mind. die zwei aus A05 (Variante A = `m_l2/n_l2` als PAR/Swizzle;
      Variante B „strict" = `m_l2/n_l2` als SEQ-Loops). → Suchachse mit ~2 gültigen Mustern.
- [ ] ⚠ **„3·3·3·3 = 81" aus der Slide ist nur die Tile-Kombinatorik** (M·N·K-Prim × L2). Mit Exec-Mustern
      und ggf. asymmetrischem `M_L2 ≠ N_L2` wird es mehr. Realistische Größe in M1 messen, nicht raten.

### M1.2 Enumerator
- [ ] Funktion `enumerate_candidates(einsum, shapes) -> list[Config]`:
      Basic-Config via `generate_config`, dann je Kandidat `Optimizer.split_dim` (M,N → l2_outer/l2/prim;
      K → k_outer/k_prim) + `permute_dims` + `make_executable`/manuelles Exec-Setzen.
- [ ] Jeder Kandidat muss `Optimizer.verify()` bestehen (K nie PAR; Reihenfolge PAR|SEQ|PRIM; PRIM enthält M,N,K).
- [ ] ⚠ **Teilbarkeit:** `split_dim` wirft `ValueError`, wenn `outer*inner != size`. Krumme Shapes (z. B. M=1234)
      gehen nicht direkt → wir splitten auf der **gepaddeten** Größe `ceildiv(M, prim*l2)*prim*l2` und nullen
      OOB im Kernel via `PaddingMode.ZERO` (so macht es A05 schon im Launcher). **Designentscheidung dokumentieren:**
      `dim_sizes` der Config = gepaddete Größen.

### M1.3 Static Pruning (vor Compile)
- [ ] Heuristische Filter, die ohne Compile auswerten:
      MMA-Teilbarkeit (fp16-Tensor-Cores: K_PRIM Vielfaches von 16; M/N-PRIM Vielfaches von 16),
      Tile passt in Register/Shared-Memory-Budget, Akku-Tile `M_PRIM*N_PRIM*4 Byte` plausibel.
- [ ] ⚠ **Ehrlich:** Die genauen cuTile-internen SMEM/Register-Limits kennen wir nicht sicher.
      Pruning ist daher **Heuristik**, kein Beweis. Deshalb in M2/M3 jeden Compile in `try/except` kapseln
      und Compile-Fehler als „verworfen" loggen, statt uns allein auf den statischen Filter zu verlassen.
- [ ] Erwartete Reduktion ~81 → ~40–60 (Slide-Schätzung; in M1 real nachzählen).

### M1.4 Ranking
- [ ] **Primär: L2-Residency-Modell**, nicht „Arithmetic Intensity". Für eine Swizzle-Gruppe `M_L2×N_L2`:
      DRAM-Traffic ≈ wiederholtes Laden von A/B; Reuse = A über `N_L2`, B über `M_L2`.
      Working-Set pro k-Schritt ≈ `M_L2*M_PRIM*K_PRIM + N_L2*N_PRIM*K_PRIM` (×2 Byte fp16) — passt es in L2?
      Score = geschätzter DRAM-Traffic (kleiner = besser), Tie-Break Occupancy.
- [ ] ⚠ **Selbstkritik:** Reines AI-Modell (FLOPs/Byte) würde L2-Reuse gar nicht abbilden — der ganze
      Effekt hängt an der zeitlichen Scheduling-Reihenfolge der CTAs, nicht an FLOPs/Byte. Wenn das
      Residency-Modell zu wackelig wird: **Ranking weglassen und alle ~40–60 geprunten Kandidaten messen**
      (bei wenigen Sekunden/Compile durchaus machbar). Das Kostenmodell ist Optimierung, kein Muss.

## M2 — Kernel-Instanziierung (parametrisierter Kernel statt String-Codegen)

*Aus einer Config eine lauffähige cuTile-Variante machen — robust, nicht über fragile Text-Templates.*

- [ ] ⚠ **Ansatz-Korrektur ggü. Pitch („Codegen aus Templates"):** A05-Kernel nehmen die Tile-Größen
      bereits als `ct.Constant[int]`. Wir brauchen daher **keinen** String-Template-Generator, sondern
      **einen generischen Kernel**, den der JIT pro Konstanten-Kombination spezialisiert (analog Triton
      `constexpr`). Das ist deutlich robuster als Code-Strings zu erzeugen und zu `exec()`-en.
      → in M2 verifizieren, dass `ct.Constant` tatsächlich pro Wert neu spezialisiert.
- [ ] Generischer Kernel (Vorlage: `kernel_l2` / `kernel_l2_strict` in `assignments/05_assignment/src/task4.py`),
      parametrisiert über `M_PRIM,N_PRIM,K_PRIM,M_L2,N_L2` + Exec-Muster.
- [ ] **Config → Launch-Adapter** `build_launch(config) -> (kernel, grid, args)`:
      Grid-Größe, Padding-Buffer (`C_pad`, am Ende zurückslicen), pid-Zerlegung aus den `dim_sizes`/`exec_types`.
- [ ] Smoke-Test: aus der A05-Config erzeugter Kernel == handgeschriebener `kernel_l2` (identisches Ergebnis & ms).
- [ ] Scope-Grenze: zunächst **zwei-Input, GEMM-artige** Kontraktionen mit **einer** K-Dim und je einer M/N-Dim.

## M3 — Benchmark & Ranking

*Nur die Top-k messen, beste Config küren, und die Tuning-Kosten ehrlich ausweisen.*

- [ ] **Correctness:** generierter Kernel vs. `torch.einsum`-Referenz in fp32→fp16 gecastet,
      `torch.allclose(rtol=1e-2, atol=1e-1)` + `max_err` (wie A05 task4c, inkl. krummer Shapes).
- [ ] **Benchmark:** `triton.testing.do_bench` mit Warmup gegen JIT (A05 nutzt warmup=200, rep=2000),
      `flops = 2*∏(dim_sizes der Original-Shape)`, ms → TFLOPS.
- [ ] **Top-k Ranking** ausgeben; Vergleich gegen Hand-L2 (Ziel ≥ 95 % von 66.10 TFLOPS) und gegen Baseline.
- [ ] ⚠ **Tuning-Kosten messen, nicht nur Kernel-Laufzeit:** Das „Minuten statt Stunden"-Versprechen hängt
      an **Compile-Zeit × #Configs**, nicht an `do_bench`-ms. Daher loggen: Anzahl tatsächlich kompilierter
      Kandidaten, Gesamt-Wall-Clock des Tunings, Anteil Compile vs. Messung. Erst das belegt den Pruning-Nutzen.
- [ ] **Ablation:** welche Achse wirkt am stärksten (Prim-Größe vs. L2-Gruppe vs. Exec-Muster)? Top-k auswerten.
- [ ] ⚠ **Erwartungs-Check:** Wir reproduzieren bestenfalls die *handoptimierte* A05-Lösung — sie zu *schlagen*
      ist nicht garantiert, weil unser Suchraum bewusst klein ist und die Hand-Lösung schon nahe am Optimum liegt.
      Realistisches Minimalziel: ≥ 95 % erreichen und zeigen, dass es **ohne Handarbeit** gefunden wird.

## M4 — Transfer auf A06 (Erweiterung)

*Generalisierung auf eine schwierigere Kontraktion zeigen — bewusst als Stretch markiert.*

- [ ] Testfall `acspx,bspy→abcyx` aus A06 einspeisen.
- [ ] ⚠ **Das ist substanziell schwerer als der Pitch suggeriert:** A06 hat **zwei** Reduktionsdims (`s`,`p`)
      und **mehrere** M-/N-/Batch-artige Dims (a,c,x → M-Seite; b,y → N-Seite). Der generische Single-K-Kernel
      aus M2 deckt das **nicht** ab. Optionen: (a) die zwei K-Dims via `Optimizer.fuse_dims` zu einer fusionieren
      (nur wenn adjazent/contig — prüfen!), oder (b) einen zweiten Kernel-Typ mit verschachtelter K-Schleife.
      → Hier ist echtes Strukturmuster-Handling nötig; ggf. doch ein zweites Template.
- [ ] Ziel: ≥ 90 % der Hand-cuTile-Performance aus A06 (49.84 TFLOPS).
- [ ] **Optional Config-Cache** keyed by `(einsum, shapes, GPU-Modell)` → gefundene Best-Config wiederverwenden.
      ⚠ GPU-Modell muss im Key stehen, weil die optimale L2-Gruppe von der L2-Größe abhängt.

---

## Offene Fragen & Designentscheidungen (vor Implementierung klären)

1. **Kostenmodell oder Brute-Force?** Wenn der geprunte Raum klein bleibt (~40–60), ist „alle messen" robuster
   als ein wackeliges L2-Residency-Modell. Entscheidung in M1 nach realer Suchraumgröße treffen.
2. **`ct.Constant`-Spezialisierung** wirklich pro Wert? Falls nicht, müssen wir doch String-Templates bauen —
   das früh in M2 verifizieren (es entscheidet über den ganzen M2-Ansatz).
3. **Padding-Semantik:** `dim_sizes` der Config = gepaddete Größen, OOB via `PaddingMode.ZERO`. Konsistent halten,
   damit `flops`/TFLOPS auf der **Original**-Shape gerechnet werden, nicht auf der gepaddeten.
4. **Static-Pruning-Limits** (SMEM/Register) sind unsicher → Compile-Fehler immer abfangen, statt blind zu vertrauen.
5. **Hardware-Abhängigkeit:** Alle Zielprozente beziehen sich auf die GPU, auf der die Baselines gemessen wurden.
   Bei GPU-Wechsel neu baseline-n.
6. **Scope-Disziplin:** Kein allgemeiner Tensor-Compiler. M0–M3 = ein K-Dim, zwei Inputs. A06 (M4) ist Stretch
   und darf scheitern, ohne das Kernprojekt zu gefährden.

## Erwartete Insights

- Performance steckt in der **Konfiguration**, nicht nur im Kernel-Code.
- Trade-off **L2-Reuse vs. Parallelität** wird über die Top-k-Configs sichtbar.
- Zweistufige Suche (statisch prunen → nur Top-k kompilieren/messen) macht Auto-Tuning praktikabel —
  **belegt über gemessene Tuning-Wall-Clock**, nicht nur behauptet.
