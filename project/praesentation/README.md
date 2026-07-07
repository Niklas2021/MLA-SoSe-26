# Präsentation — cuTile Auto-Tuner

Ordner für die Mittwochs-Präsentation. Der Folienplan liegt eine Ebene höher in
`../vortrag_aufteilung.md`.

## Das Deck

`cuTile_Auto-Tuner.pptx` (16:9, 19 Folien) wird von `build_pptx.py` erzeugt (python-pptx).
Layout ist deterministisch: feste Positionen in Zoll, Bilder werden per Seitenverhältnis in
ihre Box zentriert (nie verzerrt). Detailinfos stehen in den **Speaker-Notes**, nicht auf den
Folien. Kurze Bullets nur als Anker.

```bash
PIP_USER=0 ../../.venv/bin/pip install python-pptx      # einmalig
../../.venv/bin/python make_figures.py                  # Daten-Grafiken + Schemata
../../.venv/bin/python make_code_cards.py               # Code-Snippets als Cards
../../.venv/bin/python build_pptx.py                    # -> cuTile_Auto-Tuner.pptx
../../.venv/bin/python render_preview.py                # optional: Layout-Vorschau nach _layout/
```

`build_pptx.py` prüft am Ende selbst, dass kein Objekt über den Folienrand ragt; `render_preview.py`
rekonstruiert jede Folie als Bild (zur reinen Sichtkontrolle, kein PowerPoint nötig).

## Figures

Alle datengetriebenen Grafiken werden von `make_figures.py` erzeugt und landen in `figures/`
(je als `.png` für die Folien und `.svg` zum Nachbearbeiten). Palette: auf Rot-Grün-Schwäche
geprüfte dataviz-Default-Palette (Tuner = Blau, Default = Grau, torch = Orange, Handkernel =
dunkles Grau). CVD-validiert: Blau/Orange ΔE 96.7, Blau/Rot 75.7 (Protanopie); die
diverging-Grafik codiert „wer gewinnt" zusätzlich über die Links/Rechts-Position.

Neu erzeugen:

```bash
../../.venv/bin/python make_figures.py
```

Zahlenquelle ist `../result_dgx/study.log` (einzige Quelle mit torch.einsum für A05 **und** A06),
Balken-Rohdaten aus `../result_dgx/tune_*.csv`, Top-k-Kurve über `autotuner.rank()` (lokal, ohne GPU).

| Figure | Folie | Was sie zeigt |
|---|---|---|
| `fig_tiling` | 4 | Tiling/L2-Reuse-Schema (A links, B oben, C mittig, 2×2-Gruppe) |
| `fig_pipeline` | 5 | Pipeline-Fluss Einsum → generate → enumerate → prune → rank → tune → Cache |
| `fig_funnel` | 8 | Suchraum-Trichter 486 → 342 → 7 + Prune-Gründe |
| `fig_math` | 9 | Mathematik: 4 Prune-Filter (SMEM/Register-Formeln) + Ranking (DRAM-Traffic/Bandbreite, Roofline) |
| `fig_exec_order` | 11 | Config-Reihenfolge PAR│SEQ│PRIM + Splits (M/N/K → l2/prim) + A/B |
| `fig_regimes` | 13 | Die acht Shape-Regime (A05) mit exakten C/M/N/K + was jedes testet |
| `fig_a05_bars` | 14 | A05 je Regime, 4 Bars: Default / Tuner-top7 / Bench Best / torch |
| `fig_tuner_vs_torch` | 15 | Tuner/torch-Verhältnis (diverging um 1.0×), A05 unten, A06 gestreut |
| `fig_a06_bars` | 16 | A06 je Regime, 4 Bars: Default / Tuner-top7 / Bench Best / torch |
| `fig_a06_ladder` | 16 | A06-Referenz-Leiter: Default 26 → Hand 50 → Tuner 60 ≈ torch 60 |
| `fig_topk_curve` | 17 | % vom Optimum über Mess-Budget k (top-7 → ~97 %) |
| `fig_ranking_models` | 18 | bw / v2 / roofline: Spearman vs. Top-7-Ausbeute |
| `fig_crossgpu_lever` | 19 | Tuning-Hebel (Speedup Tuner/Default) GB10 vs. RTX 3070 |
| `fig_config_table` | 20 | Optimale Config je GPU (GB10 128×128 vs. 3070 kleineres k_prim) |
| `code_variant_a` | 7 | Code-Card: generischer Kernel mit ct.Constant + Swizzle |
| `code_prune` | 8 | Code-Card: die vier Prune-Filter (`prune_reason`) |
| `code_ring_a` | 10 | Code-Card: A06-Ring-Kernel (Batch-Decode + permute) |

`Bench Best` = bestes von 342 (A05) / 171 (A06) gemessenen Configs (Voll-Sweep-Optimum);
`Tuner-top7` = bester der Modell-Top-7 (was der Tuner praktisch liefert). `fig_tuner_vs_torch`
zeigt `BenchBest/torch` (0.38×–3.95×). Folie 9 = Mathe-Detail (Formeln), Folie 11 = Config-Reihenfolge.

Code-Cards kommen aus `make_code_cards.py` (dunkler Carbon-Look, Monospace-Raster).

## Noch offen / später einsetzen

- **RTX-3070-Ergebnisse:** erledigt — `fig_crossgpu_lever` zeigt den relativen Speedup (Tuner/Default)
  je Shape für GB10 und 3070 (Daten aus `../result_3070/`). Absolute TFLOPS sind bewusst nicht
  verglichen (andere Peak/BW/L2); die Aussage ist der Optimierungshebel + 16/16 verschiedene Configs.
- **Code-Snippets** (im Plan als C1–C3): `matmul_variant_a`, `matmul_ring_a`, `prune_reason` — kommen
  beim Folienbau als hübsch gesetzte Ausschnitte dazu (Code direkt aus `../src/autotuner/kernels.py`
  bzw. `search.py`).
