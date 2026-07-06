# Präsentation — cuTile Auto-Tuner

Ordner für die Mittwochs-Präsentation. Der Folienplan liegt eine Ebene höher in
`../vortrag_aufteilung.md`.

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
| `fig_tiling` | 2 | Tiling/L2-Reuse-Schema (A links, B oben, C mittig, 2×2-Gruppe) |
| `fig_pipeline` | 3 | Pipeline-Fluss Einsum → generate → enumerate → prune → rank → tune → Cache |
| `fig_funnel` | 6 | Suchraum-Trichter 486 → 342 → 7 + Prune-Gründe |
| `fig_a05_bars` | 9 | A05 je Regime: Default / Tuner / torch, Tuner/Default-Faktor |
| `fig_tuner_vs_torch` | 10 | Tuner/torch-Verhältnis (diverging um 1.0×), A05 unten, A06 gestreut |
| `fig_a06_bars` | 11 | A06 je Regime: Default / Tuner / torch |
| `fig_a06_ladder` | 11 | A06-Referenz-Leiter: Default 26 → Hand 50 → Tuner 60 ≈ torch 60 |
| `fig_topk_curve` | 12 | % vom Optimum über Mess-Budget k (top-7 → ~97 %) |
| `fig_ranking_models` | 13 | bw / v2 / roofline: Spearman vs. Top-7-Ausbeute |
| `fig_crossgpu_placeholder` | 14 | GB10 vs. RTX 3070 (3070 = TBD-Platzhalter) |

## Noch offen / später einsetzen

- **RTX-3070-Ergebnisse:** sobald da, in `make_figures.py` in `fig_crossgpu_placeholder` die echten
  Werte statt der TBD-Balken eintragen (und ggf. eine zweite Balkengruppe je Shape).
- **Code-Snippets** (im Plan als C1–C3): `matmul_variant_a`, `matmul_ring_a`, `prune_reason` — kommen
  beim Folienbau als hübsch gesetzte Ausschnitte dazu (Code direkt aus `../src/autotuner/kernels.py`
  bzw. `search.py`).
