#!/usr/bin/env python3
# Erzeugt alle datengetriebenen Vortrags-Grafiken nach figures/.
# Zahlen aus result_dgx/study.log (autoritativ: einzige Quelle mit torch fuer A05+A06),
# Balken-Rohdaten aus result_dgx/tune_*.csv, Top-k-Kurve ueber autotuner.rank().
# Palette: validierte, farbenblind-sichere dataviz-Default-Palette (Blau/Orange/Grau).
#
# Aufruf:  .venv/bin/python project/praesentation/make_figures.py
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
RESULTS = os.path.join(HERE, "..", "result_dgx")           # GB10
R3070 = os.path.join(HERE, "..", "result_3070", "src", "results")   # RTX 3070
R3070_ALT = os.path.join(HERE, "..", "result_3070", "results")      # large_k liegt hier
FIGDIR = os.path.join(HERE, "figures")
sys.path.insert(0, SRC)
os.environ.setdefault("RESULTS_DIR", os.path.abspath(RESULTS))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch, Rectangle

# ---- Palette (dataviz reference, light mode) ----
INK      = "#0b0b0b"
INK2     = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
AXIS     = "#c3c2b7"
C_DEFAULT = "#898781"   # naive 8x8-Baseline -> neutral/grau (das "vorher")
C_TUNER   = "#2a78d6"   # unser Tuner -> Blau (Hero)
C_TORCH   = "#eb6834"   # torch/cuBLAS -> Orange
C_HAND    = "#4f4e4a"   # A06-Handkernel -> dunkles Grau (CVD-robust ueber Helligkeit)
POS       = "#2a78d6"   # diverging: Tuner gewinnt
NEG       = "#d03b3b"   # diverging: torch gewinnt
FUNNEL    = ["#86b6ef", "#3987e5", "#1c5cab"]   # ordinal hell->dunkel

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 11,
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "legend.frameon": False, "legend.fontsize": 12,
    "svg.fonttype": "none", "figure.dpi": 200, "savefig.dpi": 200,
})


def _clean(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("  ->", name)


# ============================================================
#  DATEN aus study.log (Default = naive 8x8, Tuner = best, torch = torch.einsum fp16)
# ============================================================
# (label, default, tuner_best, torch)
A05 = [
    ("a05\n(square,b4)", 63.9, 65.49, 63.09),
    ("square\nb1",       61.8, 63.97, 80.34),
    ("tall\nM≫N",        62.6, 63.32, 82.80),
    ("wide\nN≫M",        60.0, 60.98, 80.51),
    ("small_k",          35.8, 36.14, 58.92),
    ("large_k",          42.7, 45.83, 68.89),
    ("krumm\n(padding)", 26.6, 41.83, 49.45),
    ("batch16",          45.6, 46.30, 62.39),
]
A06 = [
    ("a06\n(Referenz)",  26.3, 59.83, 60.22),
    ("square\nx=y",      58.6, 66.65, 46.82),
    ("tall\nx≫y",        31.0, 68.02, 17.24),
    ("wide\ny≫x",        26.9, 66.45, 51.14),
    ("small_k",          18.5, 22.76, 38.33),
    ("large_k",          62.5, 73.34, 27.73),
    ("krumm\n(padding)", 14.3, 20.92, 54.70),
    ("batch\n(a8c4b8)",  54.3, 61.58, 76.07),
]


def grouped_bars(data, title, fname, note):
    labels = [d[0] for d in data]
    default = [d[1] for d in data]
    tuner   = [d[2] for d in data]
    torch   = [d[3] for d in data]
    x = np.arange(len(labels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    b1 = ax.bar(x - w, default, w, label="Default (8×8)", color=C_DEFAULT)
    b2 = ax.bar(x,     tuner,   w, label="Tuner (best)",   color=C_TUNER)
    b3 = ax.bar(x + w, torch,   w, label="torch.einsum",   color=C_TORCH)
    # Tuner/Default-Faktor ueber die Tuner-Balken (Groesse betont grosse Gewinne,
    # Farbe bleibt Tuner-Blau -> kein Rot/Gruen noetig)
    for xi, (d, t) in enumerate(zip(default, tuner)):
        fac = t / d
        ax.annotate(f"{fac:.2f}×", (xi, t), textcoords="offset points", xytext=(0, 4),
                    ha="center", va="bottom", fontweight="bold",
                    fontsize=12 if fac >= 1.4 else 10.5, color=C_TUNER)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("TFLOPS")
    ax.set_ylim(0, max(torch + tuner + default) * 1.12)
    ax.set_title(title, loc="left", color=INK, pad=12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    _clean(ax)
    ax.text(0, -0.35, note, transform=ax.transAxes, fontsize=9.5, color=MUTED)
    _save(fig, fname)


def fig_a06_ladder():
    # Referenz-Shape a06: die ehrliche Leiter Default -> Hand -> Tuner ~ torch
    names  = ["Default\n(8×8)", "Handkernel\n(A06)", "Tuner", "torch.einsum"]
    vals   = [26.3, 49.84, 59.83, 60.22]
    cols   = [C_DEFAULT, C_HAND, C_TUNER, C_TORCH]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    bars = ax.bar(names, vals, color=cols, width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    va="bottom", fontsize=13, fontweight="bold", color=INK)
    # +24% Klammer zwischen Hand und Tuner
    ax.annotate("", xy=(2, 59.83), xytext=(1, 49.84),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    ax.text(1.5, 56, "+24 %", ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylabel("TFLOPS")
    ax.set_ylim(0, 72)
    ax.set_title("A06-Referenz: der Tuner schlägt den Handkernel\nund liegt gleichauf mit torch",
                 loc="left", color=INK, pad=12, fontsize=15)
    _clean(ax)
    ax.text(0, -0.16, "GB10 · study.log · fp16 · acspx,bspy→abcyx  (4,3,64,64,1536)×(4,64,64,1152)",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    _save(fig, "fig_a06_ladder")


def fig_tuner_vs_torch():
    # diverging: log2(Tuner/torch). Blau = Tuner gewinnt, Rot = torch gewinnt.
    rows = ([("A05 · " + d[0].split("\n")[0], d[2] / d[3]) for d in A05] +
            [("A06 · " + d[0].split("\n")[0], d[2] / d[3]) for d in A06])
    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    y = np.arange(len(rows))[::-1]   # oben=erste
    fig, ax = plt.subplots(figsize=(8.8, 8.2))
    for yi, r in zip(y, ratios):
        val = math.log2(r)
        ax.barh(yi, val, color=POS if r >= 1 else NEG, height=0.66)
        ax.annotate(f"{r:.2f}×", (val, yi),
                    xytext=(4 if val >= 0 else -4, 0), textcoords="offset points",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=10.5, fontweight="bold",
                    color=POS if r >= 1 else NEG)
    ax.axvline(0, color=AXIS, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ticks = [0.25, 0.5, 1, 2, 4]
    ax.set_xticks([math.log2(t) for t in ticks])
    ax.set_xticklabels([f"{t:g}×" for t in ticks])
    ax.set_xlim(math.log2(0.3), math.log2(4.6))
    ax.set_xlabel("Tuner / torch.einsum   (log-Skala)")
    ax.set_title("Wo schlägt der Tuner die Library?", loc="left", color=INK, pad=10)
    ax.text(math.log2(1.05), len(rows) - 0.3, "Tuner schneller →", color=POS, fontsize=11, fontweight="bold")
    ax.text(math.log2(0.95), len(rows) - 0.3, "← torch schneller", color=NEG, fontsize=11,
            fontweight="bold", ha="right")
    _clean(ax, grid_axis="x")
    ax.text(0, -0.115, "GB10 · study.log · geom. Mittel A05 ~0.77× · A06 ~1.17×",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    _save(fig, "fig_tuner_vs_torch")


def fig_topk_curve():
    from analyze_tune import load_csv, batch_of, sig, _pool
    from autotuner.search import enumerate_candidates, prune, rank
    from autotuner.device_props import GB10
    from problems import PROBLEMS

    K_MAX = 100
    curves = []
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if not meas:
            continue
        ok = {s: m["tflops"] for s, m in meas.items() if m["ok"]}
        if not ok:
            continue
        absbest = max(ok.values())
        batch = batch_of(p["einsum"], p["shapes"])
        cands, _ = enumerate_candidates(p["einsum"], p["shapes"])
        kept, _ = prune(cands, GB10)
        pool = _pool(kept, meas, reg_clean=True)      # v2 (Default-Vorfilter)
        order = [sig(c) for c, _ in rank(pool, GB10, batch=batch, model="bw")]
        rb, frac = 0.0, []
        for s in order:
            rb = max(rb, ok[s])
            frac.append(rb / absbest)
        # auf K_MAX auffuellen (letzter Wert gehalten)
        frac = frac[:K_MAX] + [frac[-1]] * max(0, K_MAX - len(frac))
        curves.append(frac[:K_MAX])
    curves = np.array(curves) * 100
    ks = np.arange(1, K_MAX + 1)
    mean = curves.mean(0)
    lo, hi = curves.min(0), curves.max(0)

    def firstk(t):
        for i, v in enumerate(mean):
            if v >= t:
                return i + 1
        return K_MAX
    k95, k99 = firstk(95), firstk(99)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.fill_between(ks, lo, hi, color=C_TUNER, alpha=0.13, linewidth=0,
                    label="Spanne über 16 Shapes")
    ax.plot(ks, mean, color=C_TUNER, lw=2.4, label="Mittel (16 Shapes)")
    # Referenzlinien 95 / 99 %
    for lvl in (95, 99):
        ax.axhline(lvl, color=MUTED, lw=1.0, ls=(0, (4, 3)))
        ax.text(K_MAX, lvl, f" {lvl} %", va="center", ha="left", color=MUTED, fontsize=10)
    # drei Betriebspunkte: top-7 (~95 %), top-k99 (~99 %), Voll-Sweep (100 %)
    ax.scatter([7, k99], [mean[6], mean[k99 - 1]], color=C_TUNER, zorder=5, s=60,
               edgecolor="white", linewidth=1.2)
    ax.annotate(f"top-7 · ~3 s\nØ {mean[6]:.0f} % vom Optimum", (7, mean[6]),
                xytext=(9, mean[6] - 13), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.annotate(f"top-{k99} · ~{k99 * 0.4:.0f} s\nØ 99 % vom Optimum", (k99, mean[k99 - 1]),
                xytext=(k99 * 1.05, 94.5), fontsize=11, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(K_MAX, 100.4, "Voll-Sweep → 100 %\n(~3 min)", ha="right", va="bottom",
            fontsize=10, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 7, 10, 20, k99, 100])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlim(1, K_MAX)
    ax.set_ylim(70, 102)
    ax.set_xlabel("gemessene Kandidaten  k  (Modell-Top-k, v2)")
    ax.set_ylabel("erreichter Anteil am Optimum")
    ax.set_title("Wie oft ist das Optimum in den Top-k?", loc="left", color=INK, pad=10)
    ax.legend(loc="lower right")
    _clean(ax)
    ax.text(0, -0.155, "Ground Truth = Voll-Sweep (alle Configs gemessen) · GB10 · Zeit ≈ k · 0.4 s Compile",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    _save(fig, "fig_topk_curve")


def fig_ranking_models():
    # (name, subtitle, spearman, top7-Ausbeute %, farbe, offset(pt), ha, va)
    models = [("bw · Bandbreite", 0.03, 83.0, C_DEFAULT, (14, -4), "left", "top"),
              ("v2 · bw + Reg-Filter", 0.38, 97.8, C_TUNER, (0, 15), "center", "bottom"),
              ("roofline · max(mem, compute)", 0.50, 85.5, C_TORCH, (0, -16), "center", "top")]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for name, sp, top, col, off, ha, va in models:
        ax.scatter([sp], [top], s=360, color=col, zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f"{name}\nSpearman {sp:+.2f}  ·  top-7 {top:.0f} %",
                    (sp, top), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=11.5, color=INK,
                    fontweight="bold" if ha == "center" else "normal")
    ax.set_xlabel("Spearman-Korrelation  (Modell vs. Messung)")
    ax.set_ylabel("Top-7-Ausbeute  (% vom Optimum)")
    ax.set_xlim(-0.05, 0.63)
    ax.set_ylim(78, 104)
    ax.set_title("Besserer Ranker ≠ besserer Vorfilter", loc="left", color=INK, pad=24)
    ax.text(0, 1.02, "roofline korreliert am besten (+0.50), aber v2 filtert am besten (98 % @ top-7)",
            transform=ax.transAxes, fontsize=11.5, color=INK2)
    _clean(ax)
    ax.text(0, -0.135, "16 Shapes (A05+A06) · analyze_tune.py",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    _save(fig, "fig_ranking_models")


def fig_funnel():
    # Suchraum-Trichter 486 -> 342 -> 7 (A05). Horizontale, mittig zentrierte Balken.
    stages = [("enumeriert", 486, FUNNEL[0]),
              ("nach Pruning", 342, FUNNEL[1]),
              ("gemessen (Top-7)", 7, FUNNEL[2])]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    maxv = 486
    for i, (name, val, col) in enumerate(stages):
        y = len(stages) - 1 - i
        width = val / maxv
        left = (1 - width) / 2
        ax.barh(y, width, left=left, height=0.62, color=col)
        if width >= 0.10:
            ax.text(0.5, y, f"{val}", ha="center", va="center", color="white",
                    fontsize=17, fontweight="bold")
        else:
            ax.text(left + width + 0.015, y, f"{val}", ha="left", va="center",
                    color=INK, fontsize=17, fontweight="bold")
        ax.text(-0.02, y, name, ha="right", va="center", color=INK, fontsize=12.5)
    # Reduktions-Annotationen
    ax.text(1.02, 1.5, "−144\n(126 SMEM,\n18 Register)", va="center", fontsize=10, color=MUTED)
    ax.text(1.02, 0.5, "nur Top-7\nmessen\n(~3 s)", va="center", fontsize=10, color=MUTED)
    ax.set_xlim(-0.28, 1.28)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_title("Suchraum eingrenzen: 486 → 342 → 7  (A05)", loc="left", color=INK, pad=10)
    ax.axis("off")
    _save(fig, "fig_funnel")


def _load_csv_dir(d, name):
    path = os.path.join(d, f"tune_{name}.csv")
    if not os.path.exists(path):
        return None
    import csv
    m = {}
    for r in csv.DictReader(open(path)):
        if int(r["ok"]):
            k = (r["variant"], int(r["m_prim"]), int(r["n_prim"]),
                 int(r["k_prim"]), int(r["m_l2"]), int(r["n_l2"]))
            m[k] = float(r["tflops"])
    return m or None


def _best_sig(d, name, alt=None):
    m = _load_csv_dir(d, name) or (_load_csv_dir(alt, name) if alt else None)
    if not m:
        return None
    return max(m, key=m.get)


def _fmt_cfg(s):
    return f"{s[1]}/{s[2]}/{s[3]}  {s[4]}×{s[5]}"


def fig_config_table():
    # Welche Config ist optimal je GPU? Zeigt die Divergenz konkret an Beispiel-Shapes.
    rows = [("a05", "a05"), ("tall", "tall"), ("small_k", "small_k"), ("krumm", "krumm"),
            ("a06", "a06"), ("a06_tall", "tall"), ("a06_large_k", "large_k"),
            ("a06_krumm", "krumm")]
    data = []
    for key, label in rows:
        fam = "A06 · Ring" if key.startswith("a06") else "A05 · GEMM"
        gb = _best_sig(RESULTS, key)
        r3 = _best_sig(R3070, key, R3070_ALT)
        data.append((fam, label, _fmt_cfg(gb), _fmt_cfg(r3)))

    fig, ax = plt.subplots(figsize=(11.0, 5.3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(data)
    x0, xg, xr = 0.02, 0.42, 0.71          # Spalten-Startpunkte
    top, rh = 0.86, 0.86 / (n + 1)
    # Kopfzeile
    ax.text(x0, top, "Shape", fontsize=13, fontweight="bold", color=INK, va="center")
    ax.add_patch(Rectangle((xg - 0.02, top - rh * 0.5), 0.28, rh, color=C_TUNER, alpha=0.14))
    ax.add_patch(Rectangle((xr - 0.02, top - rh * 0.5), 0.30, rh, color=C_TORCH, alpha=0.14))
    ax.text(xg + 0.12, top, "GB10  (25 MB L2)", fontsize=12.5, fontweight="bold",
            color=C_TUNER, va="center", ha="center")
    ax.text(xr + 0.13, top, "RTX 3070  (4 MB L2)", fontsize=12.5, fontweight="bold",
            color=C_TORCH, va="center", ha="center")
    prev_fam = None
    for i, (fam, label, gc, rc) in enumerate(data):
        y = top - (i + 1) * rh
        if fam != prev_fam:
            ax.text(x0, y + rh * 0.02, fam, fontsize=10, color=MUTED, va="center", style="italic")
            prev_fam = fam
        ax.text(x0 + 0.11, y, label, fontsize=12.5, color=INK, va="center")
        ax.add_patch(Rectangle((xg - 0.02, y - rh * 0.5), 0.28, rh, color=C_TUNER,
                     alpha=0.06 if i % 2 else 0.10))
        ax.add_patch(Rectangle((xr - 0.02, y - rh * 0.5), 0.30, rh, color=C_TORCH,
                     alpha=0.06 if i % 2 else 0.10))
        ax.text(xg + 0.12, y, gc, fontsize=12.5, color=INK, va="center", ha="center",
                family="DejaVu Sans Mono")
        ax.text(xr + 0.13, y, rc, fontsize=12.5, color=INK, va="center", ha="center",
                family="DejaVu Sans Mono")
    ax.text(0.02, 0.985, "Optimale Config unterscheidet sich pro GPU  (m/n/k-Prim · L2-Gruppe)",
            fontsize=15, fontweight="bold", color=INK, va="top")
    ax.text(0.02, -0.02,
            "Muster: GB10 → große 128×128-Tiles (großes L2 verträgt sie) · 3070 → kleineres "
            "k_prim=32, oft asymmetrisch/64-breit", fontsize=10.5, color=INK2, va="top")
    _save(fig, "fig_config_table")


def _lever(meas):
    # (speedup Tuner/Default, best-config-signatur)
    from problems import DEFAULT_CONFIG
    DEF = ("A", DEFAULT_CONFIG["m_prim"], DEFAULT_CONFIG["n_prim"],
           DEFAULT_CONFIG["k_prim"], DEFAULT_CONFIG["m_l2"], DEFAULT_CONFIG["n_l2"])
    best = max(meas, key=meas.get)
    return meas[best] / meas[DEF], best


def fig_crossgpu_lever():
    # Absolute TFLOPS der GB10 und 3070 sind nicht vergleichbar (andere Peak/BW/L2).
    # Vergleichbar ist der OPTIMIERUNGSHEBEL: Speedup Tuner/Default pro Shape.
    from problems import PROBLEMS
    a05 = [p["name"] for p in PROBLEMS if not p["name"].startswith("a06")]
    a06 = [p["name"] for p in PROBLEMS if p["name"].startswith("a06")]
    order = a05 + a06
    short = {"a05": "a05", "square_1b": "square", "tall": "tall", "wide": "wide",
             "small_k": "small_k", "large_k": "large_k", "krumm": "krumm", "batch16": "batch16",
             "a06": "a06", "a06_square": "square", "a06_tall": "tall", "a06_wide": "wide",
             "a06_small_k": "small_k", "a06_large_k": "large_k", "a06_krumm": "krumm",
             "a06_batch": "batch"}
    gb_sp, r_sp, labels, splitpos = [], [], [], len(a05)
    for n in order:
        g = _load_csv_dir(RESULTS, n)
        r = _load_csv_dir(R3070, n) or _load_csv_dir(R3070_ALT, n)
        gb_sp.append(_lever(g)[0] if g else float("nan"))
        r_sp.append(_lever(r)[0] if r else float("nan"))
        labels.append(short[n])
    gb_avg = float(np.nanmean(gb_sp))
    r_avg = float(np.nanmean(r_sp))

    # x-Positionen mit Luecke zwischen A05- und A06-Block
    xs = []
    x = 0.0
    for i in range(len(order)):
        if i == splitpos:
            x += 0.9
        xs.append(x)
        x += 1.0
    xs = np.array(xs)
    w = 0.38
    fig, ax = plt.subplots(figsize=(13.4, 5.6))
    xr = xs[-1] + 0.55
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(xr, 1.0, "kein\nGewinn", va="center", fontsize=9.5, color=MUTED)
    b1 = ax.bar(xs - w / 2 - 0.02, gb_sp, w, color=C_TUNER, label="GB10  (25 MB L2)")
    b2 = ax.bar(xs + w / 2 + 0.02, r_sp, w, color=C_TORCH, label="RTX 3070  (4 MB L2)")
    for xi, g, r in zip(xs, gb_sp, r_sp):
        ax.text(xi - w / 2 - 0.02, g + 0.03, f"{g:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=C_TUNER, fontweight="bold")
        ax.text(xi + w / 2 + 0.02, r + 0.03, f"{r:.1f}", ha="center", va="bottom",
                fontsize=8.5, color=C_TORCH, fontweight="bold")
    # Durchschnittslinien + Labels im freien rechten Rand (bei "kein Gewinn")
    ax.axhline(gb_avg, color=C_TUNER, lw=1.0, ls=":", alpha=0.7)
    ax.axhline(r_avg, color=C_TORCH, lw=1.0, ls=":", alpha=0.7)
    ax.text(xr, r_avg, f"Ø {r_avg:.2f}×", va="center", ha="left", fontsize=9.5,
            color=C_TORCH, fontweight="bold")
    ax.text(xr, gb_avg, f"Ø {gb_avg:.2f}×", va="center", ha="left", fontsize=9.5,
            color=C_TUNER, fontweight="bold")
    # Block-Labels
    ax.text(np.mean(xs[:splitpos]), ax.get_ylim()[1] * 0.98, "A05 · GEMM", ha="center",
            va="top", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(np.mean(xs[splitpos:]), ax.get_ylim()[1] * 0.98, "A06 · Ring", ha="center",
            va="top", fontsize=11.5, color=INK, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Speedup  Tuner / Default")
    ax.set_ylim(0.9, max(np.nanmax(gb_sp), np.nanmax(r_sp)) * 1.12)
    ax.set_xlim(-1.0, xs[-1] + 2.1)
    ax.set_title("Der Tuning-Hebel wirkt auf beiden GPUs — auf der 3070 stärker",
                 loc="left", color=INK, pad=12)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _clean(ax)
    ax.text(0, -0.30, "relativer Speedup (unitless) — absolute TFLOPS beider Karten sind nicht "
            "vergleichbar · beste Config in 16/16 Shapes je GPU verschieden",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    _save(fig, "fig_crossgpu_lever")


# ============================================================
#  Schematische Skizzen (keine Messdaten): Pipeline + Tiling
# ============================================================

def _pbox(ax, cx, cy, w, h, title, sub, edge, fill="white"):
    ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.35, cy - h / 2 - 0.5), w, h,
                 boxstyle="round,pad=0.02,rounding_size=1.4", linewidth=0,
                 facecolor="#0b0b0b12", zorder=2))                       # Schatten
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=1.4", linewidth=1.8,
                 edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(cx, cy + h * 0.15, title, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK, zorder=4)
    if sub:
        ax.text(cx, cy - h * 0.23, sub, ha="center", va="center",
                fontsize=9.5, color=INK2, zorder=4)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 45)
    ax.axis("off")
    cy, w, h = 24, 12.6, 11.5
    centers = [6.7, 21.2, 35.7, 50.2, 64.7, 79.2, 93.7]
    stages = [
        ("Eingabe", "Einsum + Shapes", MUTED, "#eef0f1"),
        ("generate_config", "Basic-Config", C_TUNER, "white"),
        ("enumerate", "Suchraum", C_TUNER, "white"),
        ("prune", "4 Filter", C_TUNER, "white"),
        ("rank", "Kostenmodell v2", C_TUNER, "white"),
        ("tune", "compile · verify\n+ do_bench", C_TORCH, "white"),
        ("Ergebnis", "Beste Config\n+ Cache", C_TUNER, "#e7f0fb"),
    ]
    # Bänder: ohne GPU vs. auf der GPU
    ax.add_patch(Rectangle((centers[1] - 8.2, 11.5), (centers[4] + 8.2) - (centers[1] - 8.2), 25.5,
                 facecolor=C_TUNER, alpha=0.06, edgecolor="none", zorder=1))
    ax.text((centers[1] + centers[4]) / 2, 38.3, "reines Python — ohne GPU  (search.py)",
            ha="center", fontsize=10.5, color=C_TUNER, fontweight="bold")
    ax.add_patch(Rectangle((centers[5] - 8.2, 11.5), 16.4, 25.5,
                 facecolor=C_TORCH, alpha=0.08, edgecolor="none", zorder=1))
    ax.text(centers[5], 38.3, "auf der GPU (GB10)", ha="center",
            fontsize=10.5, color=C_TORCH, fontweight="bold")
    # Pfeile
    for i in range(len(centers) - 1):
        ax.add_patch(FancyArrowPatch((centers[i] + w / 2, cy), (centers[i + 1] - w / 2, cy),
                     arrowstyle="-|>", mutation_scale=17, lw=1.8, color=MUTED, zorder=2))
    # Boxen
    for cx, (t, s, e, f) in zip(centers, stages):
        _pbox(ax, cx, cy, w, h, t, s, e, f)
    # Trichter-Badges unter den passenden Stufen
    for cx, val, col in [(centers[2], "486", C_TUNER), (centers[3], "342", C_TUNER),
                         (centers[5], "Top-7", C_TORCH)]:
        ax.text(cx, 14.4, val, ha="center", va="center", fontsize=12.5, fontweight="bold",
                color="white", zorder=5,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=col, edgecolor="none"))
    ax.text(2, 43, "Die Tuner-Pipeline: von Einsum zur gemessen besten Config",
            fontsize=15.5, fontweight="bold", color=INK)
    _save(fig, "fig_pipeline")


def fig_tiling():
    fig, ax = plt.subplots(figsize=(10.4, 7.6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-3.4, 9.2)
    ax.set_ylim(-1.8, 8.2)
    n = 4
    A_hl = "#bcd6f4"
    Cx0, Cy0 = 0.0, 0.0
    gcols, grows = (1, 2), (1, 2)          # 2x2-L2-Gruppe (mittig)
    # C-Gitter (Output M x N)
    for i in range(n):
        for j in range(n):
            ingrp = i in gcols and j in grows
            ax.add_patch(Rectangle((Cx0 + i, Cy0 + j), 1, 1,
                         facecolor=C_TUNER if ingrp else "white",
                         edgecolor=AXIS, linewidth=1.1, zorder=3))
    # A links (M x K), 0.8 Abstand -- Zeilen fluchten mit C
    Aw = 1.8
    Ax0 = Cx0 - 0.8 - Aw
    for j in range(n):
        ax.add_patch(Rectangle((Ax0, Cy0 + j), Aw, 1,
                     facecolor=A_hl if j in grows else "white",
                     edgecolor=AXIS, linewidth=1.1, zorder=3))
    # B oben (K x N), 0.8 Abstand -- Spalten fluchten mit C
    Bh = 1.8
    By0 = Cy0 + n + 0.8
    for i in range(n):
        ax.add_patch(Rectangle((Cx0 + i, By0), 1, Bh,
                     facecolor=A_hl if i in gcols else "white",
                     edgecolor=AXIS, linewidth=1.1, zorder=3))
    # Matrix-Labels in den freien Flaechen
    ax.text(Ax0 + Aw / 2, Cy0 + n + 0.2, "A  (M×K)", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(Cx0 + n / 2, By0 + Bh + 0.2, "B  (K×N)", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(Cx0 + n / 2, Cy0 - 0.4, "C = A · B   (M×N)", ha="center", va="top",
            fontsize=13, fontweight="bold", color=INK)
    # Callouts rechts, vertikal getrennt
    ax.annotate("1 Prim-Tile = M_PRIM×N_PRIM\n(1 CTA · 1 mma)",
                (Cx0 + 3.5, Cy0 + 3.5), xytext=(Cx0 + n + 0.9, Cy0 + 3.6),
                fontsize=10.5, color=INK2, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.annotate("L2-Gruppe  m_l2×n_l2\n(hier 2×2)",
                (Cx0 + 2, Cy0 + 2), xytext=(Cx0 + n + 0.9, Cy0 + 1.4),
                fontsize=11, color=C_TUNER, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_TUNER, lw=1.4))
    # Bild-Untertitel: die eigentliche Reuse-Aussage
    ax.text((Ax0 + Cx0 + n) / 2, -1.5,
            "Die Gruppe lädt ihre A-Zeilen und B-Spalten je einmal und nutzt sie mehrfach"
            "  →  bleibt im L2 resident",
            ha="center", va="top", fontsize=11, color=INK)
    ax.text(-3.3, 7.9, "L2-Reuse: eine Block-Gruppe teilt sich A-Zeilen und B-Spalten",
            fontsize=15, fontweight="bold", color=INK)
    _save(fig, "fig_tiling")


if __name__ == "__main__":
    print("erzeuge Figures ->", FIGDIR)
    grouped_bars(A05, "A05: Tuner bestätigt die Handarbeit, gewinnt bei krummen Shapes",
                 "fig_a05_bars", "GB10 · study.log · Faktor = Tuner/Default · torch = cuBLAS-Referenz")
    grouped_bars(A06, "A06: Tuner schlägt die naive Default durchweg deutlich",
                 "fig_a06_bars", "GB10 · study.log · Faktor = Tuner/Default · Default = aus A05 übernommene 8×8")
    fig_a06_ladder()
    fig_tuner_vs_torch()
    fig_topk_curve()
    fig_ranking_models()
    fig_funnel()
    fig_crossgpu_lever()
    fig_config_table()
    fig_pipeline()
    fig_tiling()
    print("fertig.")
