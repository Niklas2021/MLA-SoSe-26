#!/usr/bin/env python3
# Rendert Code-Snippets als dunkle "Code-Cards" (Carbon-Look) nach figures/.
# Feste Bildgroesse -> im Deck layout-sicher einbettbar. Monospace-Raster:
# jedes Zeichen sitzt auf einer festen Spaltenbreite, damit nichts verrutscht.
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BG   = "#0f1720"
EDGE = "#2a3a4a"
BASE = "#d6dee8"
KW   = "#569cd6"   # keywords
API  = "#4ec9b0"   # cuTile-API / Typen
NUM  = "#b5cea8"
COM  = "#7fa98a"   # Kommentare
PUNCT = "#9aa7b4"
DECO = "#c586c0"

KWset = {"def", "for", "if", "in", "range", "return", "else", "elif",
         "and", "or", "not", "None", "import", "from", "as", "while"}
APIset = {"ct", "mma", "load", "store", "kernel", "Constant", "PaddingMode",
          "ZERO", "zeros", "permute", "reshape", "float32", "float16", "bid",
          "astype", "index", "dtype", "padding_mode", "tile", "int"}

CHARW = 0.108     # Zoll pro Zeichen (Breitenberechnung)
FONTSZ = 12.5
LINEH = 0.285
PADX, PADY = 0.52, 0.30


def _color(tok):
    if tok in KWset:
        return KW
    if tok in APIset:
        return API
    if re.fullmatch(r"\d+", tok):
        return NUM
    if re.fullmatch(r"[(){}\[\].,:=%*/+<>@-]+", tok):
        return PUNCT
    return BASE


def render(lines, name):
    ncols = max(len(l) for l in lines)
    nrows = len(lines)
    figw = ncols * CHARW + 2 * PADX
    figh = nrows * LINEH + 2 * PADY + 0.35        # +Platz fuer die Fenster-Punkte
    fig = plt.figure(figsize=(figw, figh))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, figw)
    ax.set_ylim(0, figh)
    ax.add_patch(FancyBboxPatch((0.1, 0.1), figw - 0.2, figh - 0.2,
                 boxstyle="round,pad=0.02,rounding_size=0.16",
                 facecolor=BG, edgecolor=EDGE, linewidth=1.5))
    for k, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((PADX + 0.12 + k * 0.30, figh - 0.36), 0.075,
                     color=c, zorder=5))
    y0 = figh - 0.72
    for i, line in enumerate(lines):
        y = y0 - i * LINEH
        code, comment = line, None
        if "#" in line:
            idx = line.index("#")
            code, comment = line[:idx], line[idx:]
        x = PADX
        deco = code.lstrip().startswith("@")
        for tok in re.findall(r"\s+|\w+|\W", code):
            if tok.strip() == "":
                x += len(tok) * CHARW
                continue
            col = DECO if deco else _color(tok)
            ax.text(x, y, tok, family="DejaVu Sans Mono", fontsize=FONTSZ,
                    color=col, va="center", ha="left")
            x += len(tok) * CHARW
        if comment:
            ax.text(x, y, comment, family="DejaVu Sans Mono", fontsize=FONTSZ,
                    color=COM, va="center", ha="left", style="italic")
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=200,
                facecolor="none", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("  ->", name, f"({ncols} cols x {nrows} lines)")


C1 = [
    "@ct.kernel",
    "def matmul_variant_a(A, B, C,",
    "        M_PRIM: ct.Constant[int], N_PRIM: ct.Constant[int],",
    "        K_PRIM: ct.Constant[int], M_L2: ct.Constant[int],",
    "        N_L2:   ct.Constant[int], num_k_outer: ct.Constant[int]):",
    "    pid = ct.bid(0)                   # eine Block-ID ...",
    "    n_l2_idx = pid % N_L2; pid //= N_L2   # ... in die",
    "    m_l2_idx = pid % M_L2; pid //= M_L2   # L2-Gruppe swizzeln",
    "    acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)",
    "    for k_it in range(num_k_outer):",
    "        a = ct.load(A, ..., padding_mode=ct.PaddingMode.ZERO)",
    "        b = ct.load(B, ..., padding_mode=ct.PaddingMode.ZERO)",
    "        acc = ct.mma(a, b, acc)       # Tensor-Core-Kachel",
    "    ct.store(C, ..., tile=acc.astype(ct.float16))",
]

C2 = [
    "@ct.kernel",
    "def matmul_ring_a(A, B, C, M_PRIM: ct.Constant[int], ...,",
    "                  SIZE_B: ct.Constant[int], SIZE_S: ct.Constant[int]):",
    "    pid = ct.bid(0)                      # unabhaengige Batches:",
    "    b_idx = pid % SIZE_B; pid //= SIZE_B #   a,c nur in A, b nur in B",
    "    c_idx = pid % SIZE_C; pid //= SIZE_C",
    "    a_idx = pid",
    "    acc = ct.zeros((M_PRIM, N_PRIM), ct.float32)",
    "    for s_it in range(SIZE_S):           # 2. Reduktion s als SEQ-Loop",
    "        for k_it in range(num_k_outer):  # p als prim_k im mma",
    "            tA = ct.load(A, index=(a_idx, c_idx, s_it, k_it, ...))",
    "            tA = ct.permute(tA, (1, 0))  # Layout ist nicht mma-fertig",
    "            tB = ct.load(B, index=(b_idx, s_it, k_it, ...))",
    "            acc = ct.mma(tA, tB, acc)",
    "    ct.store(C, index=(a_idx, b_idx, c_idx, y_block, x_block), ...)",
]

C3 = [
    "def prune_reason(cand, dev, ...):",
    "    if cand.m_prim % 16 or cand.n_prim % 16 or cand.k_prim % 16:",
    "        return 'mma_align'                 # 1) MMA-Teilbarkeit",
    "    if estimate_smem_bytes(cand) > smem_limit:",
    "        return 'smem_exceeded'             # 2) Shared-Memory-Budget",
    "    if estimate_acc_registers(cand) > 0.5 * dev.regs_per_block:",
    "        return 'acc_registers'             # 3) Akku-Register",
    "    if padding_ratio(cand) > max_padding:",
    "        return 'padding_waste'             # 4) Padding-Verschwendung",
    "    return None                            # -> Kandidat ueberlebt",
]


if __name__ == "__main__":
    print("erzeuge Code-Cards ->", FIGDIR)
    render(C1, "code_variant_a")
    render(C2, "code_ring_a")
    render(C3, "code_prune")
    print("fertig.")
