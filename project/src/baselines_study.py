# Welche Baseline ist fair? Laeuft lokal aus den Sweep-CSVs, ohne GPU.
#
# Bisher war die Vergleichsbasis DEFAULT_CONFIG = die aus A05 uebernommene Hand-Config
# (128/128/64, 8x8). Das Problem: die wurde fuer EINE Shape auf EINER GPU von Hand
# hergeleitet. Auf ihrer Heimat-Shape ist sie top, sonst faellt sie ab -- und auf einer
# anderen Karte erst recht. Ein Tuner-Gewinn dagegen misst also zu einem guten Teil nur,
# wie schlecht die uebernommene Config passt, nicht was Per-Shape-Tuning bringt.
#
# Faire Baseline ist stattdessen: die beste EINE feste Config fuer diese GPU. Das ist,
# was jemand nimmt, der einmal auf der Zielkarte nachgesehen und sich dann auf eine
# Config festgelegt hat.
#
# Zwei Varianten, weil die Wahl sonst Nachwissen enthaelt:
#   fest-oracle : beste feste Config ueber ALLE Shapes, auch die bewertete
#                 -> optimistische Baseline, also konservative Schranke fuer den Tuner
#   fest-loo    : Leave-one-out, die feste Config wird auf den anderen 15 Shapes
#                 gewaehlt und auf der 16. bewertet -> ehrlich, kein Nachwissen
#
# python baselines_study.py [RESULTS_DIR]
import csv
import math
import os
import sys

DEFAULT_SIG = ("A", 128, 128, 64, 8, 8)


def geo(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def load(d):
    sweeps = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("tune_") and f.endswith(".csv")):
            continue
        m = {}
        for r in csv.DictReader(open(os.path.join(d, f))):
            if int(r["ok"]):
                m[(r["variant"], int(r["m_prim"]), int(r["n_prim"]), int(r["k_prim"]),
                   int(r["m_l2"]), int(r["n_l2"]))] = float(r["tflops"])
        if m:
            sweeps[f[5:-4]] = m
    return sweeps


def best_fixed(sweeps, names, pool):
    # die feste Config, die ueber `names` im geom. Mittel den groessten Anteil am
    # jeweiligen Per-Shape-Optimum holt
    def score(cfg):
        return geo([sweeps[n][cfg] / max(sweeps[n].values()) for n in names])
    return max(pool, key=score)


def extent(cfg):
    # Ausdehnung einer Swizzle-Gruppe auf C: (m_l2*m_prim) x (n_l2*n_prim).
    # Das ist die Groesse, die den L2-Reuse bestimmt -- nicht die Tile-Groesse allein.
    _, mp, np_, kp, ml, nl = cfg
    return ml * mp, nl * np_


def padding_waste(e, cfg):
    # gepaddetes Volumen / Original. Die Gruppen-Ausdehnung bestimmt, auf welches
    # Vielfache M und N hochgerundet werden -- das ist der Knackpunkt, nicht die
    # Tile-Groesse.
    _, mp, np_, kp, ml, nl = cfg
    ceil = lambda a, b: (a + b - 1) // b
    pm = ceil(e.orig_m, mp * ml) * mp * ml
    pn = ceil(e.orig_n, np_ * nl) * np_ * nl
    pk = ceil(e.orig_k, kp) * kp
    return (pm * pn * pk) / (e.orig_m * e.orig_n * e.orig_k)


def padding_check(sweeps, oracle):
    # Gegenprobe: erklaert allein der Padding-Ueberhang den Abstand der beiden
    # Baselines? Wenn ja, ist klar WARUM die A05-Config als Default schlecht ist.
    try:
        from autotuner.einsum_parser import parse_einsum
        from problems import PROBLEMS
    except ImportError:
        return
    print(f"\n{'Shape':13s} {'Padding Def':>12s} {'Padding fest':>13s} "
          f"{'vorhergesagt':>13s} {'gemessen':>10s}")
    print("-" * 66)
    for p in PROBLEMS:
        m = sweeps.get(p["name"])
        if not m or DEFAULT_SIG not in m:
            continue
        e = parse_einsum(p["einsum"], p["shapes"])
        wd, wf = padding_waste(e, DEFAULT_SIG), padding_waste(e, oracle)
        print(f"{p['name']:13s} {wd:11.3f}x {wf:12.3f}x {wd/wf:12.2f}x "
              f"{m[oracle]/m[DEFAULT_SIG]:9.2f}x")


def main(d):
    sweeps = load(d)
    if not sweeps:
        print(f"keine tune_*.csv in {d}")
        return
    names = sorted(sweeps)
    pool = set.intersection(*[set(m) for m in sweeps.values()])
    print(f"=== {os.path.basename(os.path.abspath(d))}: {len(names)} Shapes, "
          f"{len(pool)} ueberall lauffaehige Configs ===\n")

    oracle = best_fixed(sweeps, names, pool)
    me, ne = extent(oracle)
    dm, dn = extent(DEFAULT_SIG)
    print(f"A05-Default      {DEFAULT_SIG}   Gruppe {dm}x{dn}")
    print(f"beste feste      {oracle}   Gruppe {me}x{ne}\n")

    hdr = (f"{'Shape':13s} {'best':>7s} | {'A05-Def':>8s} {'x':>7s} | "
           f"{'fest-oracle':>11s} {'x':>7s} | {'fest-loo':>9s} {'x':>7s}")
    print(hdr)
    print("-" * len(hdr))

    r_def, r_orc, r_loo, f_def, f_orc, f_loo = [], [], [], [], [], []
    for n in names:
        m = sweeps[n]
        b = max(m.values())
        dv = m.get(DEFAULT_SIG)
        ov = m[oracle]
        # Leave-one-out: feste Config OHNE diese Shape waehlen
        loo_cfg = best_fixed(sweeps, [x for x in names if x != n], pool)
        lv = m[loo_cfg]
        if dv:
            r_def.append(b / dv); f_def.append(dv / b)
        r_orc.append(b / ov); f_orc.append(ov / b)
        r_loo.append(b / lv); f_loo.append(lv / b)
        print(f"{n:13s} {b:7.2f} | {dv if dv else 0:8.2f} {b/dv if dv else 0:6.3f}x | "
              f"{ov:11.2f} {b/ov:6.3f}x | {lv:9.2f} {b/lv:6.3f}x")

    print(f"\n{'':13s} {'Anteil am Per-Shape-Optimum':>34s}   {'Tuner-Gewinn':>13s}")
    for lbl, frac, rat in (("A05-Default (uebernommen)", f_def, r_def),
                           ("feste Config, oracle", f_orc, r_orc),
                           ("feste Config, leave-one-out", f_loo, r_loo)):
        if frac:
            print(f"  {lbl:32s} {100*geo(frac):8.1f} %   {geo(rat):12.3f}x")
    padding_check(sweeps, oracle)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "results_dgx_v2")
    main(d)
