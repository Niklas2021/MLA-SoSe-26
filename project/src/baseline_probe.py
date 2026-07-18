# Ermittelt BASELINE_CONFIGS fuer eine GPU, ohne den vollen Sweep.
#
# Die faire Baseline ist "die beste EINE feste Config fuer diese GPU". Streng bekommt
# man die nur aus einer Vollmessung -- auf der 3070 sind das 3.5 Stunden. Diese Sonde
# misst stattdessen nur einen kleinen Kandidatenpool auf allen Shapes.
#
# Gegen die GB10-Vollmessung geprueft (dort ist die Antwort bekannt):
#   --pool winners    13 Configs, 208 Messungen  -> findet 89.6 % statt 90.9 %
#   --pool neighbours 77 Configs, 1232 Messungen -> findet exakt dieselbe Config
# winners ist also ~13x billiger und unterschaetzt die Baseline um ~1.3 Punkte, was den
# Tuner-Gewinn um ~1.5 % zu gut aussehen laesst. Wer das nicht will, nimmt neighbours.
#
# Ablauf: erst die Per-Shape-Gewinner besorgen (Hybrid pro Shape), daraus den Pool
# bauen, dann jede Pool-Config auf jeder Shape messen und die mit dem besten
# geometrischen Mittel (Anteil am jeweiligen Per-Shape-Optimum) ausgeben.
#
# Auf der Zielkarte:  python baseline_probe.py [--pool winners|neighbours]
import datetime
import math
import sys

import torch
import triton.testing

import results_io
from autotune import model_ranked, _batch_and_flops, _knobs
from autotuner import strategies
from autotuner.einsum_parser import parse_einsum
from autotuner.kernels import run_candidate
from autotuner.search import (M_PRIM_CHOICES, N_PRIM_CHOICES, K_PRIM_CHOICES,
                              M_L2_CHOICES, N_L2_CHOICES)
from problems import PROBLEMS, DEFAULT_CONFIG

try:
    from autotuner.device_props import get_device_properties
    DEV = get_device_properties()
except Exception:
    from autotuner.device_props import GB10 as DEV

WARMUP, REP = 25, 100
AXIS_VALUES = {"m_prim": M_PRIM_CHOICES, "n_prim": N_PRIM_CHOICES,
               "k_prim": K_PRIM_CHOICES, "m_l2": M_L2_CHOICES, "n_l2": N_L2_CHOICES}


def geo(xs):
    return math.exp(sum(math.log(v) for v in xs) / len(xs))


def neighbours(sig):
    # sig = (m_prim, n_prim, k_prim, m_l2, n_l2, order, variant) wie strategies.sig
    out = {sig}
    for axis, vals in AXIS_VALUES.items():
        for v in vals:
            t = list(sig)
            t[strategies.IDX[axis]] = v
            out.add(tuple(t))
    return out


def make_measure(problem):
    e = parse_einsum(problem["einsum"], problem["shapes"])
    _, flops = _batch_and_flops(e)
    torch.manual_seed(0)
    A = torch.randn(problem["shapes"][0], dtype=torch.float16, device="cuda")
    B = torch.randn(problem["shapes"][1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(problem["einsum"].replace(" ", ""), A.float(), B.float()).half()

    def measure(cand):
        try:
            out = run_candidate(cand, A, B)
            torch.cuda.synchronize()
            if out.shape != ref.shape or not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                return None
            ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                         warmup=WARMUP, rep=REP)
        except Exception:
            return None
        return flops / (ms * 1e-3) / 1e12

    return measure


def main(pool_mode):
    lines, n_meas = [], 0

    def log(m=""):
        print(m)
        lines.append(m)

    log("=== Baseline-Sonde ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"GPU: {DEV.gpu_name}   Pool: {pool_mode}")
    log("")

    # Phase 1: Per-Shape-Gewinner via Hybrid
    log("Phase 1: Per-Shape-Gewinner (Hybrid)")
    winners, ranked_of, measure_of, peak = {}, {}, {}, {}
    for p in PROBLEMS:
        ranked = model_ranked(p["einsum"], p["shapes"], DEV)
        measure = make_measure(p)
        cand, tf, n = strategies.hybrid(ranked, measure, k=7)
        n_meas += n
        if cand is None:
            log(f"  {p['name']:13s} keine Config lief -- uebersprungen")
            continue
        winners[p["name"]] = strategies.sig(cand)
        ranked_of[p["name"]] = {strategies.sig(c): c for c in ranked}
        measure_of[p["name"]] = measure
        peak[p["name"]] = tf
        log(f"  {p['name']:13s} {tf:6.2f} TFLOPS  {cand.label()}")

    names = sorted(winners)
    pool = set(winners.values()) | {tuple(DEFAULT_CONFIG[k] for k in
                                          ("m_prim", "n_prim", "k_prim", "m_l2", "n_l2",
                                           "order", "variant"))}
    if pool_mode == "neighbours":
        pool = set().union(*(neighbours(s) for s in pool))
    # nur Configs, die auf ALLEN Shapes ueberhaupt im Kandidatenpool sind
    pool = [s for s in pool if all(s in ranked_of[n] for n in names)]
    log(f"\nPhase 2: {len(pool)} Kandidaten auf {len(names)} Shapes "
        f"= {len(pool)*len(names)} Messungen")

    scores = {}
    for s in pool:
        fracs = []
        for n in names:
            tf = measure_of[n](ranked_of[n][s])
            n_meas += 1
            if tf is None:
                fracs = None
                break
            fracs.append(tf / peak[n])
        if fracs:
            scores[s] = geo(fracs)

    if not scores:
        log("keine Config lief auf allen Shapes durch")
        return
    best = max(scores, key=scores.get)
    log("")
    log(f"{'Rang':>4s} {'Anteil':>8s}  Config")
    for i, s in enumerate(sorted(scores, key=scores.get, reverse=True)[:10], 1):
        log(f"{i:4d} {100*scores[s]:7.1f} %  m_prim={s[0]} n_prim={s[1]} k_prim={s[2]} "
            f"m_l2={s[3]} n_l2={s[4]} order={s[5]} variant={s[6]}")

    log(f"\n-> Eintrag fuer BASELINE_CONFIGS in problems.py:")
    log(f'    "{DEV.gpu_name}": dict(variant="{best[6]}", m_prim={best[0]}, '
        f"n_prim={best[1]}, k_prim={best[2]},")
    log(f"                          m_l2={best[3]}, n_l2={best[4]}, order={best[5]}),")
    log(f"\nGruppen-Ausdehnung: {best[3]*best[0]} x {best[4]*best[1]}   "
        f"Akkumulator: {best[0]*best[1]} fp32")
    log(f"Messungen gesamt: {n_meas}")

    results_io.write_csv(f"baseline_probe_{pool_mode}.csv",
                         ["m_prim", "n_prim", "k_prim", "m_l2", "n_l2", "order",
                          "variant", "fraction"],
                         [[*s, f"{scores[s]:.4f}"] for s in
                          sorted(scores, key=scores.get, reverse=True)])
    results_io.write_log(lines, f"baseline_probe_{pool_mode}.log")


if __name__ == "__main__":
    mode = "neighbours" if "--pool" in sys.argv and "neighbours" in sys.argv else "winners"
    if not torch.cuda.is_available():
        print("keine CUDA-GPU")
        raise SystemExit(1)
    main(mode)
