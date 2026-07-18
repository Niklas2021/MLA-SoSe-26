# Vergleicht die Suchstrategien offline gegen die Vollmessungen: jede "Messung" ist
# ein Lookup in der CSV statt ein Kernel-Start. Kostet keine GPU-Zeit und benutzt
# denselben Code wie der echte Tuner (autotuner/strategies.py).
# Lokal:  python simulate_search.py            (result_dgx)
#         RESULTS_DIR=../result_3070/results python simulate_search.py
import csv
import os
from collections import defaultdict

from autotuner.search import (enumerate_candidates, prune, rank,
                              estimate_acc_registers)
from autotuner.einsum_parser import parse_einsum
from autotuner.device_props import GB10, RTX3070
from autotuner import strategies
from problems import PROBLEMS, DEFAULT_CONFIG

REG_FRACTION_V2 = 0.4
def _default_results_dir():
    here = os.path.dirname(__file__)
    for name in ("result_dgx", "result_dgx_v1", "results"):
        d = os.path.join(here, "..", name)
        if os.path.isdir(d):
            return d
    return os.path.join(here, "..", "result_dgx")


RESULTS_DIR = os.environ.get("RESULTS_DIR", _default_results_dir())
DEV = RTX3070 if "3070" in RESULTS_DIR else GB10
DEFAULT_SIG = tuple(DEFAULT_CONFIG[k] for k in
                    ("m_prim", "n_prim", "k_prim", "m_l2", "n_l2", "order", "variant"))


def load_csv(name):
    path = os.path.join(RESULTS_DIR, f"tune_{name}.csv")
    if not os.path.exists(path):
        return None
    meas = {}
    for r in csv.DictReader(open(path)):
        if not int(r["ok"]):
            continue
        # order fehlt in den alten CSVs -> 0 (so wurde damals gemessen)
        meas[(int(r["m_prim"]), int(r["n_prim"]), int(r["k_prim"]),
              int(r["m_l2"]), int(r["n_l2"]), int(r.get("order") or 0),
              r["variant"])] = float(r["tflops"])
    return meas


def model_ranked(problem, meas):
    # v2-Vorfilter wie in autotune.py, aber nur ueber Configs, fuer die es auch eine
    # Messung gibt (sonst simuliert man Punkte, die es real nicht gab)
    e = parse_einsum(problem["einsum"], problem["shapes"])
    batch = 1
    for c in e.batch_chars:
        batch *= e.size_of[c]
    cands, _ = enumerate_candidates(problem["einsum"], problem["shapes"])
    kept, _ = prune(cands, DEV)
    pool = [c for c in kept
            if estimate_acc_registers(c) <= REG_FRACTION_V2 * DEV.regs_per_block
            and strategies.sig(c) in meas]
    return [c for c, _ in rank(pool, DEV, batch=batch, model="bw")]


def main():
    print(f"Ergebnisse aus {os.path.abspath(RESULTS_DIR)}  (dev: {DEV.gpu_name})\n")
    hdr = (f"{'Shape':13s} {'absBest':>8s} {'Default':>8s} | {'top7':>7s} {'n':>3s} | "
           f"{'hybrid':>7s} {'n':>3s} | {'voll':>5s}")
    print(hdr)
    print("-" * len(hdr))

    agg = defaultdict(list)
    for p in PROBLEMS:
        meas = load_csv(p["name"])
        if not meas:
            continue
        absbest = max(meas.values())
        ranked = model_ranked(p, meas)
        if len(ranked) < 2:
            continue
        measure = lambda c: meas.get(strategies.sig(c))

        _, t7, n7 = strategies.model_topk_only(ranked, measure, k=7)
        _, th, nh = strategies.hybrid(ranked, measure, k=7)
        deflt = meas.get(DEFAULT_SIG, float("nan"))

        print(f"{p['name']:13s} {absbest:8.1f} {deflt:8.1f} | "
              f"{100*t7/absbest:6.1f}% {n7:3d} | {100*th/absbest:6.1f}% {nh:3d} | "
              f"{len(meas):5d}")
        agg["top7"].append(t7 / absbest); agg["n7"].append(n7)
        agg["hyb"].append(th / absbest); agg["nh"].append(nh)
        agg["deflt"].append(deflt / absbest)
        agg["full"].append(len(meas))

    if not agg:
        print("keine CSVs gefunden")
        return

    def mean(k):
        return sum(agg[k]) / len(agg[k])

    print(f"\nSchnitt ueber {len(agg['top7'])} Shapes:")
    print(f"  Default (0 Messungen)          {100*mean('deflt'):5.1f} %")
    print(f"  Modell-Top-7 ({mean('n7'):.0f} Messungen)     {100*mean('top7'):5.1f} %")
    print(f"  Hybrid ({mean('nh'):.0f} Messungen)          {100*mean('hyb'):5.1f} %")
    print(f"  Vollmessung ({mean('full'):.0f} Messungen)  100.0 %")


if __name__ == "__main__":
    main()
