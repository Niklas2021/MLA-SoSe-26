# Isoliert den Reihenfolge-Knopf (M5.2): dieselbe Tile-Config, alle vier Ordnungen,
# direkt hintereinander gemessen.
#
# Warum eigens: ein --ordered-Lauf gegen einen normalen zu vergleichen ist auf der
# GB10 nicht sauber. Die beiden Laeufe dauern unterschiedlich lang und liegen zeitlich
# auseinander, und wir haben gemessen, dass dieselbe Config je nach Lastdauer bis zu
# 4.6 % auseinanderliegt (siehe M5.1). Ein Effekt von ~2 % verschwindet darin.
# Hier wird deshalb im Round-Robin gemessen (0,1,2,3, 0,1,2,3, ...) und pro Ordnung der
# Median genommen -- damit hebt sich eine monotone Drift ueber die Laufzeit weg.
#
# Auf der GB10:  python measure_order.py
import datetime

import torch
import triton.testing

import results_io
from autotuner.search import build_one_config, ORDER_CHOICES
from autotuner.einsum_parser import parse_einsum
from autotuner.kernels import run_candidate
from problems import PROBLEMS, DEFAULT_CONFIG

from autotuner.device_props import resolve_device_properties

DEV = resolve_device_properties()

ROUNDS = 3
WARMUP, REP = 25, 100


def flops_of(e):
    out_vol = 1
    for c in e.out:
        out_vol *= e.size_of[c]
    k_vol = 1
    for c in e.k_chars:
        k_vol *= e.size_of[c]
    return 2.0 * out_vol * k_vol


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    lines, rows = [], []

    def log(m=""):
        print(m)
        lines.append(m)

    log("=== M5.2 Reihenfolge-Knopf isoliert ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"GPU: {DEV.gpu_name}   {ROUNDS} Runden Round-Robin, Median je Ordnung")
    log("")
    hdr = f"{'Shape':13s} {'Tiles':>20s} " + "".join(f"{'order='+str(o):>10s}" for o in ORDER_CHOICES) + f" {'best/o0':>8s}"
    log(hdr)
    log("-" * len(hdr))

    gains = []
    for p in PROBLEMS:
        e = parse_einsum(p["einsum"], p["shapes"])
        flops = flops_of(e)
        torch.manual_seed(0)
        A = torch.randn(p["shapes"][0], dtype=torch.float16, device="cuda")
        B = torch.randn(p["shapes"][1], dtype=torch.float16, device="cuda")
        ref = torch.einsum(p["einsum"].replace(" ", ""), A.float(), B.float()).half()

        # feste Tiles: die Default-Config (Variante A -- nur die hat den Knopf)
        knobs = dict(DEFAULT_CONFIG)
        knobs["variant"] = "A"
        cands = {}
        for o in ORDER_CHOICES:
            try:
                cands[o] = build_one_config(e, "A", knobs["m_prim"], knobs["n_prim"],
                                            knobs["k_prim"], knobs["m_l2"], knobs["n_l2"],
                                            order=o)
            except Exception:
                cands[o] = None

        samples = {o: [] for o in ORDER_CHOICES}
        for _ in range(ROUNDS):
            for o in ORDER_CHOICES:          # Round-Robin gegen Drift
                c = cands[o]
                if c is None:
                    continue
                try:
                    out = run_candidate(c, A, B)
                    torch.cuda.synchronize()
                    if not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                        continue
                    ms = triton.testing.do_bench(lambda: run_candidate(c, A, B),
                                                 warmup=WARMUP, rep=REP)
                except Exception:
                    continue
                samples[o].append(flops / (ms * 1e-3) / 1e12)

        med = {o: (median(v) if v else 0.0) for o, v in samples.items()}
        if not med.get(0):
            log(f"{p['name']:13s} order=0 lief nicht -- uebersprungen")
            continue
        best_o = max(med, key=med.get)
        gain = med[best_o] / med[0]
        gains.append(gain)
        tiles = f"{knobs['m_prim']}/{knobs['n_prim']}/{knobs['k_prim']} {knobs['m_l2']}x{knobs['n_l2']}"
        log(f"{p['name']:13s} {tiles:>20s} " +
            "".join(f"{med[o]:10.2f}" for o in ORDER_CHOICES) + f" {gain:7.3f}x")
        rows.append([p["name"], p["einsum"].replace(" ", ""), tiles] +
                    [f"{med[o]:.3f}" for o in ORDER_CHOICES] + [best_o, f"{gain:.4f}"])

    if gains:
        import math
        geo = math.exp(sum(math.log(g) for g in gains) / len(gains))
        log("")
        log(f"beste Ordnung / order=0: geom. Mittel {geo:.3f}x, "
            f"max {max(gains):.3f}x, min {min(gains):.3f}x")
    results_io.write_csv("order_isolated.csv",
                         ["name", "einsum", "tiles"] +
                         [f"order{o}" for o in ORDER_CHOICES] + ["best_order", "gain"], rows)
    results_io.write_log(lines, "order_isolated.log")


if __name__ == "__main__":
    main()
