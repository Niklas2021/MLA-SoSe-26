# Praktischer Tuner: liefert fuer (einsum, shapes) die beste Config. Erst im Cache
# nachsehen, sonst suchen und die schnellste cachen. Zwei Strategien:
#   topk   -- nur die Modell-Top-k messen (v2: Register-Filter + Bandbreite). ~7
#             Messungen, ~97 % des Optimums.
#   hybrid -- Top-k als Startpunkt, dann Koordinatenabstieg ueber die Knopf-Achsen.
#             ~21 Messungen, ~99 %. Default, siehe simulate_search.py.
# Braucht die GPU (misst via run_candidate). Auf der GB10:  python autotune.py
from autotuner.search import (enumerate_candidates, prune, rank, SearchSpace,
                              estimate_acc_registers, build_one_config)
from autotuner.einsum_parser import parse_einsum
from autotuner import cache, strategies

REG_FRACTION = 0.4


def _batch_and_flops(e):
    out_vol = 1
    for c in e.out:
        out_vol *= e.size_of[c]
    k_vol = 1
    for c in e.k_chars:
        k_vol *= e.size_of[c]
    batch = 1
    for c in e.batch_chars:
        batch *= e.size_of[c]
    return batch, 2.0 * out_vol * k_vol


def _knobs(c):
    return dict(variant=c.variant, m_prim=c.m_prim, n_prim=c.n_prim,
                k_prim=c.k_prim, m_l2=c.m_l2, n_l2=c.n_l2, order=c.order)


def _sig_of(config):
    # Knopf-Dict (aus dem Cache) -> Signatur wie strategies.sig
    # order fehlt in Alt-Eintraegen -> 0, das war ja das damalige Verhalten
    return (config["m_prim"], config["n_prim"], config["k_prim"],
            config["m_l2"], config["n_l2"], config.get("order", 0), config["variant"])


def candidate_from_config(einsum, shapes, config):
    # gecachte Knoepfe zurueck zu einem Candidate (fuer run_candidate)
    e = parse_einsum(einsum, shapes)
    return build_one_config(e, config["variant"], config["m_prim"], config["n_prim"],
                            config["k_prim"], config["m_l2"], config["n_l2"],
                            order=config.get("order", 0))


def model_ranked(einsum, shapes, dev, space=None):
    # v2-Vorfilter: Register-Fresser raus, dann nach Bandbreiten-Traffic ranken.
    # Gibt den ganzen Pool zurueck (sortiert) -- der Abstieg braucht auch die hinteren.
    e = parse_einsum(einsum, shapes)
    batch, _ = _batch_and_flops(e)
    cands, _ = enumerate_candidates(einsum, shapes, space)
    kept, _ = prune(cands, dev)
    pool = [c for c in kept if estimate_acc_registers(c) <= REG_FRACTION * dev.regs_per_block]
    return [c for c, _ in rank(pool, dev, batch=batch, model="bw")]


def model_topk(einsum, shapes, dev, k=7, space=None):
    return model_ranked(einsum, shapes, dev, space)[:k]


def _make_measure(einsum, shapes, flops, warmup, rep):
    # Callback fuer die Strategien: kompilieren, gegen torch pruefen, benchen.
    # Kaputte/falsche Configs geben None -- die Strategie ueberspringt sie dann.
    import torch
    import triton.testing
    from autotuner.kernels import run_candidate

    torch.manual_seed(0)
    A = torch.randn(shapes[0], dtype=torch.float16, device="cuda")
    B = torch.randn(shapes[1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(einsum.replace(" ", ""), A.float(), B.float()).half()

    def measure(cand):
        try:
            out = run_candidate(cand, A, B)
            torch.cuda.synchronize()
            if not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                return None
            ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                         warmup=warmup, rep=rep)
        except Exception:
            return None
        return flops / (ms * 1e-3) / 1e12

    return measure


def autotune(einsum, shapes, dev, k=7, use_cache=True, warmup=50, rep=200,
             strategy="hybrid", space=None):
    # -> (config-dict, tflops, aus_cache)
    space_size = (space or SearchSpace()).size()
    hit = cache.lookup(einsum, shapes, dev.gpu_name) if use_cache else None
    if hit and cache.good_enough(hit, strategy, space_size):
        return hit["config"], hit["tflops"], True

    e = parse_einsum(einsum, shapes)
    _, flops = _batch_and_flops(e)
    measure = _make_measure(einsum, shapes, flops, warmup, rep)
    ranked = model_ranked(einsum, shapes, dev, space)

    if strategy == "hybrid":
        # ein zu schwacher Cache-Treffer taugt nicht als Antwort, aber sehr wohl als
        # Startpunkt -- so kann ein Upgrade (topk->hybrid, eng->weit) nie schlechter
        # ausgehen als der Lauf davor
        seeds = [_sig_of(hit["config"])] if hit else []
        cand, tf, n = strategies.hybrid(ranked, measure, k=k, extra_seeds=seeds)
    elif strategy == "topk":
        cand, tf, n = strategies.model_topk_only(ranked, measure, k=k)
    elif strategy == "full":
        cand, tf, n = strategies.full_sweep(ranked, measure)
    else:
        raise ValueError(f"unbekannte Strategie: {strategy!r}")

    if cand is None:
        raise RuntimeError("keine Config lief korrekt durch")
    cache.store(einsum, shapes, dev.gpu_name, _knobs(cand), tf, strategy, n, space_size)
    return _knobs(cand), tf, False


if __name__ == "__main__":
    # python autotune.py [topk|hybrid|full] [--wide] [--ordered]
    #   --wide    nimmt zusaetzlich die kleinen Tiles (32/16) dazu -- lohnt auf GPUs, wo
    #             die Optima am unteren Gitterrand kleben (3070).
    #   --ordered nimmt die vier Traversierungs-Reihenfolgen dazu (M5.2).
    #   Beides kombinierbar. Default ist der enge Raum mit order=0, damit der Lauf
    #   mit der bisherigen Auswertung vergleichbar bleibt.
    import sys
    import time

    import torch
    from autotuner import search as search_mod
    from problems import PROBLEMS
    try:
        from autotuner.device_props import get_device_properties
        dev = get_device_properties()
    except Exception:
        from autotuner.device_props import GB10 as dev
    if not torch.cuda.is_available():
        print("keine CUDA-GPU")
        raise SystemExit(1)
    import datetime
    import results_io

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strat = args[0] if args else "hybrid"
    wide, ordered = "--wide" in sys.argv, "--ordered" in sys.argv
    kw = {}
    if wide:
        kw = dict(m_prim_choices=list(search_mod.WIDE_MN_PRIM_CHOICES),
                  n_prim_choices=list(search_mod.WIDE_MN_PRIM_CHOICES),
                  k_prim_choices=list(search_mod.WIDE_K_PRIM_CHOICES))
    if ordered:
        kw["orders"] = list(search_mod.ORDER_CHOICES)
    space = SearchSpace(**kw) if kw else None
    size = (space or SearchSpace()).size()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log(f"=== M5.1 Tuner-Lauf ({strat}) ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log(f"GPU: {dev.gpu_name}   Strategie: {strat}   Suchraum: {size}")
    log()

    rows = []
    for p in PROBLEMS:
        t0 = time.perf_counter()
        cfg, tf, _ = autotune(p["einsum"], p["shapes"], dev, strategy=strat, space=space)
        dt = time.perf_counter() - t0
        _, _, hit = autotune(p["einsum"], p["shapes"], dev, strategy=strat, space=space)
        entry = cache.lookup(p["einsum"], p["shapes"], dev.gpu_name)
        n = entry["n_measured"]
        log(f"{p['name']:12s} {tf:6.1f} TFLOPS  {n:3d} Messungen  "
            f"{dt:5.1f} s  (2. Aufruf aus Cache: {hit})")
        # space_size = nominale Knopf-Kombinatorik (entscheidet die Cache-Vertraeglichkeit),
        # n_candidates = was nach Enumeration/Pruning/Register-Filter uebrig bleibt und
        # die Strategie wirklich sieht. Bei den Ring-Shapes klaffen die auseinander,
        # weil Variante B dort komplett wegfaellt.
        n_cand = len(model_ranked(p["einsum"], p["shapes"], dev, space))
        rows.append([p["name"], p["einsum"].replace(" ", ""), strat, size, n_cand,
                     cfg["variant"], cfg["m_prim"], cfg["n_prim"], cfg["k_prim"],
                     cfg["m_l2"], cfg["n_l2"], cfg["order"], f"{tf:.3f}", n, f"{dt:.2f}"])

    # Artefakte fuers Repo: Suffix, damit ein --wide-Lauf den engen nicht ueberschreibt
    tag = strat + ("_wide" if wide else "") + ("_ordered" if ordered else "")
    results_io.write_csv(f"autotune_{tag}.csv",
                         ["name", "einsum", "strategy", "space_size", "n_candidates",
                          "variant", "m_prim", "n_prim", "k_prim", "m_l2", "n_l2",
                          "order", "tflops", "n_measured", "wall_s"], rows)
    results_io.write_log(lines, f"autotune_{tag}.log")
