# Praktischer Tuner: liefert fuer (einsum, shapes) die beste Config. Erst im Cache
# nachsehen, sonst die Modell-Top-k messen (v2: Register-Filter + Bandbreite, der beste
# Top-k-Vorfilter aus der Ranking-Studie) und die schnellste cachen.
# Braucht die GPU (misst via run_candidate). Auf der GB10:  python autotune.py
from autotuner.search import (enumerate_candidates, prune, rank,
                              estimate_acc_registers, build_one_config)
from autotuner.einsum_parser import parse_einsum
from autotuner import cache

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
                k_prim=c.k_prim, m_l2=c.m_l2, n_l2=c.n_l2)


def candidate_from_config(einsum, shapes, config):
    # gecachte Knoepfe zurueck zu einem Candidate (fuer run_candidate)
    e = parse_einsum(einsum, shapes)
    return build_one_config(e, config["variant"], config["m_prim"], config["n_prim"],
                            config["k_prim"], config["m_l2"], config["n_l2"])


def model_topk(einsum, shapes, dev, k=7):
    # v2-Vorfilter: Register-Fresser raus, dann nach Bandbreiten-Traffic ranken
    e = parse_einsum(einsum, shapes)
    batch, _ = _batch_and_flops(e)
    cands, _ = enumerate_candidates(einsum, shapes)
    kept, _ = prune(cands, dev)
    pool = [c for c in kept if estimate_acc_registers(c) <= REG_FRACTION * dev.regs_per_block]
    ranked = rank(pool, dev, batch=batch, model="bw")
    return [c for c, _ in ranked][:k]


def autotune(einsum, shapes, dev, k=7, use_cache=True, warmup=50, rep=200):
    # -> (config-dict, tflops, aus_cache)
    if use_cache:
        hit = cache.lookup(einsum, shapes, dev.gpu_name)
        if hit:
            return hit["config"], hit["tflops"], True

    import torch
    import triton.testing
    from autotuner.kernels import run_candidate

    e = parse_einsum(einsum, shapes)
    _, flops = _batch_and_flops(e)
    torch.manual_seed(0)
    A = torch.randn(shapes[0], dtype=torch.float16, device="cuda")
    B = torch.randn(shapes[1], dtype=torch.float16, device="cuda")
    ref = torch.einsum(einsum.replace(" ", ""), A.float(), B.float()).half()

    best = None
    for cand in model_topk(einsum, shapes, dev, k):
        try:
            out = run_candidate(cand, A, B)
            torch.cuda.synchronize()
            if not torch.allclose(out, ref, rtol=1e-2, atol=1e-1):
                continue
            ms = triton.testing.do_bench(lambda: run_candidate(cand, A, B),
                                         warmup=warmup, rep=rep)
        except Exception:
            continue
        tf = flops / (ms * 1e-3) / 1e12
        if best is None or tf > best[1]:
            best = (_knobs(cand), tf)

    if best is None:
        raise RuntimeError("keine der Top-k-Configs lief korrekt durch")
    cache.store(einsum, shapes, dev.gpu_name, best[0], best[1])
    return best[0], best[1], False


if __name__ == "__main__":
    import torch
    from problems import PROBLEMS
    try:
        from autotuner.device_props import get_device_properties
        dev = get_device_properties()
    except Exception:
        from autotuner.device_props import GB10 as dev
    if not torch.cuda.is_available():
        print("keine CUDA-GPU")
        raise SystemExit(1)
    print(f"GPU: {dev.gpu_name}")
    for p in PROBLEMS:
        cfg, tf, _ = autotune(p["einsum"], p["shapes"], dev)      # tunt + cacht
        _, _, hit = autotune(p["einsum"], p["shapes"], dev)        # 2. Aufruf: Cache
        print(f"{p['name']:12s} {tf:6.1f} TFLOPS  {cfg}  (2. Aufruf aus Cache: {hit})")
