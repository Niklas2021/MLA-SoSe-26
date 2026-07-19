# Suchstrategien ueber dem geprunten Kandidatenraum. Reines Python, kein cuTile:
# gemessen wird ueber eine measure-Callback, die von aussen reinkommt. Dadurch laeuft
# derselbe Code gegen die GPU (autotune.py) und gegen die CSVs (simulate_search.py).
AXES = ["m_prim", "n_prim", "k_prim", "m_l2", "n_l2", "order", "variant"]
IDX = {a: i for i, a in enumerate(AXES)}


def sig(cand):
    return (cand.m_prim, cand.n_prim, cand.k_prim, cand.m_l2, cand.n_l2,
            getattr(cand, "order", 0), cand.variant)


def _swap(s, axis, val):
    t = list(s)
    t[IDX[axis]] = val
    return tuple(t)


class Search:
    # haelt den Kandidatenpool und zaehlt die echten Messungen (memoisiert, damit
    # ein zweimal besuchter Punkt nicht doppelt zaehlt)
    def __init__(self, candidates, measure):
        self.by_sig = {sig(c): c for c in candidates}
        self.measure = measure
        self.done = {}

    def eval(self, s):
        # -> tflops, oder None wenn die Config nicht im Pool ist / nicht lief
        if s in self.done:
            return self.done[s]
        cand = self.by_sig.get(s)
        val = self.measure(cand) if cand is not None else None
        self.done[s] = val
        return val

    def values(self, axis):
        return sorted({s[IDX[axis]] for s in self.by_sig}, key=str)

    @property
    def n_measured(self):
        return sum(1 for v in self.done.values() if v is not None)


# Reihenfolge nach gemessener Wichtigkeit (Ablation ueber 16 Shapes): die Prim-Form
# dominiert, danach kommen k_prim, L2-Gruppe und Variante. m_prim/n_prim gemeinsam,
# weil sie ueber das Registerbudget zusammenhaengen -- einzeln laeuft der Abstieg in
# 256x256 rein und bleibt dort haengen.
DEFAULT_ORDER = ["m_prim", "k_prim", "m_l2", "n_l2", "order", "variant"]


def descend(search, seed, best, order=None, passes=3, joint_mn=True):
    # Koordinatenabstieg: eine Achse variieren, Rest fest, Bestes behalten. Wiederholt,
    # bis sich in einem vollen Durchlauf nichts mehr aendert (Achsen sind nicht
    # unabhaengig, ein Pass reicht also nicht immer).
    order = order or DEFAULT_ORDER
    cur = seed
    for _ in range(passes):
        changed = False
        for axis in order:
            if joint_mn and axis == "m_prim":
                cands = [_swap(_swap(cur, "m_prim", m), "n_prim", n)
                         for m in search.values("m_prim") for n in search.values("n_prim")]
            elif joint_mn and axis == "n_prim":
                continue
            else:
                cands = [_swap(cur, axis, v) for v in search.values(axis)]
            for c in cands:
                t = search.eval(c)
                if t is not None and t > best:
                    best, cur, changed = t, c, True
        if not changed:
            break
    return cur, best


def hybrid(ranked_candidates, measure, k=7, order=None, passes=3, extra_seeds=()):
    # Modell-Top-k messen (breiter Startpunkt), dann von der gemessen besten aus
    # absteigen. Das Modell taugt nicht als Ranker, aber als Startpunktlieferant --
    # und der Abstieg korrigiert, wo es danebenliegt.
    #
    # extra_seeds sind zusaetzliche Startpunkte, typisch die schon bekannte Beste aus
    # einem frueheren (engeren oder schwaecheren) Lauf. Ohne die kann ein groesserer
    # Suchraum das Ergebnis VERSCHLECHTERN: mehr Optionen aendern auch die Seeds und
    # damit den Abstiegspfad, und Greedy landet woanders (auf der GB10 gemessen:
    # a06 61.7 -> 49.2 beim Wechsel auf --wide). Mit dem alten Optimum als Seed kann
    # der groessere Raum per Konstruktion nicht mehr schlechter werden.
    # -> (bester Candidate, tflops, Anzahl Messungen)
    search = Search(ranked_candidates, measure)
    starts = [sig(c) for c in ranked_candidates[:k]] + [s for s in extra_seeds]
    seeds = [(search.eval(s), s) for s in dict.fromkeys(starts)]
    seeds = [(t, s) for t, s in seeds if t is not None]
    if not seeds:
        return None, 0.0, search.n_measured
    best, seed = max(seeds)
    cur, best = descend(search, seed, best, order=order, passes=passes)
    return search.by_sig[cur], best, search.n_measured


def model_topk_only(ranked_candidates, measure, k=7):
    # der bisherige Modus (nur Modell-Top-k messen), als Vergleichsbasis
    search = Search(ranked_candidates, measure)
    scored = [(search.eval(sig(c)), c) for c in ranked_candidates[:k]]
    scored = [(t, c) for t, c in scored if t is not None]
    if not scored:
        return None, 0.0, search.n_measured
    best, cand = max(scored, key=lambda x: x[0])
    return cand, best, search.n_measured


def full_sweep(candidates, measure):
    search = Search(candidates, measure)
    scored = [(search.eval(sig(c)), c) for c in candidates]
    scored = [(t, c) for t, c in scored if t is not None]
    if not scored:
        return None, 0.0, search.n_measured
    best, cand = max(scored, key=lambda x: x[0])
    return cand, best, search.n_measured


if __name__ == "__main__":
    # Selbsttest ohne GPU: synthetischer Kandidatenraum mit bekanntem Optimum
    from dataclasses import dataclass

    @dataclass
    class Fake:
        m_prim: int
        n_prim: int
        k_prim: int
        m_l2: int
        n_l2: int
        order: int
        variant: str

    cands = [Fake(m, n, k, ml, nl, o, v)
             for m in (64, 128, 256) for n in (64, 128, 256) for k in (32, 64, 128)
             for ml in (2, 4, 8) for nl in (2, 4, 8) for o in (0, 1) for v in ("A", "B")]
    OPT = (128, 256, 64, 8, 2, 1, "B")

    # Score faellt mit dem Abstand zum Optimum -> achsenweiser Abstieg findet es
    def score(c):
        s = sig(c)
        d = sum(0 if a == b else 1 for a, b in zip(s, OPT))
        return 100.0 - 10 * d

    # absichtlich schlechte Modell-Reihenfolge (Optimum ganz hinten): der Abstieg
    # muss es trotzdem finden, das Top-k allein nicht
    ranked = sorted(cands, key=lambda c: score(c))
    _, t_topk, n_topk = model_topk_only(ranked, score, k=7)
    best, t_hyb, n_hyb = hybrid(ranked, score, k=7)
    print(f"top7   {t_topk:6.1f} ({n_topk} Messungen)")
    print(f"hybrid {t_hyb:6.1f} ({n_hyb} Messungen) -> {sig(best)}")
    assert t_hyb == 100.0 and sig(best) == OPT, "Abstieg findet das Optimum nicht"
    assert t_hyb > t_topk, "Hybrid muss das Top-k schlagen"
    assert n_hyb < len(cands), "Hybrid darf nicht alles messen"

    # Configs, die nicht laufen (measure -> None), duerfen die Suche nicht kippen
    def flaky(c):
        return None if c.m_prim == 256 else score(c)

    best2, t2, _ = hybrid(ranked, flaky, k=7)
    assert best2.m_prim != 256 and t2 == 100.0

    # Monotonie: ein bekannter guter Startpunkt darf nie verschlechtern. Nachbau des
    # GB10-Falls a06 (61.7 -> 49.2 bei --wide), wo der Abstieg ohne Seed in ein
    # schlechteres lokales Optimum lief.
    def rugged(c):
        # zwei Optima; das Modell fuehrt zum schlechteren, das gute ist nur ueber
        # den Seed erreichbar
        s = sig(c)
        near = lambda t: sum(0 if a == b else 1 for a, b in zip(s, t))
        return max(100.0 - 10 * near(OPT), 70.0 - 10 * near((64, 64, 32, 2, 2, 0, "A")))

    known = (128, 256, 64, 8, 4, 1, "B")     # ein Schritt neben OPT
    _, t_ohne, _ = hybrid(ranked, rugged, k=7)
    _, t_mit, _ = hybrid(ranked, rugged, k=7, extra_seeds=[known])
    assert t_mit >= t_ohne, f"Seed hat verschlechtert: {t_mit} < {t_ohne}"
    assert t_mit >= rugged(Fake(*known)), "Seed-Ergebnis unterboten"
    print(f"Monotonie: ohne Seed {t_ohne:.0f}, mit Seed {t_mit:.0f}")
    print("strategies ok")
