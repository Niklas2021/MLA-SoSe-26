# Config-Cache: die getunte Beste pro (einsum, shapes, GPU) als JSON. Damit muss man
# eine Shape nur einmal tunen und liest danach direkt die Config. Das GPU-Modell steht
# im Key, weil die optimale L2-Gruppe von der L2-Groesse abhaengt (andere GPU -> neu tunen).
import json
import os

CACHE_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "cache", "tuned_configs.json"))


def make_key(einsum, shapes, gpu_name):
    shp = ";".join(",".join(str(d) for d in s) for s in shapes)
    return f"{einsum.replace(' ', '')}|{shp}|{gpu_name}"


def load():
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE) as f:
        return json.load(f)


def lookup(einsum, shapes, gpu_name):
    return load().get(make_key(einsum, shapes, gpu_name))


def store(einsum, shapes, gpu_name, config, tflops, strategy="topk", n_measured=None,
          space_size=None):
    data = load()
    data[make_key(einsum, shapes, gpu_name)] = {"config": config,
                                                "tflops": round(tflops, 3),
                                                "strategy": strategy,
                                                "n_measured": n_measured,
                                                "space_size": space_size}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# wie gruendlich ein Eintrag gesucht wurde. Ein Treffer, der schwaecher getunt wurde
# als jetzt verlangt, darf nicht bedienen -- sonst liefert der Cache still das
# schlechtere Ergebnis (top7 ~97 % vs hybrid ~99 %). Alte Eintraege ohne Feld = topk.
STRENGTH = {"topk": 1, "hybrid": 2, "full": 3}


def good_enough(entry, strategy, space_size=None):
    if STRENGTH.get(entry.get("strategy", "topk"), 1) < STRENGTH.get(strategy, 1):
        return False
    # dasselbe Argument fuer den Suchraum: wer im engen Raum getunt hat, hat die
    # kleinen Tiles nie gesehen und darf eine --wide-Anfrage nicht bedienen.
    if space_size is not None:
        return (entry.get("space_size") or 0) >= space_size
    return True


if __name__ == "__main__":
    # Selbsttest ohne GPU
    cfg = dict(variant="A", m_prim=128, n_prim=128, k_prim=32, m_l2=4, n_l2=2)
    store("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)], "TEST-GPU", cfg, 66.6)
    hit = lookup("cmk, ckn -> cmn", [(4, 4096, 4096), (4, 4096, 4096)], "TEST-GPU")
    print("hit:", hit)
    assert hit and hit["config"]["k_prim"] == 32
    assert lookup("cmk,ckn->cmn", [(1, 1, 1)], "TEST-GPU") is None  # anderer Key

    # ein topk-Eintrag darf eine hybrid-Anfrage NICHT bedienen, umgekehrt schon
    assert good_enough({"strategy": "hybrid"}, "topk")
    assert not good_enough({"strategy": "topk"}, "hybrid")
    assert not good_enough({}, "hybrid")          # Altbestand ohne Feld = topk

    # dasselbe fuer den Suchraum: eng getunt bedient keine weite Anfrage
    eng = {"strategy": "hybrid", "space_size": 486}
    assert good_enough(eng, "hybrid", 486)
    assert not good_enough(eng, "hybrid", 2048)
    assert good_enough({"strategy": "hybrid", "space_size": 2048}, "hybrid", 486)
    assert not good_enough({"strategy": "hybrid"}, "hybrid", 486)   # Altbestand
    print("cache ok:", CACHE_FILE)
