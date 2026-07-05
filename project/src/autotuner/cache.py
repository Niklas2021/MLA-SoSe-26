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


def store(einsum, shapes, gpu_name, config, tflops):
    data = load()
    data[make_key(einsum, shapes, gpu_name)] = {"config": config, "tflops": round(tflops, 3)}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    # Selbsttest ohne GPU
    cfg = dict(variant="A", m_prim=128, n_prim=128, k_prim=32, m_l2=4, n_l2=2)
    store("cmk,ckn->cmn", [(4, 4096, 4096), (4, 4096, 4096)], "TEST-GPU", cfg, 66.6)
    hit = lookup("cmk, ckn -> cmn", [(4, 4096, 4096), (4, 4096, 4096)], "TEST-GPU")
    print("hit:", hit)
    assert hit and hit["config"]["k_prim"] == 32
    assert lookup("cmk,ckn->cmn", [(1, 1, 1)], "TEST-GPU") is None  # anderer Key
    print("cache ok:", CACHE_FILE)
