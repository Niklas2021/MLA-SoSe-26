# Smoke-Test: laeuft die Toolchain (ct.kernel + ct.launch + do_bench) auf der GPU
# und welche Properties hat die Karte. Log nach results/smoke_test.log.
import datetime
import os
import sys

import torch
import triton.testing
import cuda.tile as ct


TILE = 1024
N = 1 << 20  # 1M Elemente


@ct.kernel
def add_kernel(A, B, C, num_tiles: ct.Constant[int]):
    pid = ct.bid(0)
    a = ct.load(A, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(B, index=(pid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(C, index=(pid,), tile=a + b)


def ceildiv(a, b):
    return (a + b - 1) // b


def run_add(A, B):
    n = A.shape[0]
    num_tiles = ceildiv(n, TILE)
    pad = num_tiles * TILE
    C_pad = torch.zeros((pad,), dtype=A.dtype, device="cuda")
    A_pad = torch.zeros((pad,), dtype=A.dtype, device="cuda")
    B_pad = torch.zeros((pad,), dtype=B.dtype, device="cuda")
    A_pad[:n] = A
    B_pad[:n] = B
    ct.launch(torch.cuda.current_stream(), (num_tiles,), add_kernel,
              (A_pad, B_pad, C_pad, num_tiles))
    return C_pad[:n]


def cutile_version():
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(ct, attr, None)
        if v:
            return str(v)
    try:
        from importlib.metadata import version
        for pkg in ("cuda-tile", "cuda_tile", "nvidia-cuda-tile"):
            try:
                return version(pkg)
            except Exception:
                continue
    except Exception:
        pass
    return "unbekannt"


def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=== M0 Smoke-Test der Umgebung ===")
    log(f"Zeit: {datetime.datetime.now().isoformat(timespec='seconds')}")
    log("")

    if not torch.cuda.is_available():
        log("FEHLER: torch.cuda.is_available() == False -- keine CUDA-GPU sichtbar.")
        _write(lines)
        sys.exit(1)

    # Environment + Hardware
    props = torch.cuda.get_device_properties(0)
    l2_bytes = getattr(props, "L2_cache_size", 0)
    log("--- Environment ---")
    log(f"torch:            {torch.__version__}")
    log(f"CUDA (torch):     {torch.version.cuda}")
    log(f"cuTile:           {cutile_version()}")
    log(f"GPU-Modell:       {props.name}")
    log(f"Compute Cap.:     {props.major}.{props.minor}")
    log(f"SM-Count:         {props.multi_processor_count}")
    log(f"Global Mem:       {props.total_memory / 1e9:.1f} GB")
    log(f"L2-Cache-Groesse: {l2_bytes} Byte ({l2_bytes / 1e6:.2f} MB)")
    log("")

    # verification: ct.kernel + ct.launch 
    log("--- Kernel-Smoke (ct.kernel + ct.launch) ---")
    torch.manual_seed(0)
    A = torch.randn((N,), dtype=torch.float16, device="cuda")
    B = torch.randn((N,), dtype=torch.float16, device="cuda")
    C = run_add(A, B)
    torch.cuda.synchronize()
    max_err = (C.float() - (A.float() + B.float())).abs().max().item()
    ok = torch.allclose(C, A + B, rtol=1e-2, atol=1e-2)
    log(f"vektor-add ({N} elem, TILE={TILE}): max_err={max_err:.5f}  allclose={ok}")
    if not ok:
        log("FEHLER: Kernel-Ergebnis weicht ab")
        _write(lines)
        sys.exit(1)
    log("")

    # do_bench
    log("--- triton.testing.do_bench ---")
    for _ in range(3):
        run_add(A, B)
    torch.cuda.synchronize()
    ms = triton.testing.do_bench(lambda: run_add(A, B), warmup=50, rep=200)
    gbps = 3 * N * A.element_size() / (ms * 1e-3) / 1e9  # 2 read + 1 write
    log(f"do_bench: {ms:.4f} ms   ~{gbps:.1f} GB/s (mem-bound add)")
    log("")
    log("OK: Toolchain laeuft (ct.kernel + ct.launch + do_bench).")

    _write(lines)


def _write(lines):
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(results_dir, "smoke_test.log"))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[Log geschrieben in: {path}]")


if __name__ == "__main__":
    main()
