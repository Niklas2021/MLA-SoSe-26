import sys
import os
import importlib.util

# generate_config aus 05/task2 via importlib laden (Namenskonflikt mit dieser Datei vermeiden)
_src05 = '/home/mla03/build/05_assignment/src'
_src06 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src05)
_spec = importlib.util.spec_from_file_location('a05_task2', os.path.join(_src05, 'task2.py'))
_mod = importlib.util.module_from_spec(_spec)
sys.modules['a05_task2'] = _mod
_spec.loader.exec_module(_mod)
generate_config = _mod.generate_config
sys.path.insert(0, _src06)  # 06/src nach 05-Import wieder nach vorne


# Tensorformen aus den Daten (lf_tr_64_intermediate.npz)
# tensor_acspx: (a, c, s, p, x) = (4, 3, 64, 64, 1536)
# tensor_bspy:  (b, s, p, y)    = (4, 64, 64, 1536)
SHAPE_ACSPX = (4, 3, 64, 64, 1536)
SHAPE_BSPY  = (4, 64, 64, 1152)
EINSUM      = 'acspx,bspy->abcyx'


def print_config(cfg, label=""):
    if label:
        print(f"--- {label} ---")
    print(f"  data_type:  {cfg.data_type.name}")
    print(f"  prim_main:  {cfg.prim_main.name}")
    print(f"  prim_last:  {cfg.prim_last.name}")
    print(f"  prim_first: {cfg.prim_first.name}")
    print(f"  dim_types:  {[d.name for d in cfg.dim_types]}")
    print(f"  exec_types: {[e.name for e in cfg.exec_types]}")
    print(f"  dim_sizes:  {cfg.dim_sizes}")
    print("  strides:")
    for i, s in enumerate(cfg.strides):
        print(f"    tensor {i}: {s}")
    print()


def task2():
    cfg = generate_config(EINSUM, [SHAPE_ACSPX, SHAPE_BSPY])
    print_config(cfg, "Task 2 – initiale Config")
    return cfg


if __name__ == "__main__":
    task2()
