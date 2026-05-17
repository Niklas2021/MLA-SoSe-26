import sys
import os
import importlib.util

# Optimizer aus 05/task3 via importlib (Namenskonflikt mit 06/task3.py vermeiden)
_src05 = '/home/mla03/build/05_assignment/src'
_src06 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src05)
_spec = importlib.util.spec_from_file_location('a05_task3', os.path.join(_src05, 'task3.py'))
_mod = importlib.util.module_from_spec(_spec)
sys.modules['a05_task3'] = _mod
_spec.loader.exec_module(_mod)
Optimizer = _mod.Optimizer
sys.path.insert(0, _src06)  # 06/src nach 05-Import wieder nach vorne
from task2 import task2, print_config


# Tile-Größen
PRIM_M = 128
PRIM_N = 128
PRIM_K = 64   # p ist bereits 64 -> direkt als prim_k, s bleibt k_seq

# L2-Super-Tile: wie viele prim-Blöcke pro Richtung
# x=1536 -> 12 Blöcke -> 6 * 2 * 128
# y=1152 ->  9 Blöcke -> 3 * 3 * 128
M_L2 = 2
N_L2 = 3


def task3():
    cfg = task2()
    opt = Optimizer(cfg)

    # x(1536) -> (x_outer=12, prim_m), dann (x_l2_outer=6, x_l2)
    opt.split_dim(4, 12, PRIM_M)
    opt.split_dim(4, 6, M_L2)

    # y(1152) -> (y_outer=9, prim_n), dann (y_l2_outer=3, y_l2); p bleibt prim_k
    opt.split_dim(8, 9, PRIM_N)
    opt.split_dim(8, 3, N_L2)

    # [par..., m_l2, n_l2, k_seq, prim_m, prim_n, prim_k]
    opt.permute_dims([0, 1, 4, 7, 8, 5, 9, 2, 6, 10, 3])
    opt.make_executable()

    print_config(cfg, "Task 3 – optimierte Config")
    return cfg


if __name__ == "__main__":
    task3()
