import os
# must be set before cuda.tile is first imported – the compiler reads
# this flag during initialisation, not at launch time
os.environ["CUDA_TILE_LOGS"] = "CUTILEIR"

import cuda.tile as ct
import cupy as cp
from math import ceil

# ---- config ----
ROWS   = 2048
COLS   = 64
TILE_H = 64
TILE_W = 64
# ----------------


@ct.kernel
def copy_tile(src, dst, bh: ct.Constant[int], bw: ct.Constant[int]):
    row_blk = ct.bid(0)
    col_blk = ct.bid(1)
    data = ct.load(src, index=(row_blk, col_blk), shape=(bh, bw))
    ct.store(dst, index=(row_blk, col_blk), tile=data)


if __name__ == "__main__":
    src = cp.random.rand(ROWS, COLS).astype(cp.float16)
    dst = cp.zeros_like(src)

    grid = (ceil(ROWS / TILE_H), ceil(COLS / TILE_W), 1)

    print("- launching kernel –\n")
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        copy_tile,
        (src, dst, TILE_H, TILE_W),
    )
    cp.cuda.Device().synchronize()
    print("\n--- kernel finished ---\n")
