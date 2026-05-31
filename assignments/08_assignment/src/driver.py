"""
XRT Python driver for Assignment 08.

Usage (from the assignment directory, after building xclbins):
    python3 src/driver.py

Requires: pyxrt, numpy, torch
"""

import time

import numpy as np
import torch
import pyxrt

# ── Problem size (M x K @ K x N) ──────────────────────────────────────────────
M, K, N = 16, 64, 16
FLOPS = 2 * M * N * K          # multiply-add = 2 FLOPs
NPU_CLOCK_HZ = 1.8e9           # XDNA2 compute-tile clock


def verify(in0: torch.Tensor, in1: torch.Tensor, out: torch.Tensor) -> None:
    """
    Verify the NPU output against a CPU reference and report the error.

    Computation: out += in0 @ in1  (out scratchpad is zero-initialized).

    Parameters
    ----------
    in0, in1 : bfloat16 torch tensors
    out : bfloat16 torch tensor
    """

    expected = in0.float() @ in1.float()
    diff = (out.float() - expected).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"[error] max abs error  {max_abs:.4f}")
    print(f"[error] mean abs error {mean_abs:.4f}")

    assert torch.allclose(out.float(), expected, atol=0.5, rtol=0.2), (
        f"[FAIL] matmul mismatch! max abs error: {max_abs}"
    )


def benchmark(kernel, bo_instr, insts_nbytes, bo_in0, bo_in1, bo_out,
              iters: int = 2000) -> None:
    """
    Measure end-to-end kernel throughput (includes host launch + DMA overhead).

    For the *compute* performance of the kernel itself use the analytical
    cycle count: GFLOPS = FLOPS * NPU_CLOCK_HZ / cycles  (see Task 6).
    """
    # warmup
    for _ in range(50):
        kernel(3, bo_instr, insts_nbytes, bo_in0, bo_in1, bo_out).wait()

    t0 = time.perf_counter()
    for _ in range(iters):
        kernel(3, bo_instr, insts_nbytes, bo_in0, bo_in1, bo_out).wait()
    elapsed = time.perf_counter() - t0

    per_call_us = elapsed / iters * 1e6
    gflops = iters * FLOPS / elapsed / 1e9
    print(f"[bench] {iters} runs: {per_call_us:.2f} us/call, "
          f"{gflops:.2f} GFLOPS (end-to-end, incl. host/DMA overhead)")


def run() -> None:
    xclbin_path = "build/final_matmul.xclbin"
    insts_path = "build/insts_matmul.bin"

    insts = np.fromfile(insts_path, dtype=np.uint32)

    device = pyxrt.device(0)
    xclbin = pyxrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)
    uuid = xclbin.get_uuid()
    context = pyxrt.hw_context(device, uuid)
    kname = xclbin.get_kernels()[0].get_name()
    kernel = pyxrt.kernel(context, kname)

    bo_instr = pyxrt.bo(device, insts.nbytes, pyxrt.bo.cacheable, kernel.group_id(1))
    bo_instr.write(insts.tobytes(), 0)
    bo_instr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, insts.nbytes, 0)

    torch.manual_seed(42)
    data_in0 = torch.randn(16, 64, dtype=torch.bfloat16)
    data_in1 = torch.randn(64, 16, dtype=torch.bfloat16)
    data_out = torch.zeros(16, 16, dtype=torch.bfloat16)

    # Create buffer objects with corresponding size
    bo_in0 = pyxrt.bo(device, data_in0.nbytes, pyxrt.bo.host_only, 0)
    bo_in1 = pyxrt.bo(device, data_in1.nbytes, pyxrt.bo.host_only, 0)
    bo_out = pyxrt.bo(device, data_out.nbytes, pyxrt.bo.host_only, 0)

    # Copy data to buffer objects
    bo_in0.write(data_in0.view(torch.int16).numpy().tobytes(), 0)
    bo_in1.write(data_in1.view(torch.int16).numpy().tobytes(), 0)
    bo_out.write(data_out.view(torch.int16).numpy().tobytes(), 0)

    # View buffer objects as torch tensor
    tensor_in0 = torch.frombuffer(
        bo_in0.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in0.shape)
    ).view(data_in0.shape)
    tensor_in1 = torch.frombuffer(
        bo_in1.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_in1.shape)
    ).view(data_in1.shape)
    tensor_out = torch.frombuffer(
        bo_out.map(),
        dtype=torch.bfloat16,
        count=np.prod(data_out.shape)
    ).view(data_out.shape)
    assert torch.equal(data_in0, tensor_in0)
    assert torch.equal(data_in1, tensor_in1)
    assert torch.equal(data_out, tensor_out)

    # Sync buffer objects: to device
    bo_in0.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in0.nbytes, 0)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_in1.nbytes, 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, data_out.nbytes, 0)

    h = kernel(3, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)
    h.wait()

    # Sync output buffer object: from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, data_out.nbytes, 0)

    verify(tensor_in0, tensor_in1, tensor_out)

    print("[PASS] matmul verification passed.")

    benchmark(kernel, bo_instr, insts.nbytes, bo_in0, bo_in1, bo_out)


if __name__ == "__main__":
    run()
