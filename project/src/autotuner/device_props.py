# GPU-Properties die wir fuers Pruning/Ranking brauchen.
from dataclasses import dataclass


@dataclass
class DeviceProperties:
    gpu_name: str
    l2_cache: int                  # Bytes
    smem_per_block: int            # Opt-in Limit, noch ohne reserved
    smem_per_sm: int
    number_sm: int
    regs_per_block: int            # 32-bit Register pro Block
    reserved_smem_per_block: int
    mem_clock_khz: int
    mem_bus_bits: int

    def usable_smem_per_block(self):
        return self.smem_per_block - self.reserved_smem_per_block

    def peak_dram_bandwidth(self):
        # MemoryClockRate ist bei der GB10 schon die effektive Datenrate -> kein x2
        return self.mem_clock_khz * 1e3 * (self.mem_bus_bits / 8)


def get_device_properties():
    # cupy/torch erst hier importieren, damit die Dataclass auch ohne GPU geht
    import cupy as cp
    import torch
    attr = dict(cp.cuda.Device().attributes.items())
    return DeviceProperties(
        gpu_name=torch.cuda.get_device_properties(0).name,
        l2_cache=attr["L2CacheSize"],
        smem_per_block=attr["MaxSharedMemoryPerBlockOptin"],
        smem_per_sm=attr["MaxSharedMemoryPerMultiprocessor"],
        number_sm=attr["MultiProcessorCount"],
        regs_per_block=attr["MaxRegistersPerBlock"],
        reserved_smem_per_block=attr["ReservedSharedMemoryPerBlock"],
        mem_clock_khz=attr["MemoryClockRate"],
        mem_bus_bits=attr["GlobalMemoryBusWidth"],
    )


# GB10-Werte fest hinterlegt (aus smoke_test.log), fuers Testen ohne GPU
GB10 = DeviceProperties(
    gpu_name="NVIDIA GB10",
    l2_cache=25165824,
    smem_per_block=101376,
    smem_per_sm=102400,
    number_sm=48,
    regs_per_block=65536,
    reserved_smem_per_block=1024,
    mem_clock_khz=8533000,
    mem_bus_bits=256,
)
