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
    core_clock_khz: int = 0        # SM-Takt (ClockRate), fuers Compute-Peak
    # fp16-Tensor-FMAs pro SM pro Takt. Der EINZIGE Wert, der nicht aus den CUDA-
    # Attributen lesbar ist -> Architektur-Schaetzung. 512 passt zur GB10: 48 SM *
    # 2.418 GHz * 512 * 2 ~ 119 TFLOPS, und der gemessene Bestwert (75) liegt bei ~63 %.
    # Setzt nur den Roofline-Umschaltpunkt (memory vs compute), nicht die Reihenfolge
    # innerhalb eines Regimes.
    tensor_flop_per_sm_cycle: int = 512

    def usable_smem_per_block(self):
        return self.smem_per_block - self.reserved_smem_per_block

    def peak_dram_bandwidth(self):
        # MemoryClockRate ist bei der GB10 schon die effektive Datenrate -> kein x2
        return self.mem_clock_khz * 1e3 * (self.mem_bus_bits / 8)

    def peak_tensor_flops(self):
        # SMs * Takt * FMAs/SM/Takt * 2 (FMA = 2 FLOP)
        return self.number_sm * self.core_clock_khz * 1e3 * self.tensor_flop_per_sm_cycle * 2


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
        core_clock_khz=attr["ClockRate"],
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
    core_clock_khz=2418000,
)

# RTX 3070 (GA104, WSL). L2/SM aus smoke_test.log; mem_clock so gesetzt, dass die
# effektive GDDR6-Bandbreite ~448 GB/s rauskommt (256 bit * 14 Gbps). Zum Nachrechnen
# der Cross-GPU-Roofline-Regime (analyze_tune mit dev=RTX3070). Peak ~81 TFLOPS.
RTX3070 = DeviceProperties(
    gpu_name="NVIDIA GeForce RTX 3070",
    l2_cache=4194304,
    smem_per_block=101376,
    smem_per_sm=102400,
    number_sm=46,
    regs_per_block=65536,
    reserved_smem_per_block=1024,
    mem_clock_khz=14000000,
    mem_bus_bits=256,
    core_clock_khz=1725000,
)
