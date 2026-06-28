from dataclasses import dataclass


@dataclass
class DeviceProperties:
    # alle Speichergroessen sind in Bytes
    gpu_name: str                  # zB "NVIDIA GB10"
    l2_cache: int
    smem_per_block: int            # Opt-in Limit pro Block (noch OHNE reserved abgezogen)
    smem_per_sm: int               # shared memory pro SM gesamt
    number_sm: int                 # Anzahl SMs, fuer Grid size
    regs_per_block: int            # max 32-bit Register pro Block (Akku-Check)
    reserved_smem_per_block: int   # reserviertes SMEM, muss vom Budget abgezogen werden
    mem_clock_khz: int             # Speichertakt (effektive Datenrate) in kHz
    mem_bus_bits: int              # Breite des Speicherbusses in Bit

    def usable_smem_per_block(self):
        return self.smem_per_block - self.reserved_smem_per_block

    def peak_dram_bandwidth(self):
        """Peak-DRAM-Bandbreite in Byte/s. MemoryClockRate ist bei der GB10
        schon die effektive Datenrate (8533 MT/s), daher kein zusaetzliches x2."""
        return self.mem_clock_khz * 1e3 * (self.mem_bus_bits / 8)


def get_device_properties() -> DeviceProperties:
    # cupy/torch erst hier importieren, damit die Dataclass auch ohne GPU
    # (z.B. lokal auf dem Mac fuer die search-Tests) importierbar bleibt.
    import cupy as cp
    import torch

    attr = dict(cp.cuda.Device().attributes.items())
    return DeviceProperties(
        gpu_name = torch.cuda.get_device_properties(0).name,
        l2_cache = attr["L2CacheSize"],
        smem_per_block = attr["MaxSharedMemoryPerBlockOptin"],
        smem_per_sm = attr["MaxSharedMemoryPerMultiprocessor"],
        number_sm = attr["MultiProcessorCount"],
        regs_per_block = attr["MaxRegistersPerBlock"],
        reserved_smem_per_block = attr["ReservedSharedMemoryPerBlock"],
        mem_clock_khz = attr["MemoryClockRate"],
        mem_bus_bits = attr["GlobalMemoryBusWidth"],
    )


# Die GB10-Werte fest hinterlegt (aus results/smoke_test.log + project_diary.md),
# damit wir Pruning/Ranking lokal ohne GPU testen koennen.
GB10 = DeviceProperties(
    gpu_name = "NVIDIA GB10",
    l2_cache = 25165824,            # 25.17 MB
    smem_per_block = 101376,        # MaxSharedMemoryPerBlockOptin
    smem_per_sm = 102400,
    number_sm = 48,
    regs_per_block = 65536,
    reserved_smem_per_block = 1024,
    mem_clock_khz = 8533000,        # 8533 MT/s
    mem_bus_bits = 256,
)
