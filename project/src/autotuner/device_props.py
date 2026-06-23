import cupy as cp
import torch
from dataclasses import dataclass


@dataclass
class DeviceProperties:
    # alle Speichergroessen sind in Bytes
    gpu_name: str            # zB "NVIDIA GB10" 
    l2_cache: int           
    smem_per_block: int      # nutzbares SMEM pro Block (Opt-in, Reserved bereits abgezogen)
    smem_per_sm: int       # shared memory pro SM geamt
    number_sm: int           # Anzahl SMs, für Grid size


def get_device_properties() -> DeviceProperties:
    attr = dict(cp.cuda.Device().attributes.items())
    return DeviceProperties(
        gpu_name = torch.cuda.get_device_properties(0).name,
        l2_cache = attr["L2CacheSize"],
        smem_per_block = attr["MaxSharedMemoryPerBlockOptin"],
        smem_per_sm = attr["MaxSharedMemoryPerMultiprocessor"],
        number_sm = attr["MultiProcessorCount"],
    )



