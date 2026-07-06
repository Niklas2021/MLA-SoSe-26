from dataclasses import dataclass
from enum import Enum

# Enums für die Konfiguration

class DimType(Enum):
    M = "m"
    N = "n"
    K = "k"
    C = "c"

class ExecType(Enum):
    SEQ = "seq"
    PAR = "par"
    PRIM = "prim"

class PrimType(Enum):
    GEMM = "gemm"
    BGEMM = "bgemm"

class LastType(Enum):
    NONE = "none"
    ELWISE_MUL = "elwise_mul"

class FirstType(Enum):
    ZERO = "zero"

class DataType(Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"


# Config Dataclass - hält alle Infos für eine Kontraktion
@dataclass
class Config:
    data_type: DataType
    prim_main: PrimType
    prim_last: LastType
    prim_first: FirstType

    # pro Dimension
    dim_types: list
    exec_types: list
    dim_sizes: list

    # pro Tensor, pro Dimension (0 = Dim existiert nicht in diesem Tensor)
    strides: list
