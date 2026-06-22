from .config import Config, DimType, ExecType, PrimType, LastType, FirstType, DataType


def generate_config(einsum_str, input_shapes):
    # Einsum parsen, z.B. "cmk, ckn -> cmn"
    lhs, rhs = einsum_str.replace(" ", "").split("->")
    input_strs = lhs.split(",")
    output_str = rhs

    # alle einzigartigen Dims in Reihenfolge des ersten Auftretens sammeln
    all_dims = []
    for s in input_strs + [output_str]:
        for c in s:
            if c not in all_dims:
                all_dims.append(c)

    in_a = set(input_strs[0])
    in_b = set(input_strs[1])
    in_out = set(output_str)

    # Dimensionstyp bestimmen
    dim_types = []
    for d in all_dims:
        if d in in_a and d in in_b and d in in_out:
            dim_types.append(DimType.C)
        elif d in in_a and d in in_b and d not in in_out:
            dim_types.append(DimType.K)
        elif d in in_a and d not in in_b and d in in_out:
            dim_types.append(DimType.M)
        else:
            dim_types.append(DimType.N)

    # Größen der Dims aus den input_shapes lesen
    dim_to_size = {}
    for tensor_str, shape in zip(input_strs, input_shapes):
        for char, size in zip(tensor_str, shape):
            dim_to_size[char] = size

    dim_sizes = [dim_to_size[d] for d in all_dims]

    # Row-major Strides pro Tensor berechnen (0 wenn Dim nicht vorkommt)
    all_tensor_strs = input_strs + [output_str]
    strides = []
    for tensor_str in all_tensor_strs:
        # Stride von rechts aufbauen
        tensor_dims = list(tensor_str)
        tensor_sizes = [dim_to_size[d] for d in tensor_dims]
        tensor_strides = [0] * len(tensor_dims)
        s = 1
        for i in reversed(range(len(tensor_dims))):
            tensor_strides[i] = s
            s *= tensor_sizes[i]

        # auf alle_dims mappen, fehlende Dims kriegen Stride 0
        stride_row = []
        for d in all_dims:
            if d in tensor_str:
                idx = tensor_str.index(d)
                stride_row.append(tensor_strides[idx])
            else:
                stride_row.append(0)
        strides.append(stride_row)

    # alles auf SEQ setzen
    exec_types = [ExecType.SEQ] * len(all_dims)

    return Config(
        data_type=DataType.FLOAT16,
        prim_main=PrimType.GEMM,
        prim_last=LastType.NONE,
        prim_first=FirstType.ZERO,
        dim_types=dim_types,
        exec_types=exec_types,
        dim_sizes=dim_sizes,
        strides=strides,
    )
