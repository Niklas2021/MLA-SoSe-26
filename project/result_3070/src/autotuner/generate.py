from .config import Config, DimType, ExecType, PrimType, LastType, FirstType, DataType


def generate_config(einsum_props, size_of):
    # Kein Re-Parsing: alles kommt aus den schon geparsten einsum_props.
    # size_of wird explizit reingereicht (kann gepaddete Groessen enthalten).
    all_dims = einsum_props.all_dims

    # Dimensionstyp pro Dim, aligned mit all_dims (aus den Props, nicht neu klassifiziert)
    dim_types = []
    for d in all_dims:
        # Listen, nicht die prim-Chars -- sonst landen a06s extra-Dims (a,c,s,b) faelschlich in N
        if d in einsum_props.batch_chars:
            dim_types.append(DimType.C)
        elif d in einsum_props.k_chars:
            dim_types.append(DimType.K)
        elif d in einsum_props.m_chars:
            dim_types.append(DimType.M)
        else:
            dim_types.append(DimType.N)

    dim_sizes = [size_of[d] for d in all_dims]

    # Row-major Strides pro Tensor berechnen (0 wenn Dim nicht vorkommt)
    strides = []
    for tensor_str in (einsum_props.in_a, einsum_props.in_b, einsum_props.out):
        # Stride von rechts aufbauen
        tensor_dims = list(tensor_str)
        tensor_sizes = [size_of[d] for d in tensor_dims]
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
