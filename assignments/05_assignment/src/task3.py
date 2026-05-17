from task1 import DimType, ExecType


class Optimizer:
    def __init__(self, config):
        self.config = config

    def split_dim(self, dim_id, outer_size, inner_size):
        cfg = self.config
        if outer_size * inner_size != cfg.dim_sizes[dim_id]:
            raise ValueError(f"outer * inner muss gleich dim_sizes[{dim_id}] sein")

        old_type = cfg.dim_types[dim_id]
        old_exec = cfg.exec_types[dim_id]

        # Strides aufteilen: outer = alter_stride * inner_size, inner = alter_stride
        for t in range(len(cfg.strides)):
            s = cfg.strides[t][dim_id]
            cfg.strides[t][dim_id] = s * inner_size if s != 0 else 0
            cfg.strides[t].insert(dim_id + 1, s)

        cfg.dim_types[dim_id] = old_type
        cfg.dim_types.insert(dim_id + 1, old_type)
        cfg.exec_types[dim_id] = old_exec
        cfg.exec_types.insert(dim_id + 1, old_exec)
        cfg.dim_sizes[dim_id] = outer_size
        cfg.dim_sizes.insert(dim_id + 1, inner_size)

    def fuse_dims(self, dim_id_a, dim_id_b):
        cfg = self.config
        size_a = cfg.dim_sizes[dim_id_a]
        size_b = cfg.dim_sizes[dim_id_b]

        # Adjazenz und konsistente Reihenfolge über alle Tensoren prüfen
        # order=True -> a ist outer, order=False -> b ist outer
        order = None
        for t in range(len(cfg.strides)):
            sa = cfg.strides[t][dim_id_a]
            sb = cfg.strides[t][dim_id_b]
            if sa == 0 and sb == 0:
                continue
            if sa == 0 or sb == 0:
                raise ValueError(f"Tensor {t}: eine Dim fehlt, Fusion nicht möglich")
            if sa == sb * size_b:
                if order is False:
                    raise ValueError(f"Tensor {t}: inkonsistente Reihenfolge – a ist outer, vorher war b outer")
                order = True
            elif sb == sa * size_a:
                if order is True:
                    raise ValueError(f"Tensor {t}: inkonsistente Reihenfolge – b ist outer, vorher war a outer")
                order = False
            else:
                raise ValueError(f"Tensor {t}: Dims {dim_id_a} und {dim_id_b} sind nicht adjazent")

        # Strides zusammenführen: innerer Stride (der kleinere) bleibt
        for t in range(len(cfg.strides)):
            sa = cfg.strides[t][dim_id_a]
            sb = cfg.strides[t][dim_id_b]
            cfg.strides[t][dim_id_a] = min(sa, sb) if (sa != 0 and sb != 0) else 0
            cfg.strides[t].pop(dim_id_b)

        cfg.dim_sizes[dim_id_a] = size_a * size_b
        cfg.dim_sizes.pop(dim_id_b)
        # dim_type und exec_type von a behalten, b entfernen
        cfg.dim_types.pop(dim_id_b)
        cfg.exec_types.pop(dim_id_b)

    def permute_dims(self, permutation):
        cfg = self.config
        cfg.dim_types = [cfg.dim_types[i] for i in permutation]
        cfg.exec_types = [cfg.exec_types[i] for i in permutation]
        cfg.dim_sizes = [cfg.dim_sizes[i] for i in permutation]
        for t in range(len(cfg.strides)):
            cfg.strides[t] = [cfg.strides[t][i] for i in permutation]

    def make_executable(self):
        cfg = self.config
        n = len(cfg.dim_types)
        exec_types = [None] * n

        # letzte M, N, K jeweils als PRIM markieren
        for needed_type in [DimType.M, DimType.N, DimType.K]:
            for i in reversed(range(n)):
                if cfg.dim_types[i] == needed_type and exec_types[i] is None:
                    exec_types[i] = ExecType.PRIM
                    break

        # Rest: K -> SEQ, sonst PAR
        for i in range(n):
            if exec_types[i] is not None:
                continue
            if cfg.dim_types[i] == DimType.K:
                exec_types[i] = ExecType.SEQ
            else:
                exec_types[i] = ExecType.PAR

        cfg.exec_types = exec_types

        # Reihenfolge PAR | SEQ | PRIM
        par_ids = [i for i in range(n) if exec_types[i] == ExecType.PAR]
        seq_ids = [i for i in range(n) if exec_types[i] == ExecType.SEQ]
        prim_ids = [i for i in range(n) if exec_types[i] == ExecType.PRIM]
        self.permute_dims(par_ids + seq_ids + prim_ids)

        self.verify()

    def verify(self):
        cfg = self.config
        exec_types = cfg.exec_types
        dim_types = cfg.dim_types
        n = len(exec_types)

        # 1) kein K darf PAR sein
        for i in range(n):
            if dim_types[i] == DimType.K and exec_types[i] == ExecType.PAR:
                raise ValueError(f"Dim {i} ist K mit PAR - nicht erlaubt")

        # 2+3) Reihenfolge muss PAR -> SEQ -> PRIM sein
        order = {ExecType.PAR: 0, ExecType.SEQ: 1, ExecType.PRIM: 2}
        for i in range(n - 1):
            if order[exec_types[i]] > order[exec_types[i + 1]]:
                raise ValueError(f"falsche Reihenfolge bei Dim {i}: {exec_types[i]} vor {exec_types[i+1]}")

        # 4) PRIM muss mind. M, N, K enthalten
        prim_types = {dim_types[i] for i in range(n) if exec_types[i] == ExecType.PRIM}
        for needed in [DimType.M, DimType.N, DimType.K]:
            if needed not in prim_types:
                raise ValueError(f"PRIM Dims enthalten kein {needed}")
