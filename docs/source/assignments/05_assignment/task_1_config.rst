Task 1: Config Class
=====================

Aufgabenstellung
----------------

Definition der Python-Typen die zusammen eine Tensor-Kontraktions-Konfiguration
repräsentieren (siehe Folie 16ff der Vorlesung).

**a)** Sechs Enums:

- ``DimType``: ``M``, ``N``, ``K``, ``C``
- ``ExecType``: ``SEQ``, ``PAR``, ``PRIM``
- ``PrimType``: ``GEMM``, ``BGEMM``
- ``LastType``: ``NONE``, ``ELWISE_MUL``
- ``FirstType``: ``ZERO``
- ``DataType``: ``FLOAT16``, ``FLOAT32``

**b)** Eine ``Config`` Dataclass mit acht Feldern:

================  ==========================  ==========================================
Feld              Typ                         Bedeutung
================  ==========================  ==========================================
``data_type``     ``DataType``                Numerische Präzision
``prim_main``     ``PrimType``                (B)GEMM-Primitive im Kernel
``prim_last``     ``LastType``                Optionale elementweise Op nach Akku
``prim_first``    ``FirstType``               Initialisierung des Akkus
``dim_types``     ``list[DimType]``           Pro Dimension: Index-Typ
``exec_types``    ``list[ExecType]``          Pro Dimension: Execution-Strategie
``dim_sizes``     ``list[int]``               Pro Dimension: Größe
``strides``       ``list[list[int]]``         Pro Tensor, pro Dimension: Stride
================  ==========================  ==========================================

In ``strides`` bedeutet ein Eintrag von ``0``, dass die jeweilige Dimension in
dem Tensor nicht vorkommt.

Vollständige Implementierung
----------------------------

.. literalinclude:: ../../../../assignments/05_assignment/src/task1.py
   :language: python
