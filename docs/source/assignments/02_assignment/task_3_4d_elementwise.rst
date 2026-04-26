Task 3: 4D Tensor Elementwise Addition
=======================================

Aufgabenstellung
----------------

**a)** Zwei 4D-Tensoren ``A`` und ``B`` der Form ``(M, N, K, L)`` sollen
elementweise addiert werden – in zwei Kernel-Varianten:

1. Jedes Kernel-Programm berechnet ein 2D-Tile über ``K`` und ``L``,
   parallelisiert wird über ``M`` und ``N``.
2. Jedes Kernel-Programm berechnet ein 2D-Tile über ``M`` und ``N``,
   parallelisiert wird über ``K`` und ``L``.

**b)** Beide Kernels werden mit ``triton.testing.do_bench`` bei den
Dimensionen ``M=16, N=128, K=16, L=128`` gegeneinander gebenchmarkt.

Implementierte Funktionen
-------------------------

.. literalinclude:: ../../../../assignments/02_assignment/src/task3.py
   :language: python

Output
-------------
.. code-block:: text
    1: is correct:  True
    2: is correct:  True

    Benchmark results (average runtime):
    sum_kl  (tile over K,L  | grid over M,N): 0.1410 ms
    sum_mn  (tile over M,N  | grid over K,L): 0.4820 ms

    Analysis:
    sum_kl is faster by a factor of ~3.42x.

Teilaufgabe a)
-------------



Teilaufgabe b)
-------------