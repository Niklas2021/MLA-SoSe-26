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

*(folgt)*

Unsere Lösung
-------------

*(folgt)*
