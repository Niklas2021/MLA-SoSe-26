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

Beide Kernels führen dieselbe Operation aus – elementweise Addition zweier 4D-Tensoren mit dem Unterschied, *welche* Dimensionen parallelisiert werden und welche innerhalb eines Blocks als Tile verarbeitet werden.

**Kernel 1 (sum_kl):** 

Jeder Block bekommt über ``ct.bid(0)`` und ``ct.bid(1)`` einen festen ``(M, N)``-Index zugeteilt und lädt dann ein vollständiges ``(1, 1, K, L)``-Tile aus beiden Eingabetensoren. 
Der Grid hat die Form ``(M, N, 1)``, also laufen insgesamt ``M × N`` Blöcke parallel. 
Innerhalb eines Blocks wird das gesamte ``(K, L)``-Slice auf einmal addiert und zurückgeschrieben.

**Kernel 2 (sum_mn):** 

Hier ist es umgekehrt: der Grid hat die Form ``(K, L, 1)``, jeder Block bekommt einen ``(K, L)``-Index und lädt ein ``(M, N, 1, 1)``-Tile. 

Pro Block wird also ein komplettes ``(M, N)``-Slice verarbeitet.

Teilaufgabe b)
-------------

``sum_kl`` ist ~3,4× schneller, obwohl beide Kernels gleich viele Blöcke starten (je 128*16 = 2048) und pro Block gleich viele Elemente anfassen. Der Grund liegt im Speicherlayout.
Der Tensor liegt row-major im Speicher, die letzte Dimension L ist die schnellste. 

Ein ``(1, 1, K, L)``-Tile liegt zusammenhängend – coalesced access, Cache wird gut ausgenutzt.
Ein ``(M, N, 1, 1)``-Tile greift dagegen auf Elemente zu, die jeweils K × L Einträge auseinander liegen. Das ist strided access. Das erklärt den Faktor 3,4