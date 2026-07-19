Kernel-Instanziierung
^^^^^^^^^^^^^^^^^^^^^

.. Quelle: project/src/autotuner/kernels.py

Ein generischer Kernel statt Codegen
""""""""""""""""""""""""""""""""""""

.. Inhalt:
   - Der Pitch sprach von "Codegen aus Templates". Warum wir das nicht gemacht
     haben: String-Templates + exec() sind fragil, und die A05-Kernel nehmen ihre
     Tile-Groessen ohnehin als ct.Constant -- der JIT spezialisiert pro Wert
     (wie Triton mit constexpr).
   - Dass die Spezialisierung wirklich pro Wert passiert, wurde als Allererstes
     verifiziert (measure_compile.py), weil der ganze Ansatz daran haengt.
     Ergebnis: ja, ~0.4 s pro Compile.

Variante A: Swizzle über die Block-ID
"""""""""""""""""""""""""""""""""""""

.. Inhalt:
   - literalinclude matmul_variant_a. Den pid-Decode Zeile fuer Zeile erklaeren:
     aus einer linearen bid werden n_l2, m_l2, n_l2_outer, m_l2_outer, c.
   - Warum diese Reihenfolge L2-Reuse erzeugt. code_variant_a.png passt hier.

Variante B: L2-Gruppe als SEQ-Loops
"""""""""""""""""""""""""""""""""""

.. Inhalt:
   - literalinclude matmul_variant_b, Unterschied zum Swizzle: weniger CTAs,
     dafuer Schleifen im Block.

Zweite Familie: der Ring-Kernel
"""""""""""""""""""""""""""""""

.. Inhalt:
   - Warum A06 einen eigenen Kernel braucht und kein Umbau reicht: A05 hat einen
     GETEILTEN Batch (c indiziert A, B und C), A06 UNABHAENGIGE Batches (a,c nur
     in A, b nur in B). Der M2-Kernel wuerde alle a-x-b-Kombinationen rechnen
     statt der Diagonale. Das ist der wichtigste konzeptionelle Punkt des Kapitels.
   - literalinclude matmul_ring_a. Der Per-Tile-permute fuer das nicht-mma-fertige
     Layout, und die aeussere SEQ-Schleife ueber s.
   - code_ring_a_v2.png einbinden.
   - Einordnung: eine neue Topologie braucht ein neues Template, beliebige Shapes
     innerhalb einer Familie nicht. Genauso arbeiten cuBLAS/CUTLASS und Triton.

Dispatcher
""""""""""

.. Inhalt:
   - run_candidate als einzige Stelle, die aus einem Candidate einen Kernel-Start
     macht. Kurzer Codeausschnitt (Stand nach den Erweiterungen -- oder hier den
     einfachen Stand zeigen und im Erweiterungs-Kapitel den finalen).
