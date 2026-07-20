.. _kernel:

Kernel-Instanziierung
^^^^^^^^^^^^^^^^^^^^^

Ab hier braucht es die GPU. Es gibt einen einzigen generischen cuTile-Kernel pro Variante, dessen Tile-Größen als `ct.Constant[int]` deklariert sind.

Der Kern von Variante A: die Tile-Größen kommen als `ct.Constant`, und der
L2-Gruppen-Swizzle wird aus der Block-ID dekodiert:

.. literalinclude:: ../../project/src/autotuner/kernels.py
   :language: python
   :caption: kernels.py — matmul_variant_a(), Block-ID-Decode
   :start-at: @ct.kernel
   :end-at: n_block = n_l2_outer_idx * N_L2 + n_l2_idx

In der Dekodier-Reihenfolge steckt die L2-Gruppe von oben: `n_l2` und `m_l2` sind
die innersten Faktoren der Block-ID, benachbarte IDs laufen also durch dieselbe
Gruppe, bevor die Gruppe wechselt. Danach die eigentliche Rechnung — über die
K-Streifen laden, akkumulieren, am Ende einmal zurückschreiben:

.. literalinclude:: ../../project/src/autotuner/kernels.py
   :language: python
   :caption: kernels.py — matmul_variant_a(), K-Schleife und Store
   :start-at: acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)
   :end-at: ct.store(C, index=(c_idx, m_block, n_block), tile=out)
   :dedent:

Variante B (`matmul_variant_b`) ist derselbe Kernel mit `m_l2/n_l2` als zwei
SEQ-Loops im CTA statt als Swizzle; der Launcher startet entsprechend ein um
`m_l2 · n_l2` kleineres Grid. Jede kompilierte Config wird gegen `torch.einsum`
geprüft (`allclose`, rtol=1e-2, atol=1e-1), bevor ihre Zeit zählt.
