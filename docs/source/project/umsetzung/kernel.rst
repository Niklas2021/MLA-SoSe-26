Kernel-Instanziierung
^^^^^^^^^^^^^^^^^^^^^

Ab hier braucht es die GPU. Es gibt einen einzigen generischen cuTile-Kernel pro Variante, dessen Tile-Größen als `ct.Constant[int]` deklariert sind.

Der Kern von Variante A: die Tile-Größen kommen als `ct.Constant`, und der
L2-Gruppen-Swizzle wird aus der Block-ID dekodiert:

.. code-block:: python
   :caption: project/src/autotuner/kernels.py:13-35

   @ct.kernel
   def matmul_variant_a(A, B, C,
                        M_PRIM: ct.Constant[int],
                        N_PRIM: ct.Constant[int],
                        K_PRIM: ct.Constant[int],
                        M_L2: ct.Constant[int],
                        N_L2: ct.Constant[int],
                        num_m_l2_outer: ct.Constant[int],
                        num_n_l2_outer: ct.Constant[int],
                        num_k_outer: ct.Constant[int]):
       pid = ct.bid(0)
       n_l2_idx = pid % N_L2
       pid = pid // N_L2
       m_l2_idx = pid % M_L2
       pid = pid // M_L2
       n_l2_outer_idx = pid % num_n_l2_outer
       pid = pid // num_n_l2_outer
       m_l2_outer_idx = pid % num_m_l2_outer
       pid = pid // num_m_l2_outer
       c_idx = pid
       m_block = m_l2_outer_idx * M_L2 + m_l2_idx
       n_block = n_l2_outer_idx * N_L2 + n_l2_idx

In der Dekodier-Reihenfolge steckt die L2-Gruppe von oben: `n_l2` und `m_l2` sind
die innersten Faktoren der Block-ID, benachbarte IDs laufen also durch dieselbe
Gruppe, bevor die Gruppe wechselt. Danach die eigentliche Rechnung — über die
K-Streifen laden, akkumulieren, am Ende einmal zurückschreiben:

.. code-block:: python
   :caption: project/src/autotuner/kernels.py:37-48

       acc = ct.zeros((M_PRIM, N_PRIM), dtype=ct.float32)
       for k_it in range(num_k_outer):
           a_tile = ct.load(A, index=(c_idx, m_block, k_it),
                            shape=(1, M_PRIM, K_PRIM), padding_mode=ct.PaddingMode.ZERO)
           b_tile = ct.load(B, index=(c_idx, k_it, n_block),
                            shape=(1, K_PRIM, N_PRIM), padding_mode=ct.PaddingMode.ZERO)
           a_tile = ct.reshape(a_tile, (M_PRIM, K_PRIM))
           b_tile = ct.reshape(b_tile, (K_PRIM, N_PRIM))
           acc = ct.mma(a_tile, b_tile, acc)
       out = ct.reshape(acc, (1, M_PRIM, N_PRIM)).astype(ct.float16)
       ct.store(C, index=(c_idx, m_block, n_block), tile=out)

Variante B (`matmul_variant_b`) ist derselbe Kernel mit `m_l2/n_l2` als zwei
SEQ-Loops im CTA statt als Swizzle; der Launcher startet entsprechend ein um
`m_l2 · n_l2` kleineres Grid. Jede kompilierte Config wird gegen `torch.einsum`
geprüft (`allclose`, rtol=1e-2, atol=1e-1), bevor ihre Zeit zählt.
