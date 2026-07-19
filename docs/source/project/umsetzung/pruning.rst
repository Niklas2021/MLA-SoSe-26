Statisches Pruning
^^^^^^^^^^^^^^^^^^

Bevor ein einziger Kernel kompiliert wird, wirft `prune` alles raus, was sich als nicht sinnvoll erkennen lässt. Es gibt 4 Filter:

.. code-block:: python
   :caption: project/src/autotuner/search.py:184-193

   def prune_reason(cand, dev, buffer_stages, reg_fraction, max_padding, smem_limit):
       if cand.m_prim % MMA_ALIGN or cand.n_prim % MMA_ALIGN or cand.k_prim % MMA_ALIGN:
           return "mma_align"
       if estimate_smem_bytes(cand, buffer_stages) > smem_limit:
           return "smem_exceeded"
       if estimate_acc_registers(cand) > dev.regs_per_block * reg_fraction:
           return "acc_registers"
       if padding_ratio(cand) > max_padding:
           return "padding_waste"
       return None

1. **MMA-Teilbarkeit:** die Prim-Größen müssen Vielfache von 16 sein, sonst passt
   die fp16-Tensor-Core-Kachel nicht. Im Standardraum erfüllen das alle Werte; der
   Filter fängt nur handgestrickte Suchräume ab.
2. **SMEM-Budget:** die beiden fp16-Operand-Tiles mal Double-Buffering, müssen ins
   nutzbare Shared Memory passen — auf der GB10 rund 100 KB. Das ist der Filter,
   der tatsächlich aussortiert.

   .. code-block:: python
      :caption: project/src/autotuner/search.py:167-171

      def estimate_smem_bytes(cand, buffer_stages):
          # die beiden fp16-Operand-Tiles mal Stages. Akku liegt in Registern.
          a_tile = cand.m_prim * cand.k_prim
          b_tile = cand.k_prim * cand.n_prim
          return (a_tile + b_tile) * 2 * buffer_stages

3. **Akku-Register:** der Akkumulator braucht `M_PRIM · N_PRIM` fp32-Werte in
   Registern. Mehr als die halbe Registerdatei (`65536 · 0.5 = 32768`) lassen wir
   nicht zu — das trifft vor allem die `256×256`-Kacheln.
4. **Padding:** wächst das gepaddete Volumen auf mehr als das Achtfache des
   Originals, fliegt der Kandidat raus.

Für die A05-Referenz (`4096³`) ergibt das 486 → 342: 126 Kandidaten fallen wegen
SMEM, 18 wegen der Akku-Register. MMA- und Padding-Filter greifen hier gar nicht,
weil 4096 glatt durch alle Knöpfe teilbar ist.

Interessanter ist, was das Pruning nicht kann. Alle vier Filter hängen
ausschließlich an den Prim-Größen — weder `m_l2/n_l2` noch die Variante tauchen in
ihnen auf. Die 2 Achsen, die den L2-Reuse steuern, kann statisches Pruning also
gar nicht anfassen. Man könnte hoffen, sie über die übliche Cache-Regel
einzuschränken („Gruppen-Working-Set muss ins L2 passen"), aber auf der GB10 mit
ihren 25 MB L2 passt selbst die größte überlebende Gruppe locker hinein. Die
Entscheidung über Gruppengröße und Variante bleibt damit komplett der Messung
überlassen,nicht weil unser Filter zu schwach wäre, sondern weil die Hardware an
dieser Stelle nichts verbietet.

Das Pruning ist eine Heuristik, kein Beweis: `buffer_stages`, `smem_limit`,
`reg_fraction` und `max_padding` sind Parameter mit optimistischen Defaults. Fällt
eine Config fälschlich durch, fängt sie das `try/except` um das Kompilieren im
Mess-Harness — sie scheitert dort sauber, statt still falsch zu rechnen.
