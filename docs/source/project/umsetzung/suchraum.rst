Suchraum und Enumerator
^^^^^^^^^^^^^^^^^^^^^^^

Für jeden der 6 Knöpfe gibt es eine Reihe hardware-sinnvoller Werte:

.. code-block:: python
   :caption: project/src/autotuner/search.py:16-22

   # die Knoepfe (aus dem Pitch)
   M_PRIM_CHOICES = [64, 128, 256]
   N_PRIM_CHOICES = [64, 128, 256]
   K_PRIM_CHOICES = [32, 64, 128]
   M_L2_CHOICES = [2, 4, 8]
   N_L2_CHOICES = [2, 4, 8]
   VARIANT_CHOICES = ["A", "B"]   # A = m_l2/n_l2 als PAR (swizzle), B = als SEQ-Loops

Macht `3 · 3 · 3 · 3 · 3 · 2 = 486` Kombinationen — mehr als die 81 aus dem
ursprünglichen Pitch. Die 81 zählten nur die Tile-Kombinatorik und ließen zwei
Achsen weg: das asymmetrische `M_L2 ≠ N_L2` und die zweite Ausführungsvariante.
Beide gehören zum Freiheitsgrad, also zählen sie mit.

`enumerate_candidates` iteriert über das Kreuzprodukt und baut für jede
Kombination via `build_one_config` einen `Candidate`. Kombinationen, die
`Optimizer.verify()` nicht bestehen, fallen mit einem `except` raus — noch ohne
jedes Pruning:

.. code-block:: python
   :caption: project/src/autotuner/search.py:150-155

                               try:
                                   candidates.append(build_one_config(
                                       einsum_props, variant,
                                       m_prim, n_prim, k_prim, m_l2, n_l2))
                               except (ValueError, NotImplementedError):
                                   skipped += 1

Die Hand-Config als Sanity-Check
"""""""""""""""""""""""""""""""""

Ein Suchraum taugt nur, wenn die gesuchte Lösung überhaupt darin liegt. Wir kennen
eine gute Lösung: die handoptimierte A05-Config (128/128/64, 8×8, Variante A). Der
`__main__`-Block in `search.py` prüft deshalb, dass genau diese Config im
enumerierten Set auftaucht und auch das Pruning übersteht. Würde der Tuner sie
unterwegs verlieren, könnte er sie auch nie finden.

Krumme Shapes
"""""""""""""

`split_dim` verlangt exakte Teilbarkeit (`outer · inner` muss die alte Größe
ergeben), sonst wirft es. Eine Shape wie `M = 1500` geht also nicht direkt. Statt
solche Fälle abzulehnen, padden wir auf die nächste teilbare Größe hoch, der
Überhang wird später im Kernel über `PaddingMode.ZERO` genullt:

.. code-block:: python
   :caption: project/src/autotuner/search.py:76-84

       # split_dim will exakte Teilbarkeit, also runden wir krumme Groessen hoch.
       # dim_sizes sind damit gepaddet, der Ueberhang wird im Kernel genullt.
       m_l2_outer = ceildiv(einsum_props.orig_m, m_prim * m_l2)
       n_l2_outer = ceildiv(einsum_props.orig_n, n_prim * n_l2)
       k_outer = ceildiv(einsum_props.orig_k, k_prim)

       padded_m = m_l2_outer * m_l2 * m_prim
       padded_n = n_l2_outer * n_l2 * n_prim
       padded_k = k_outer * k_prim

Die `dim_sizes` der Config sind damit die gepaddeten Größen — die TFLOPS rechnen
wir aber konsequent auf der Original-Shape, sonst würde man sich die Padding-Arbeit
als Leistung schönrechnen.
