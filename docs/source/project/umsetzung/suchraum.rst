Suchraum und Enumerator
^^^^^^^^^^^^^^^^^^^^^^^

Für jeden der 6 Knöpfe gibt es eine Reihe hardware-sinnvoller Werte:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — die sechs Knöpfe
   :start-at: # die Knoepfe (aus dem Pitch)
   :end-at: VARIANT_CHOICES

Macht `3 · 3 · 3 · 3 · 3 · 2 = 486` Kombinationen — mehr als die 81 aus dem
ursprünglichen Pitch. Die 81 zählten nur die Tile-Kombinatorik und ließen zwei
Achsen weg: das asymmetrische `M_L2 ≠ N_L2` und die zweite Ausführungsvariante.
Beide gehören zum Freiheitsgrad, also zählen sie mit.

`enumerate_candidates` iteriert über das Kreuzprodukt und baut für jede
Kombination via `build_one_config` einen `Candidate`. Kombinationen, die
`Optimizer.verify()` nicht bestehen, fallen mit einem `except` raus — noch ohne
jedes Pruning:

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — enumerate_candidates(), ungültige Configs raus
   :start-at: try:
   :end-at: skipped += 1
   :dedent:

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

.. literalinclude:: ../../project/src/autotuner/search.py
   :language: python
   :caption: search.py — krumme Größen hochrunden
   :start-at: # split_dim will exakte Teilbarkeit
   :end-at: padded_k = k_outer
   :dedent:

Die `dim_sizes` der Config sind damit die gepaddeten Größen — die TFLOPS rechnen
wir aber konsequent auf der Original-Shape, sonst würde man sich die Padding-Arbeit
als Leistung schönrechnen.
