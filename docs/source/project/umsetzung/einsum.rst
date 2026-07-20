Einsum-Parsing und Klassifikation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bevor der Tuner irgendetwas kacheln kann, muss er wissen, welche Rolle jede
Dimension spielt. Der Einsum-String allein sagt das nicht direkt — er nennt nur
Buchstaben. Aus welchen Tensoren ein Buchstabe kommt, verrät aber seine Rolle,
und genau das macht ``parse_einsum``.

Klassifikation der Dimensionen
""""""""""""""""""""""""""""""

Jeder Buchstabe steht in einer Teilmenge der drei Operanden (Input A, Input B,
Output). Aus dieser Zugehörigkeit folgt die Rolle eindeutig:

===================  ==============  ================================
in A / B / Out       Rolle           Bedeutung
===================  ==============  ================================
in allen dreien      **Batch (C)**   unabhängige Wiederholung
in A und B           **K**           wird wegsummiert (nicht im Output)
in A und Output      **M**           Zeilenachse des Ergebnisses
sonst (B und Out)    **N**           Spaltenachse des Ergebnisses
===================  ==============  ================================

Am A05-Beispiel ``cmk, ckn -> cmn``: ``c`` steht in beiden Inputs und im Output,
ist also Batch; ``k`` steht in A und B, aber nicht im Output, ist also die
Reduktionsachse; ``m`` steht in A und Output (M), ``n`` in B und Output (N).

Bei A06 ``acspx, bspy -> abcyx`` fällt das Ergebnis weniger offensichtlich aus:
``a`` und ``c`` stehen nur in A und im Output (beide M), ``x`` ebenso (M),
``b`` steht nur in B und im Output (N), ``y`` auch (N), und ``s`` und ``p``
stehen in beiden Inputs, aber nicht im Output (beide K). Es gibt hier also
mehrere M-, N- und K-Achsen und keinen geteilten Batch.

.. literalinclude:: ../../project/src/autotuner/einsum_parser.py
   :language: python
   :caption: einsum_parser.py — Rolle je Dimension
   :start-at: set_a, set_b, set_out = set(in_a)
   :end-at: n_chars.append(d)
   :dedent:

Wahl der prim-Dimensionen
"""""""""""""""""""""""""

Der Kernel rechnet eine einzelne ``M_PRIM × N_PRIM``-Kachel über eine
``K_PRIM``-Reduktion. Er braucht also aus jeder Sorte genau eine Achse, die in
die ``mma``-Kachel eingeht — die *prim*-Achse. Als prim wird jeweils die
**innerste** Achse ihrer Sorte im Tensor gewählt, weil die im Speicher mit
Stride 1 liegt und sich ohne Umkopieren laden lässt:

.. literalinclude:: ../../project/src/autotuner/einsum_parser.py
   :language: python
   :caption: einsum_parser.py — innerste Achse einer Sorte
   :pyobject: _innermost

Damit ist für A05 ``m/n/k`` schlicht die jeweils einzige Achse. Für A06 ergibt
sich ``x`` als prim-M, ``y`` als prim-N und ``p`` als prim-K — jeweils der
letzte Buchstabe ihrer Sorte im String.

Eine Bedingung muss die prim-K-Wahl erfüllen: ``p`` muss in **beiden** Inputs
die innerste K-Achse sein. Wäre sie es nur in A, dann läge sie in B nicht mit
Stride 1, und der ``mma``-Load müsste erst transponieren oder fusionieren — das
kann der Kernel nicht. Solche Fälle lehnt der Parser ab, statt still das falsche
Layout zu laden:

.. literalinclude:: ../../project/src/autotuner/einsum_parser.py
   :language: python
   :caption: einsum_parser.py — Guard für prim-K
   :start-at: m_char = _innermost(in_a, set(m_chars))
   :end-at: Fusion/Transpose noetig")
   :dedent:

Alle Achsen, die nicht prim werden, bleiben übrig und bekommen im Mehrdim-Fall
eine eigene Rolle. ``is_multi`` unterscheidet die beiden Familien: A05 hat je
eine M/N/K-Achse und ist nicht multi, A06 hat Zusatzachsen und ist es.

.. literalinclude:: ../../project/src/autotuner/einsum_parser.py
   :language: python
   :caption: einsum_parser.py — Restachsen aufteilen
   :start-at: extra_m_chars = [c for c in m_chars
   :end-at: seq_k_chars = [c for c in k_chars if c != k_char]
   :dedent:

Die Zusatz-M- und -N-Achsen (bei A06 ``a``, ``c`` auf der A-Seite und ``b`` auf
der B-Seite) werden zu unabhängigen Batches, die über die Block-ID parallel
laufen. Eine zusätzliche K-Achse (``s``) wird zu einer zweiten sequenziellen
Reduktionsschleife im Kernel, geschachtelt um die prim-K-Schleife. Der
Ring-Kernel im :ref:`Kernel-Kapitel <kernel>` setzt genau das um.

Einordnung
""""""""""

Der Parser klassifiziert allgemeiner, als die beiden Kernel am Ende rechnen
können. Ein String wie ``cmk, cnk -> cmn`` (B in NT-Layout) wird sauber
klassifiziert, obwohl der Kernel das resultierende Layout nicht korrekt laden
kann. Diese Lücke blieb lange offen und wurde erst später mit einem Layout-Guard
geschlossen — das steht in
:ref:`Einsum-Abdeckung <einsum-abdeckung>`.
