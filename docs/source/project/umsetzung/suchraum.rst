Suchraum und Enumerator
=======================

.. Quelle: project/src/autotuner/search.py

Die Knöpfe
----------

.. Inhalt:
   - Tabelle der sechs Knoepfe mit Wertemenge und Bedeutung:
     m_prim/n_prim {64,128,256}, k_prim {32,64,128}, m_l2/n_l2 {2,4,8},
     Variante {A,B}. Bei m_l2/n_l2 betonen: das ist die *zeitliche* Block-Gruppe
     (Gegenstueck zu group_size_m im Triton-Tutorial), keine raeumliche Kachel.
   - Variante A vs B erklaeren: m_l2/n_l2 als PAR (Swizzle ueber die bid) gegen
     SEQ-Loops im CTA. fig_exec_order.png passt hier.
   - Die Zahl 486 herleiten (3^5 * 2) und gegen die 81 aus dem Pitch stellen --
     die zaehlten nur die Tile-Kombinatorik ohne asymmetrisches m_l2!=n_l2 und
     ohne die zweite Exec-Variante.
   - literalinclude SearchSpace.

Aufbau einer Kandidaten-Config
------------------------------

.. Inhalt:
   - build_one_config Schritt fuer Schritt: generate_config fuer die Basic-Config,
     dann split_dim auf M (l2_outer, l2, prim), N und K, dann Exec-Types setzen.
   - Warum _split_tracked die Labels mitfuehrt: Indizes verschieben sich beim
     Split, ueber Namen bleibt es lesbar. Kurzer Codeausschnitt.
   - Variante A ueber make_executable (PAR|SEQ|PRIM), Variante B ueber eine feste
     permute_dims-Reihenfolge. Warum B den Mehrdim-Fall nicht abdeckt.
   - literalinclude build_one_config.

Padding krummer Shapes
----------------------

.. Inhalt:
   - Das Problem: split_dim verlangt outer*inner == alte Groesse, M=1234 geht also
     nicht direkt.
   - Die Loesung: auf ceildiv(M, prim*l2)*prim*l2 hochrunden, Ueberhang im Kernel
     ueber PaddingMode.ZERO nullen, TFLOPS aber auf der ORIGINAL-Shape rechnen.
   - Diese Stelle ist wichtiger, als sie aussieht -- die Gruppen-Ausdehnung
     m_l2*m_prim bestimmt, auf welches Vielfache gerundet wird. Vorwaertsverweis
     auf Erweiterungen/Baseline, wo genau das zur Kernfrage wird.
