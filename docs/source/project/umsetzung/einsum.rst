Einsum-Parsing und Klassifikation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Inhalt: wie aus dem String die Rollen der Dimensionen werden.
   Quelle: project/src/autotuner/einsum_parser.py

Klassifikation der Dimensionen
""""""""""""""""""""""""""""""

.. Inhalt:
   - Die Regel als Tabelle: Dim in A+B+Out = Batch (C), in A+B = K, in A+Out = M,
     sonst N. Am A05-Beispiel "cmk,ckn->cmn" und am A06-Beispiel
     "acspx,bspy->abcyx" durchspielen.
   - literalinclude von parse_einsum mit :pyobject: parse_einsum

Wahl der prim-Dimensionen
"""""""""""""""""""""""""

.. Inhalt:
   - Warum die *innerste* Dim ihrer Sorte gewaehlt wird (stride 1 -> mma-tauglich).
     _innermost erklaeren.
   - Der Guard: prim-K muss in A UND B innerste K-Dim sein, sonst braeuchte der
     Load eine Fusion/Transposition. Codeausschnitt der Pruefung.
   - Der Mehrdim-Fall (A06): extra_m_chars, extra_n_chars, seq_k_chars und was
     daraus im Kernel wird (PAR-Batch bzw. SEQ-Loop). is_multi() als Weiche.

Einordnung
""""""""""

.. Inhalt:
   - Wichtige Ehrlichkeit: der Parser klassifiziert allgemeiner, als die Kernel
     rechnen koennen. Diese Luecke war lange offen und wurde erst spaeter
     geschlossen -- Vorwaertsverweis auf Erweiterungen/Abdeckung.
