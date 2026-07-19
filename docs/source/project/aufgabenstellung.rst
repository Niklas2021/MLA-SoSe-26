Aufgabenstellung
----------------

.. Diese Seite ist das Gegenstueck zu den "Aufgabenstellung"-Abschnitten der
   Assignments: WAS wollten wir bauen und WARUM -- noch keine Loesung.

Ausgangslage
^^^^^^^^^^^^

.. Inhalt:
   - In A05 haben wir die L2-optimale Aufteilung von "cmk,ckn->cmn" von Hand
     hergeleitet, in A06 dieselbe Handarbeit fuer die Ring-Kontraktion
     "acspx,bspy->abcyx". Verweis auf die beiden Assignment-Seiten setzen.
   - Der Kern der Handarbeit: m_prim/n_prim/k_prim und die L2-Gruppe m_l2/n_l2
     begruenden. Das ist eine endliche Menge von Entscheidungen -- also
     automatisierbar.
   - Warum das nicht trivial ist: die Entscheidung haengt an Shape UND Hardware,
     und die Begruendungen aus der Vorlesung (L2-Reuse, Arithmetic Intensity)
     liefern nur eine Vorauswahl, keine Rangfolge.

Zielsetzung
^^^^^^^^^^^

.. Inhalt:
   - Ein Satz Kernziel: aus (Einsum-String, Shapes) automatisch eine gute
     Tiling-Config finden, ohne dass ein Mensch tunt.
   - Die Pipeline als Vorschau: Basic-Config erzeugen -> Suchraum aufspannen ->
     billig vorfiltern -> Kernels instanziieren -> messen -> beste Config ausgeben.
     Hier fig_pipeline.png einbinden (aus project/praesentation/figures/).
   - Abgrenzung, was der Tuner NICHT tut: er schreibt keine Kernel. Er sucht
     Configs innerhalb einer Kernel-Familie. Das ist derselbe Ansatz wie bei
     cuBLAS/CUTLASS (endliche Template-Menge) und Triton (@autotune pro @jit).

Erfolgskriterien
^^^^^^^^^^^^^^^^

.. Inhalt:
   - Die urspruenglich gesetzten Ziele als Tabelle: A05-Handloesung zu >= 95 %
     reproduzieren, A06-Transfer zu >= 90 %, Korrektheit gegen torch.einsum.
   - Ehrlich dazusagen, dass sich die interessante Frage im Verlauf verschoben hat:
     von "erreichen wir die Handloesung?" zu "was ist ueberhaupt eine faire
     Vergleichsbasis?" -- das wird in Erweiterungen/Baseline und in der Evaluation
     aufgeloest. Hier nur ankuendigen.

Abgrenzung und Rahmenbedingungen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Inhalt:
   - Scope: zwei Inputs, fp16 mit fp32-Akku, Row-Major.
   - Zwei Kernel-Familien (GEMM-artig und Ring), kein allgemeiner Tensor-Compiler.
   - Hardware: primaer DGX Spark (GB10), zweite Karte RTX 3070 fuer die
     Portabilitaetsfrage. Detaillierte Specs stehen in der Evaluation.
   - Werkzeuge: cuTile, triton.testing.do_bench, torch.einsum als Referenz.
