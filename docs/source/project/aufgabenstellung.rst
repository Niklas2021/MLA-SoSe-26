Aufgabenstellung
----------------

Ausgangslage: warum ein Tuner? 
^^^^^^^^^^^^
Wie findet man für eine Tensor-Kontraktion automatisch eine schnelle cuTile-Config?
Darum ging es in unserem Projekt und dieser Post erklärt, welcher Spielraum
beim Ausführen einer Kontraktion überhaupt existiert, wie daraus ein Suchraum wird
und wie der Tuner ihn durcharbeitet.

In Assignment 05 und 06 haben wir zwei Kontraktionen von Hand getunt: den batched
Matmul `cmk,ckn->cmn` und die Tensor-Ring-Kontraktion `acspx,bspy->abcyx`. Der Kernel-Code war dabei nie das Problem. Die Rechenvorschrift ist in beiden
Fällen trivial — 2 fp16-Tensoren rein, über die gemeinsame Achse summieren,
fp32 akkumulieren. Das Performance steckt woanders: in der Config, also darin, wie
man die Arbeit in Kacheln zerlegt und in welcher Reihenfolge man die Kacheln
abarbeitet.

Diese Herleitung zweimal per Hand zu machen hat gereicht, um zu sehen, dass sie
nicht skaliert. Die optimale Config hängt an der Shape und an der GPU — Cache-Größe,
Registerbudget, Zahl der SMs — und müsste für jede neue Kombination neu gefunden
werden.

Zielsetzung
^^^^^^^^^^^

Genau das automatisiert der Tuner: er bekommt einen Einsum-String und
Shapes und liefert eine gute cuTile-Config, per Messung bestätigt.

Abgrenzung und Rahmenbedingungen
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Was er nicht ist: ein allgemeiner Tensor-Compiler. Er deckt zwei Struktur-Familien
ab (GEMM-artig wie A05, Ring wie A06) und tunt die Configs, nicht den Kernel-Code.