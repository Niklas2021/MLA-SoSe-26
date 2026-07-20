Projekt: cuTile Auto-Tuner für Tensor-Kontraktionen
====================================================

Der cuTile Auto-Tuner findet für eine Tensor-Kontraktion automatisch eine schnelle
Tiling-Config. Eingabe ist ein Einsum-String mit den Shapes der Operanden, Ausgabe
eine cuTile-Config, deren Güte per Messung bestätigt ist. Motiviert ist das Projekt
aus den Assignments A05 und A06, in denen wir den batched Matmul `cmk,ckn->cmn` und
die Tensor-Ring-Kontraktion `acspx,bspy->abcyx` von Hand getunt haben — zweimal
dieselbe Herleitung, die sich für jede neue Shape und jede neue Karte wiederholen
müsste und deshalb nicht skaliert.

Team: Niklas Becker-Klöser und Daria Elagina

Der Tuner läuft für zehn von elf geprüften Einsum-Formen und auf zwei sehr
verschiedenen Karten, der NVIDIA GB10 (DGX Spark) und der RTX 3070. Statt aller
342 überlebenden Configs misst er je Shape rund 22 und erreicht damit 99.1 % des
Voll-Sweep-Optimums.

Die Entstehungsgeschichte mit allen Zwischenständen ist in der Arbeits-Roadmap
``project/projekt_b_cutile_autotuner.md`` festgehalten.

.. Nummerierung kommt aus custom.css (CSS-Counter), nicht aus .. sectnum:: --
   sectnum zaehlt den Seitentitel als Ebene 1 und faengt deshalb bei 1.1 an.

.. contents:: Inhaltsverzeichnis
   :local:
   :depth: 3
   :class: word-toc

.. include:: /project/aufgabenstellung.rst

.. include:: /project/umsetzung.rst

.. include:: /project/erweiterungen.rst

.. include:: /project/evaluation.rst

.. include:: /project/schlussfolgerungen.rst

.. include:: /project/limitations.rst
