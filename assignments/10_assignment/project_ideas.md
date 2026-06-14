# Projektideen (Group Specific Component)

Kurzbeschreibung der beiden Projektideen für den Pitch am 17.06.2026.
Team: Niklas Becker-Klöser und Daria Elagina

## Idee A — XDNA NPU: Durchsatz-optimierter Whole-NPU-GEMM

In Assignment 10 haben wir den GEMM zwar erfolgreich über alle 32 Compute-Tiles
zum Laufen gebracht, mussten die Datenbewegung aber bewusst ausbremsen, damit das
Ergebnis stimmt: Zwischen den beiden Durchläufen über die `a`-Dimension steht eine
Barriere, die das eigentlich mögliche Überlappen von Datentransfer und Rechnung
verhindert. In diesem Projekt wollen wir genau dort ansetzen und die Datenbewegung
so umbauen, dass die Transfers von L3 über L2 nach L1 wieder parallel zur Rechnung
laufen (Double-Buffering), und nebenbei die Aufteilung der Matrix auf die Tiles für
beliebige Größen einstellbar machen. Am Ende wollen wir über verschiedene
Konfigurationen messen, wie viel schneller das wird und ab welchem Punkt nicht mehr
die Rechenleistung, sondern die Speicherbandbreite des NPU der limitierende Faktor
ist.

## Idee B — GPU / cuTile: Auto-Tuning für Tensor-Kontraktionen

In Assignment 05 haben wir die L2-optimale Aufteilung für eine Kontraktion von Hand
hergeleitet und begründet — für eine neue Kontraktion oder eine andere GPU müsste
man diese Überlegung aber jedes Mal von vorne anstellen. Die Idee für dieses Projekt
ist, den Optimizer aus Assignment 05 so zu erweitern, dass er die guten
Tiling-Konfigurationen selbst findet, indem er verschiedene Varianten durchprobiert
und mit `do_bench` misst, welche davon am schnellsten läuft. Als Testfälle nehmen wir
die batched Matmul aus Assignment 05 und die Tensor-Ring-Kontraktion aus Assignment
06 und schauen, ob das automatische Tuning unsere handgemachte Lösung erreicht oder
sogar schlägt.
