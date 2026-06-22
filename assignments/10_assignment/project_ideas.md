# Projektideen (Group Specific Component)

Kurzbeschreibung der beiden Projektideen für den Pitch am 17.06.2026.
Team: Niklas Becker-Klöser und Daria Elagina

## Idee A — XDNA NPU: Durchsatz-optimierter Whole-NPU-GEMM

In Assignment 10 haben wir den GEMM zwar erfolgreich über alle 32 Compute-Tiles
zum Laufen gebracht, die Datenbewegung und die Layout-Transformationen sind aber
noch stark auf genau die vorgegebenen Größen festgeschrieben. In diesem Projekt
wollen wir daraus einen parametrisierbaren Whole-NPU-GEMM machen: Aus Matrixgrößen
und Tile-Splits sollen die FIFO-Typen, `dimensionsToStream`-Angaben,
DMA-Deskriptoren und Offsets systematisch erzeugt werden, inklusive einer sauberen
Beschreibung des Output-Layouts auf dem Memory-Tile. Danach benchmarken wir mehrere
zulässige Split-Konfigurationen und vergleichen, ob wir uns dem bekannten
bf16-XDNA2-Referenzwert von 14.71 TOPS annähern können oder ob Kernel-Performance,
Speicherbandbreite oder Descriptor-Scheduling limitieren.

## Idee B — GPU / cuTile: Auto-Tuning für Tensor-Kontraktionen

In Assignment 05 haben wir die L2-optimale Aufteilung für eine Kontraktion von Hand
hergeleitet und begründet — für eine neue Kontraktion oder eine andere GPU müsste
man diese Überlegung aber jedes Mal von vorne anstellen. Die Idee für dieses Projekt
ist ein bewusst eingeschränkter Auto-Tuner für den Optimizer aus Assignment 05:
Er erzeugt nur einen kleinen, gültigen Suchraum an Tile-Splits und
Ausführungsreihenfolgen, generiert daraus cuTile-Kernelvarianten aus Templates und
misst sie mit `do_bench`. Als Mindestziel reproduzieren wir die handoptimierte
batched Matmul aus Assignment 05; danach erweitern wir auf die Tensor-Ring-
Kontraktion aus Assignment 06 und werten aus, ob das automatische Tuning die
manuelle Lösung erreicht oder schlägt.
