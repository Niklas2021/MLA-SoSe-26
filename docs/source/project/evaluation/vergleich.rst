Vergleich der beiden Karten
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Hardware im Detail
""""""""""""""""""

Die beiden Karten sind sehr unterschiedlich gebaut, und genau die Unterschiede
erklären, warum das Tuning auf ihnen verschieden viel bringt.

=========================  =====================  =====================
Merkmal                    GB10                   RTX 3070
=========================  =====================  =====================
Architektur                Blackwell (GB10)       Ampere (GA104)
SMs                        48                     46
SM-Takt                    2.418 GHz              1.725 GHz
L2-Cache                   25 MB                  4 MB
Speicher                   LPDDR5X, integriert    GDDR6, dediziert
Speichertakt (effektiv)    8.533 GHz              14 GHz
Busbreite                  256 bit                256 bit
Bandbreite (rechnerisch)   273 GB/s               448 GB/s
SMEM je Block (Opt-in)     99 KB                  99 KB
SMEM je SM                 100 KB                 100 KB
Register je Block          65536                  65536
fp16-Peak (geschätzt)      119 TFLOPS             81 TFLOPS
=========================  =====================  =====================

Ehrlich gekennzeichnet: fast alle Werte kommen direkt aus den CUDA-Attributen. Zwei
sind es nicht. Die fp16-Tensor-Leistung pro SM und Takt (512 FMAs) ist eine
Architektur-Schätzung — sie setzt nur den Roofline-Umschaltpunkt, nicht die
Reihenfolge der Configs. Und der 3070-Speichertakt ist so gesetzt, dass die
effektive GDDR6-Bandbreite von rund 448 GB/s hinkommt.

Drei Unterschiede zählen für den Tuner. Der L2 ist mit 25 MB gegen 4 MB um den
Faktor sechs größer. Der Speicher ist bei der GB10 integriertes LPDDR5X, das sie
sich mit der CPU teilt (was auch den :ref:`Messrahmen-Effekt <messrahmen>`
verursacht), bei der 3070 dediziertes GDDR6. Und daraus folgt ein
Regime-Unterschied: die GB10 hat mehr Rechenleistung (119 gegen 81 TFLOPS) bei
weniger Bandbreite (273 gegen 448 GB/s) und dazu das große L2 — sie arbeitet also
deutlich compute-limitierter als die 3070.

Wie viel hat das Tuning gebracht
""""""""""""""""""""""""""""""""

Der Nutzen lässt sich auf zwei Config-Ebenen auftragen, jeweils als Anteil am je
Shape besten gefundenen Wert. Absolute TFLOPS taugen dafür nicht: die beiden
Karten haben unterschiedliche Peak-Leistung und Bandbreite, ihre TFLOPS sind
nicht direkt vergleichbar. Der Anteil am Erreichbaren ist es.

.. image:: ../../project/praesentation/figures/fig_tuning_ladder.png
   :alt: Anteil am je Shape besten Wert, fremde und feste Config, GB10 gegen RTX 3070
   :width: 95%

Die 3070 startet auf beiden Ebenen deutlich tiefer und hat entsprechend mehr
aufzuholen. Die GB10-Werte beziehen sich dabei auf das Voll-Sweep-Optimum, die
3070-Werte mangels Voll-Sweep auf den besten je Shape gefundenen Wert (Hybrid);
beide Referenzen sind praktisch gleichwertig, auf der GB10 liefert die
Hybrid-Referenz 79 % statt 78.7 %. Die frühere ``fig_crossgpu_lever`` zeigte noch
die zurückgezogenen 1.88×-Zahlen aus gemischten v1/v2-Daten und wird nicht mehr
verwendet.

Daraus lassen sich zwei Aussagen sauber trennen, die vorher unter „Ø 1.88× auf
der 3070" vermischt waren:

1. **Was kostet es, beim GPU-Wechsel die alte Config zu behalten?** Die auf der
   GB10-Heimatshape entstandene A05-Config holt auf der 3070 nur rund 62 % —
   Neu-Tunen bringt dort also etwa 1.6×. Das rechtfertigt den GPU-Modell-Anteil im
   Cache-Key.
2. **Was bringt Per-Shape-Tuning gegen eine passende feste Config derselben
   Karte?** 1.12× auf der GB10, 1.32× auf der 3070.

Warum die kleinere Karte mehr profitiert
""""""""""""""""""""""""""""""""""""""""

Der interessante Punkt steckt in der mittleren Zeile: die beste feste Config holt
auf der 3070 nur 75.9 %, auf der GB10 aber 89.5 %. Auf der 3070 liegen die Optima
der einzelnen Shapes also viel weiter auseinander — eine einzelne feste Config
kann es keiner Shape recht machen, und genau deshalb lohnt Per-Shape-Tuning dort
mehr.

Ein Erklärungsversuch, ausdrücklich als solcher: bei nur 4 MB L2 hängt mehr an der
Wahl der L2-Gruppe, weil der Working-Set einer großen Gruppe nicht mehr
hineinpasst und der Traffic dann wirklich zählt. Das 25-MB-L2 der GB10 verzeiht
dagegen fast jede Gruppenwahl, weil ohnehin alles im Cache bleibt. Mit den
vorhandenen Daten lässt sich das nur teilweise belegen — für eine saubere Analyse
fehlt der 3070-Voll-Sweep.

Eine früher berichtete Aussage gehört hier korrigiert: dass die beste Config „in
16 von 16 Shapes je Karte verschieden" sei, beruhte auf dem defekten alten
3070-Sweep und wird nicht mehr in dieser Form vertreten.

Portabilität des Verfahrens
"""""""""""""""""""""""""""

Was über beide Karten unverändert lief, ist die eigentliche gute Nachricht:
Enumerator, Pruning, beide Kernel-Familien, der flexible Kernel, die Hybrid-Suche
und der Cache mit GPU-Schlüssel. Nichts davon ist auf die GB10 hartkodiert, alles
läuft über die ``DeviceProperties`` der jeweiligen Karte.

Genau an dieser Stelle lauerte allerdings ein Fehler. Fehlte ``cupy``, fiel die
Erkennung still auf die GB10-Properties zurück — und rechnete dann mit 25 MB L2 und
48 SMs auf einer 3070. Aufgefallen ist das, weil ein 3070-Lauf „GPU: NVIDIA GB10"
ins Log schrieb. Seitdem bricht eine unbekannte Karte ab, statt falsche Werte
anzunehmen. Für ein Verfahren, dessen ganzer Zweck die Anpassung an die Hardware
ist, wäre eine still angenommene falsche Karte der denkbar schlechteste Fehler
gewesen.
