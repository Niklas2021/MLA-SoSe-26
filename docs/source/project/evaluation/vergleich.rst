Vergleich der beiden Karten
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Dieses Kapitel soll die Frage beantworten: wie gut hat der Tuner auf welcher
   Karte geholfen, und warum unterscheidet es sich? Der Auftraggeber will hier
   ausdruecklich detaillierte Hardware-Specs.

Hardware im Detail
""""""""""""""""""

.. Inhalt:
   - Grosse Vergleichstabelle beider Karten. Werte aus project/project_diary.md
     (vollstaendiger cuda-Attribut-Dump der GB10) und
     project/src/autotuner/device_props.py. Zeilen mindestens:
     Architektur / Compute Capability / SM-Zahl / Kerntakt / L2-Cache /
     Speichertyp und -takt / Busbreite / rechnerische Bandbreite /
     MaxSharedMemoryPerBlock und Optin / ReservedSharedMemoryPerBlock /
     MaxRegistersPerBlock und PerMultiprocessor / MaxThreadsPerMultiProcessor /
     MaxBlocksPerMultiprocessor / Integrated ja-nein / rechnerischer fp16-Peak.
   - Die drei Unterschiede hervorheben, die fuer den Tuner zaehlen:
     L2 25 MB gegen 4 MB (Faktor 6), integrierter LPDDR-Shared-Memory gegen
     dediziertes GDDR6, und der daraus folgende Unterschied im
     Compute-vs-Bandbreite-Regime.
   - Ehrlich kennzeichnen, welche Werte gemessen und welche geschaetzt sind
     (tensor_flop_per_sm_cycle ist eine Architektur-Schaetzung; die
     3070-Speichertaktangabe ist so gesetzt, dass die effektive Bandbreite
     hinkommt).

Wie viel hat das Tuning gebracht?
"""""""""""""""""""""""""""""""""

.. Inhalt:
   - Die zentrale Tabelle, drei Ebenen pro Karte:
     fremde Config (A05-Default) / passende feste Config / Per-Shape-Tuning,
     jeweils als Anteil am Per-Shape-Optimum.
     GB10: 78.7 % - 89.5 % - 100 %.  3070: 39.1 % - 75.9 % - 100 %.
   - Daraus die beiden sauber getrennten Aussagen (Kosten des Nicht-Neutunens
     beim GPU-Wechsel gegen Nutzen des Per-Shape-Tunings).
   - fig_crossgpu_lever.png -- pruefen, ob die Figur noch die alten Zahlen zeigt;
     falls ja, neu erzeugen oder nicht verwenden.

Warum die kleinere Karte mehr profitiert
""""""""""""""""""""""""""""""""""""""""

.. Inhalt:
   - Der eigentlich interessante Teil: auf der 3070 holt die beste feste Config nur
     75.9 %, auf der GB10 89.5 %. Die Optima liegen dort also viel weiter
     auseinander.
   - Erklaerungsversuch anbieten und als solchen kennzeichnen: bei 4 MB L2 haengt
     mehr an der Gruppenwahl, waehrend das 25-MB-L2 der GB10 vieles verzeiht.
     Belegen laesst sich das mit den vorhandenen Daten nur teilweise.
   - Beste Config je Shape und Karte gegenueberstellen, soweit die Datenlage es
     zulaesst -- und dabei sagen, dass die frueher berichtete Aussage
     "16/16 Shapes verschieden" auf den defekten Daten beruhte.

Portabilität des Verfahrens
"""""""""""""""""""""""""""

.. Inhalt:
   - Was auf beiden Karten unveraendert funktioniert hat: Enumerator, Pruning,
     beide Kernel-Familien, der flex-Kernel, die Hybrid-Suche, der GPU-Key im
     Cache. Alles ist ueber device_props parametrisiert, nichts ist auf die GB10
     hartkodiert.
   - Der Fehler, der genau hier gelauert hat, gehoert erwaehnt: ein stiller
     Fallback auf GB10-Properties, wenn cupy fehlt. Aufgefallen, weil ein
     3070-Lauf "GPU: NVIDIA GB10" ins Log schrieb. Behoben -- unbekannte Karte
     bricht jetzt ab, statt falsche Werte anzunehmen.
