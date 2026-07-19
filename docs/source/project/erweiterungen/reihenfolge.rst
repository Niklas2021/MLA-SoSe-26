Loop-Reihenfolgen als Suchachse
===============================

.. Quellen: kernels.py (SWIZZLE_MN/OUTER_MN), search.py (ORDER_CHOICES),
   project/src/measure_order.py

Der ungenutzte Freiheitsgrad
----------------------------

.. Inhalt:
   - Befund: Optimizer.permute_dims und fuse_dims stammen aus A05 und modellieren
     beliebige Dim-Reihenfolgen -- der Tuner nutzte davon aber exakt zwei fest
     verdrahtete Ordnungen (Variante A/B), und die Kernel lasen die Config gar
     nicht, sondern dekodierten pid hart.

Vorab-Evidenz aus vorhandenen Daten
-----------------------------------

.. Inhalt:
   - Die Spiegel-Analyse, mit der wir VOR dem Bauen gepruefte haben, ob sich der
     Aufwand lohnt: bei quadratischen Shapes mit m_prim==n_prim ist
     (m_l2=a, n_l2=b) gegen (b,a) rein eine Frage der Swizzle-Richtung. Waere sie
     egal, muesste das Verhaeltnis 1.00 sein -- gemessen Median 1.48 (a05).
   - Diese Art "erst aus vorhandenen Daten pruefen, dann bauen" ist methodisch
     erwaehnenswert.

Umsetzung
---------

.. Inhalt:
   - order als siebter Knopf, zwei Bits: welche Achse die schnellste
     bid-Komponente ist, und welche Gruppen-Achse aussen zuerst laeuft.
     order=0 ist bitgleich das alte Verhalten.
   - Nur fuer Variante A -- bei B sind m_l2/n_l2 SEQ-Loops, es gibt kein Swizzling
     ueber die bid.
   - Suchraum 486 -> 1215, nach Pruning 342 -> 855.
   - Codeausschnitt des pid-Decodes mit den beiden Flags.

Ergebnis
--------

.. Inhalt:
   - Der grosse Einzelfall: die vom --ordered-Lauf gewaehlte a05-Config steht im
     Sweep bei order=0 auf 44.7 und misst mit order=2 dann 67.9 -- +52 % auf
     identischen Tiles. Mechanistisch erklaeren (m_l2=2 -> nur zwei Zeilen hohe
     Gruppe, alte Aussenreihenfolge laeuft erst alle N-Gruppen durch).
   - Aber: am Tuner-Endergebnis nur +1.7 %, und das liegt im Messrauschen.
   - Isoliert gemessen (measure_order.py, Round-Robin gegen Drift): GB10 1.026x,
     RTX 3070 1.012x.
   - Ehrlichkeit an zwei Stellen:
     * unsere Vorhersage war, dass der Knopf auf der 3070 MEHR bringt (kleines L2).
       Gemessen bringt er dort weniger. Vorhersage widerlegt.
     * measure_order nagelt die Tiles auf die 8x8-Gruppe fest -- ausgerechnet die
       quadratische Gruppe, wo der Knopf am wenigsten Hebel hat. Das ist eine
       Schwaeche des Experiments, die benannt gehoert.
   - Kosten: der flex-Kernel kompiliert knapp doppelt so langsam (0.62 s gegen
     0.325 s pro Messung). Deshalb bleibt --ordered optional.
