Erweiterter Suchraum
^^^^^^^^^^^^^^^^^^^^

.. Quelle: search.py (SearchSpace.wide, WIDE_*_CHOICES)

Motivation aus der Randanalyse
""""""""""""""""""""""""""""""

.. Inhalt:
   - Die Diagnose: sitzt der Gewinner am Rand des Gitters, ist der Raum
     abgeschnitten. Auf der GB10 46 % der Gewinner-Koordinaten am Rand, auf der
     3070 65 % -- und fast alle am UNTEREN.
   - Schlussfolgerung: der Suchraum wurde auf der GB10 entworfen und passt zu ihr.
   - WICHTIG: die 3070-Zahl stammt aus dem alten Sweep, der sich spaeter als fuer
     batch=1 unbrauchbar herausgestellt hat. Diese Einschraenkung muss hier
     stehen, mit Verweis auf Evaluation/Datenbasis.

Umsetzung
"""""""""

.. Inhalt:
   - M/N_PRIM {32,64,128,256}, K_PRIM {16,32,64,128}; 32 bleibt mit MMA_ALIGN=16
     sauber. Raum 486 -> 1152, nach Pruning 342 -> 954.
   - Bewusst NICHT Default: die bisherigen Messungen und die Hybrid-Auswertung
     beziehen sich auf den engen Raum, ein stiller Wechsel wuerde die
     Vergleichbarkeit zerstoeren.
   - Der Cache unterscheidet beide ueber space_size -- ein eng getunter Eintrag
     bedient keine --wide-Anfrage (dieselbe Logik wie topk vs hybrid).

Ergebnis
""""""""

.. Inhalt:
   - GB10: netto ein Nullsummenspiel bei doppelten Kosten. Ohne die zwei
     Ausreisser 98.8 %.
   - RTX 3070: +6.0 % im geometrischen Mittel -- hier traegt es.
   - Der eine grosse Gewinn ist sauber erklaerbar und sollte ausgeschrieben
     werden: a06_krumm +46 % auf der GB10, weil p=48 mit k_prim=16 exakt in drei
     Kacheln aufgeht statt auf 64 gepaddet zu werden. Der Gewinn kommt also nicht
     daher, dass kleine Tiles "besser" waeren, sondern dass sie die Shape TEILEN.
   - Daraus die Lehre fuer die Zukunft: den Raum adaptiv waehlen (kleine k_prim
     nur, wenn K % 32 != 0) statt ihn global zu verdoppeln. Als offener Punkt
     markieren, nicht als umgesetzt.
