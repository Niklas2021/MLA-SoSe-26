Limitations
===========

.. Ehrliche Grenzen des Projekts. Nach Art gegliedert, nicht nach Schwere.

Grenzen des Verfahrens
----------------------

.. Inhalt:
   - Koordinatenabstieg ist greedy: die Achsen sind nicht unabhaengig (m_prim und
     m_l2 bestimmen gemeinsam das Padding), deshalb bleibt er in lokalen Optima
     haengen. Beispiel krumm bei 86.3 %, unabhaengig von der Strategie.
   - Das analytische Modell rankt nicht (Spearman ~0 bzw. +0.5 fuer die Roofline,
     die dafuer als Vorfilter schlechter ist). Ein besseres Modell haetten wir
     gern gehabt.
   - Der Suchraum ist von Hand gesetzt und war auf der GB10 entworfen. Adaptive
     Wahl (kleine k_prim nur bei unteilbarem K) ist erkannt, aber nicht umgesetzt.

Grenzen der Abdeckung
---------------------

.. Inhalt:
   - Zwei Inputs, fp16 mit fp32-Akku, Row-Major.
   - Zwei Kernel-Familien. Eine neue Topologie (andere Batch- oder
     Reduktionsstruktur, mehr Inputs) braucht ein neues Template -- der Tuner
     automatisiert die Config-Suche, nicht das Kernel-Schreiben.
   - Batch-Dims muessen aussen stehen und in allen drei Tensoren gleich sortiert
     sein. Was daran liegt und was nicht behebbar waere.
   - Variante B gibt es nur fuer kanonisches Layout und order=0.
   - Die s/k-Verschachtelung im Ring-Kernel ist nicht durchsucht.

Grenzen der Messung
-------------------

.. Inhalt:
   - Der Messrahmen-Effekt: bis 4.6 % Abweichung derselben Config je nach
     Lastdauer. Alle berichteten Vergleiche muessen innerhalb einer Messreihe
     gelesen werden.
   - Das Spitzenfeld liegt ohnehin innerhalb weniger Prozent -- Unterschiede unter
     ~3 % sind nicht interpretierbar.
   - Keine Wiederholungsmessungen mit Streuungsangabe. Wir berichten Einzelwerte
     bzw. Mediane aus wenigen Runden; Konfidenzintervalle gibt es nicht.

Grenzen der Datenbasis
----------------------

.. Inhalt:
   - Der erste 3070-Sweep ist fuer batch=1 unbrauchbar (3.3-4.8x zu niedrig), die
     Ursache liess sich nicht mehr bestimmen. Ein Ersatz haette 3.5 Stunden
     gekostet und wurde bewusst nicht gemacht.
   - Deshalb steht die 3070-Baseline auf der Sonde statt auf einer Vollmessung,
     mit bekanntem Bias (~1.3 Punkte zu niedrig).
   - Zwei Karten sind eine duenne Grundlage fuer Portabilitaetsaussagen; beide sind
     zudem NVIDIA-Consumer/Workstation-Klasse, keine Datacenter-GPU.
   - Die "beste feste Config" als Baseline ist mit Nachwissen gewaehlt und damit
     eine optimistische, also konservative Schranke fuer den Tuner-Gewinn.

Was wir mit mehr Zeit gemacht hätten
------------------------------------

.. Inhalt:
   - Kurze, konkrete Liste statt vager Ausblick: frischer 3070-Voll-Sweep,
     adaptiver Suchraum, NT/TN auch fuer Variante B, s/k-Verschachtelung,
     Wiederholungsmessungen mit Streuung, dritte Karte mit anderer L2-Groesse.
