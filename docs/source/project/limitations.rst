Limitations
-----------

Grenzen des Verfahrens
^^^^^^^^^^^^^^^^^^^^^^

Der Koordinatenabstieg ist greedy, und die Achsen sind nicht unabhängig:
``m_prim`` und ``m_l2`` bestimmen gemeinsam, auf welches Vielfache die M-Achse
gepaddet wird. Ein Abstieg, der beide getrennt anfasst, kann in einem lokalen
Optimum hängen bleiben. Am deutlichsten bei ``krumm`` — dort erreicht der Hybrid
98.1 % des Optimums, während er sonst bei 99–100 % liegt.

Das analytische Kostenmodell rankt schwach. Über die 16 Shapes liegt die
Spearman-Korrelation zwischen Schätzung und Messung beim Bandbreiten-Modell bei
+0.02, bei der Roofline immerhin bei +0.50 (die dafür als Vorfilter schlechter
abschneidet, siehe :ref:`Ranking <ranking>`). Der Grund ist teils fundamental:
auf der GB10 ist der Betrieb compute-limitiert und das L2 so groß, dass der
DRAM-Traffic über die Configs kaum variiert — dem Modell fehlt schlicht das
Signal, an dem es unterscheiden könnte. Ein besseres Modell hätten wir gern
gehabt, aber es zieht nur, wo die Configs sich im Traffic überhaupt
unterscheiden.

Der Suchraum ist von Hand gesetzt und war ursprünglich auf der GB10 entworfen.
Die adaptive Wahl (kleine Tiles pro Achse nur, wo die Shape sie braucht) ist
inzwischen umgesetzt (:ref:`Erweiterter Suchraum <erweiterter-suchraum>`), das
Grundraster der Werte selbst bleibt aber eine begründete Wahl, keine Herleitung.

Grenzen der Abdeckung
^^^^^^^^^^^^^^^^^^^^^

Der Tuner deckt zwei Inputs in fp16 mit fp32-Akkumulation und Row-Major-Layout
ab, in zwei Kernel-Familien (GEMM wie A05, Ring wie A06). Eine neue Topologie —
andere Batch- oder Reduktionsstruktur, mehr als zwei Inputs — braucht ein neues
Kernel-Template. Der Tuner automatisiert die Config-Suche, nicht das Schreiben
des Kernels.

Die Batch-Dimensionen müssen außen stehen und in allen drei Tensoren gleich
sortiert sein. Ein String wie ``mck,ckn->mcn`` (Batch innen) wird deshalb
abgelehnt, weil das eine echte Transposition der Daten wäre und nicht per
``view`` zu beheben ist.

Zwei Kernel-Grenzen bleiben offen, beide bewusst nicht geschlossen:

* Variante B gibt es nur für das kanonische Layout und ``order=0``. Der
  flexible Kernel für transponierte Layouts (NT/TN) existiert nur für Variante A.
  Das ist verschmerzbar, weil Variante A in der Vollmessung alle 16 Shapes
  gewinnt — bestes B liegt bei 76–98 % von bestem A, für die Ring-Familie gibt es
  B gar nicht. Ein transponiertes B würde also nie gewählt, ein eigener Kernel
  dafür wäre toter Code.
* Die Verschachtelung der beiden Reduktionen im Ring-Kernel ist fest: die
  ``s``-Achse läuft außen um die prim-K-Schleife. Welche der beiden außen läuft,
  ist nicht Teil des Suchraums, obwohl ``s`` auf allen acht Ring-Shapes
  nichttrivial ist (Größe 8–128).

Grenzen der Messung
^^^^^^^^^^^^^^^^^^^

Der :ref:`Messrahmen-Effekt <messrahmen>` begrenzt, was sich vergleichen lässt:
dieselbe Config misst je nach Lastdauer bis zu 4.6 % auseinander. Alle
berichteten Vergleiche müssen deshalb innerhalb einer Messreihe gelesen werden.

Das Spitzenfeld liegt ohnehin innerhalb weniger Prozent. Unterschiede unter etwa
3 % sind mit unserem Aufbau nicht interpretierbar — das ist auch der Grund, warum
ein perfekter Ranker wenig zusätzlichen Nutzen hätte.

Wir berichten Einzelwerte beziehungsweise Mediane aus wenigen Runden. Es gibt
keine Wiederholungsmessungen mit Streuungsangabe und damit keine
Konfidenzintervalle.

Grenzen der Datenbasis
^^^^^^^^^^^^^^^^^^^^^^

Der erste 3070-Sweep ist für ``batch=1`` unbrauchbar (Faktor 3.3–4.8 zu niedrig),
die Ursache ließ sich nicht mehr bestimmen (siehe
:ref:`Datenbasis <datenbasis>`). Ein Ersatz hätte einen vollen Sweep von rund
3.5 Stunden gekostet und wurde bewusst nicht gemacht. Die 3070-Baseline steht
deshalb auf der Sonde statt auf einer Vollmessung, mit bekanntem Bias.

Zwei Karten sind eine dünne Grundlage für Portabilitätsaussagen, zumal beide
NVIDIA-Consumer- beziehungsweise Workstation-Klasse sind, keine Datacenter-GPU.

Die „beste feste Config" als Baseline ist mit Nachwissen aus der Vollmessung
gewählt. Sie ist damit eine optimistische, also konservative Schranke für den
Tuner-Gewinn — gegen eine schwächere Baseline sähe der Tuner besser aus.

Was wir mit mehr Zeit gemacht hätten
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Drei Punkte ließen sich noch angehen, in absteigender Attraktivität:

* **Ein gelernter Ranker aus den vorhandenen Sweeps.** Wir haben 16 Shapes mit je
  171–342 gemessenen Configs, also mehrere tausend gelabelte Punkte. Ein
  einfacher gelernter Ranker (etwa Gradient Boosting auf Tile- und Gruppengrößen,
  Padding-Ratio, Occupancy-Schätzung, Arithmetic Intensity) würde das Ranking
  gegenüber dem analytischen Modell deutlich verbessern, und das komplett offline
  aus den CSVs, ohne neuen GPU-Lauf. Der Preis wäre der Bruch mit dem erklärbaren
  analytischen Ansatz, und der praktische Gewinn ist gedeckelt, weil der Hybrid
  schon bei 99 % liegt.
* **Kopplungsbewusste Abstiegsschritte.** Der Abstieg bewegt ``m_prim`` und
  ``n_prim`` schon gemeinsam (wegen der Register). Auch ``(m_prim, m_l2)`` und
  ``(n_prim, n_l2)`` als Paar zu bewegen träfe die Padding-Kopplung, an der
  ``krumm`` hängt. Kostet mehr Messungen, der Deckel ist aber niedrig.
* **Die s/k-Verschachtelung als Suchachse.** Ein Knopf für die Reihenfolge der
  beiden Ring-Reduktionen. Anders als die ersten beiden Punkte führt das neuen
  Kernel-Code ein und ließe sich nicht offline validieren — es bräuchte einen
  GPU-Lauf.

Dazu die Mess- und Datenlücken, die keine Umsetzung sind, sondern Zeit gekostet
hätten: ein frischer 3070-Voll-Sweep als Ersatz für den unbrauchbaren alten,
Wiederholungsmessungen mit Streuungsangabe und eine dritte Karte mit anderer
L2-Größe für belastbarere Portabilitätsaussagen.
