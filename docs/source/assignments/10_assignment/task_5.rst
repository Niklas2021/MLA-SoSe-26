Task 5: Selection of Spatial Dimensions (optional)
==================================================

Aufgabenstellung
----------------

Erklären, wie sich der Datentransfer ändert, wenn die Rollen der Dimensionen
``y`` und ``x`` getauscht werden, d. h.

- ``M → a·y·p·m`` mit ``a=4``, ``y=4``, ``p=2``, ``m=8``,
- ``N → b·x·q·n`` mit ``b=1``, ``x=8``, ``q=2``, ``n=8``, und
- ``K → c·r·k``   mit ``c=16``, ``r=8``, ``k=8``.

In welchen Fällen sollte ``in0`` entlang der Zeilen broadcastet werden, in
welchen ``in1``? Wann ist kein Performance-Unterschied zu erwarten?

Lösung
------

Was sich ändert
~~~~~~~~~~~~~~~

Räumlich sind weiterhin die **Spalten** (8) und **Zeilen** (4) des Arrays. Neu
ist nur, welche logische Dimension darauf abgebildet wird:

- Bisher: ``x`` (Teil von ``M``) → Spalten, ``y`` (Teil von ``N``) → Zeilen.
- Jetzt:  ``x`` (Teil von ``N``) → Spalten, ``y`` (Teil von ``M``) → Zeilen.

``M`` wird nun über die **Zeilen** zerlegt (``y``), ``N`` über die **Spalten**
(``x``). Damit wandern die Broadcast-Richtungen mit:

- ``in0`` hängt von ``M`` (also ``y``) ab, nicht von ``x``. Es ist damit für
  alle Spalten einer festen Zeile gleich und wird **entlang der Zeilen
  broadcastet**.
- ``in1`` (hängt von ``N``, also von ``x``, ab, nicht von ``y``) ist für alle
  Zeilen einer festen Spalte gleich und wird **entlang der Spalten
  broadcastet**.

Das ist genau die Vertauschung gegenüber der Originalaufteilung, wo ``in0``
entlang der Spalten und ``in1`` entlang der Zeilen broadcastet wurde.

Regel
~~~~~

Eine Eingabe wird **entlang der Zeilen** broadcastet, wenn die von ihr
*nicht* indizierte Matrixdimension auf die Zeilen abgebildet ist:

- ``in0`` entlang der Zeilen ⇔ ``M`` ist auf die Zeilen verteilt (``y ⊂ M``).
- ``in1`` entlang der Zeilen ⇔ ``N`` ist auf die Zeilen verteilt (``y ⊂ N``).

Im Original (``y ⊂ N``) wird ``in1`` entlang der Zeilen broadcastet; in der
getauschten Variante (``y ⊂ M``) ist es ``in0``.

Output-Seite
~~~~~~~~~~~~~

Der Join erfolgt immer entlang einer **Spalte** (die Zeilen einer Spalte werden
zusammengeführt). Im Original lieferten die vier Zeilen einer Spalte
verschiedene ``N``-Anteile (``y ⊂ N``) → eine Spalte ergab volle ``N``-Breite.
In der getauschten Variante liefern die Zeilen verschiedene ``M``-Anteile
(``y ⊂ M``) → eine Spalte ergibt einen ``M``-Streifen, und ``x ⊂ N`` verteilt
``N`` über die Spalten. Die Join-Offsets adressieren entsprechend ``M`` statt
``N``.

Wann kein Performance-Unterschied?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Die bewegte Datenmenge ist in beiden Varianten identisch (dieselbe
Gesamtmatrix). Ein Unterschied entsteht nur durch die **Asymmetrie des
Spatial-Grids** (8 Spalten ≠ 4 Zeilen) und die unterschiedlichen
Shim-Bandbreiten je Richtung:

- Solange beide Eingaben über je eine Broadcast-Achse repliziert werden und das
  Grid quadratisch wäre (``#Spalten = #Zeilen``), wäre **kein** Unterschied zu
  erwarten — die Rollen sind dann symmetrisch.
- Bei ``8 × 4`` hängt es davon ab, welche Eingabe über die **längere** Achse
  (8 Spalten) repliziert wird und wie viele ``L3L2``-Streams pro Richtung nötig
  sind. Ist die Zahl der ``L3L2``-Transfers und die pro Shim eingehende
  Datenmenge in beiden Varianten gleich groß, ist ebenfalls kein
  Performance-Unterschied zu erwarten.
