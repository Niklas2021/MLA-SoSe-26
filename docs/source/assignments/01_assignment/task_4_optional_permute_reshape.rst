Task 4 (Optional): Permute und Reshape
======================================

Aufgabenstellung
----------------

Zu beantworten sind im Kern diese Punkte:

1. Wie ändern sich ``shape`` und ``stride`` nach ``permute`` / ``reshape``?
2. Werden Daten im Speicher verschoben?
3. Was ist der Unterschied zwischen ``reshape`` und ``view``?
4. Warum kann ``reshape`` bei einem permutierten Tensor anders reagieren
   als bei einem frisch erzeugten Tensor?

Implementiertes Test-Skript
---------------------------

.. literalinclude:: ../../../../assignments/01_assignment/src/assignment_01_optional_permute_reshape.py
   :language: python

Konkreter Output des Test-Skripts
---------------------------------

.. literalinclude:: task4_optional_output.txt
   :language: text

Antworten auf die Fragen 1 bis 4
-----------------------------------------------------------

1. ``shape`` und ``stride`` ändern sich nach ``permute`` deutlich, was man direkt sieht:
   ``base`` hat ``shape=(2, 3, 4), stride=(12, 4, 1)``, aber ``permuted`` hat
   ``shape=(3, 2, 4), stride=(4, 12, 1)``. Bei ``reshaped_base`` sieht man dagegen
   ``shape=(6, 4), stride=(4, 1)``.

2. Die Daten werden bei ``permute`` in diesem Beispiel nicht verschoben:
   ``base`` und ``permuted`` teilen Speicher (``True``) und haben denselben ``storage_ptr``.
   Bei ``reshaped_base`` ist es ebenfalls noch derselbe Speicher. Bei
   ``reshaped_permuted`` ist der ``storage_ptr`` aber anders und
   ``permuted und reshaped_permuted teilen Speicher: False``.

3. ``reshape`` und ``view`` unterscheiden sich praktisch daran, dass ``view``
   nur mit passendem Speicherlayout funktioniert. Das sieht man am Output:
   ``view auf permuted: funktioniert nicht`` mit der gezeigten Fehlermeldung.
   ``reshape`` auf demselben Tensor funktioniert hingegen.

4. Warum ``reshape`` nach ``permute`` anders reagieren kann:
   ``permute`` erzeugt hier eine nicht-zusammenhängende Sicht
   (``contiguous=False`` bei ``permuted``). Deshalb kann ``view`` nicht greifen und
   ``reshape`` muss für ``reshaped_permuted`` auf neue Daten wechseln
   (anderer ``storage_ptr``). Beim ursprünglichen ``base`` mit ``contiguous=True``
   ist ``view`` dagegen möglich (``base_view``) und bleibt auf denselben Daten.
