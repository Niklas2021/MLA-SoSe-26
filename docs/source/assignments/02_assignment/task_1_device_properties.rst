Task 1: GPU Device Properties
==============================

Aufgabenstellung
----------------

Über ``cp.cuda.Device().attributes.items()`` sollen die Werte für
``L2CacheSize``, ``MaxSharedMemoryPerMultiprocessor`` und ``ClockRate``
auf dem DGX Spark ausgelesen und reportiert werden.

Implementierte Funktion
-----------------------

.. literalinclude:: ../../../../assignments/02_assignment/src/task1.py
   :language: python

Output
-------------
.. code-block:: text

    L2CacheSize: 25165824
    MaxSharedMemoryPerMultiprocessor: 102400
    ClockRate: 2418000

1. L2CacheSize: 25.165.824 = ~25 MB (25MB laut Specification)
2. Max Shared Memory per SM: 102.400 = ~100KB (128KB laut Specification)
3. ClockRate: 2.418.000 = ~2.4 GHz Taktfrequenz (2.5GHz laut Specification)
