Task 4: Data Layouts and Pointer Updates
=========================================

Aufgabenstellung
----------------

Das Datenlayout und die nötigen Pointer-Updates skizzieren, die zum gewählten
Register-Blocking passen.

Lösung
------

L1-Layout (alle BF16, 2 Byte/Element; eine 8x8-Kachel = 128 Byte):

==========  ===================  ==================  ==============================
Tensor      L1-View              8x8-Kachel          Byte-Offset der Kachel
==========  ===================  ==================  ==============================
``in0``     ``prmk`` (2,8,8,8)   ``mk``              ``(p*8 + r) * 128``
``in1``     ``rqkn`` (8,2,8,8)   ``kn`` -> ``nk``    ``(r*2 + q) * 128``
``out``     ``pqmn`` (2,2,8,8)   ``mn``              ``(p*2 + q) * 128``
==========  ===================  ==================  ==============================

Eine 8x8-BF16-Kachel (1024 bit) wird mit **zwei** 512-bit-Loads geladen
(``[ptr,#0]`` low, ``[ptr,#64]`` high); ein FP32-Akkumulator (2048 bit) fasst
eine ganze 8x8-Kachel.

Pointer-Schema (nur Post-Increment-Walks und ``mov p,p``-Resets — **keine
Pointer-Arithmetik im Skalar-Register**, das hatte im ersten Versuch die
Pointer korrumpiert):

.. code-block:: text

   p0 (in0):  je r: cml2[p0],#64 ; cmh2[p0],#64   -> +128/r
              laeuft kontinuierlich; nach Pass p=0 (8*128=1024) steht p0
              exakt auf in0[1][0] -> fliesst in Pass p=1 (kein Reset)

   p1 (in1):  je r: x2[p1],#64 ; x4[p1],#64 (q=0) ; x3[p1],#64 ; x5[p1],#64 (q=1)
              -> +256/r (beide q-Kacheln sind im Speicher benachbart)
              Reset zu Beginn jedes Passes: mov p1, p3   (p3 = in1-Basis)

   p2 (out):  Laden mit Offsets [p2,#0..#192] (out[p][0], out[p][1]);
              Speichern mit Post-Increment -> +256/Pass -> fliesst nach p=1

Datenfluss pro ``r`` (asymmetrisch, weil nur ``in1`` transponiert werden muss):

.. code-block:: text

   in0[p][r]:  vlda.conv (cml2/cmh2) -> dm2 --vconv--> ex0  (mk, direkt)
   in1[r][q]:  vldb (x2,x4) -> shuffle (x6,x7) -> vmul*1.0 -> dm3 --vconv--> ex6
               (kn  ->  nk transponiert  ->  FP32  ->  BFP16)
   MAC:        dm0 += ex0 * ex6   (q=0) ;   dm1 += ex0 * ex8   (q=1)

Da ``in0`` und ``out`` kontinuierlich durchlaufen und nur ``in1`` pro Pass auf
``p3`` zurückgesetzt wird, kommt der Kernel ganz ohne Skalar-Pointer-Arithmetik
aus.
