Supported Encodings
======================

The design of this packages revolves around a number of immutable
types, each of which representing a specific label-encoding.
These types are contained within their own namespace ``LabelEnc``.
The reason for the namespace is mainly convenience, as it allows
for a simple form of autocompletion and also more concise names that
could otherwise be considered to be too ambiguous.

TrueFalse
-----------

.. type:: LabelEnc.TrueFalse

   TODO

.. function:: label(::LabelEnc.TrueFalse) -> [true, false]

.. function:: nlabel(::LabelEnc.TrueFalse) -> 2

ZeroOne
-----------

.. type:: LabelEnc.ZeroOne

   TODO

   .. attribute:: cutoff


MarginBased
------------

.. type:: LabelEnc.MarginBased

   TODO

OneVsRest
------------

.. type:: LabelEnc.OneVsRest

   TODO

   .. attribute:: poslabel

   .. attribute:: neglabel

Indices
------------

.. type:: LabelEnc.Indices

   TODO

OneOfK
-------------

.. type:: LabelEnc.OneOfK

   TODO

NativeLabels
-------------

.. type:: LabelEnc.NativeLabels

   TODO

   .. attribute:: label

   .. attribute:: invlabel

FuzzyBinary
-------------

.. type:: LabelEnc.FuzzyBinary

   A vector-based binary label interpretation without a specific
   labeltype. It is primarily intended for fuzzy comparision of
   binary true targets and predicted targets.
   It basically assumes that the encoding is either `TrueFalse`,
   `ZeroOne`, or `MarginBased` by treating all non-negative values
   as positive outputs.

