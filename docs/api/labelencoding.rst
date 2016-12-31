Supported Encodings
======================

The design of this packages revolves around a number of immutable
types, each of which representing a specific label-encoding.
These types are contained within their own namespace ``LabelEnc``.
The reason for the namespace is mainly convenience, as it allows
for a simple form of autocompletion and also more concise names that
could otherwise be considered to be too ambiguous.

Shared Interface
-------------------

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

Abstract LabelEncoding
-------------------------

.. type:: LabelEncoding{T,K,M}

   Abstract super-type of all label encodings. Mainly intended for
   dispatch. As such this type is not exported.

   .. attribute:: T

      The label-type of the encoding, which specifies which concrete
      type a label has.

   .. attribute:: K

      The number of labels that the label-encoding can deal with.
      So for binary encodings this will be the constant ``2``

   .. attribute:: M

      The number of array dimensions that the encoding works with.
      For most encodings this will be ``1``, meaning that a target
      array of that encoding is expected to be some vector.
      In contrast to this the encoding :class:`OneOfK` has ``M=2``,
      because it represents the target array as a matrix.


