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

We can also query the labels that an encoding uses to represent
the classes.

.. function:: label(encoding) -> Vector

   Returns all the labels that a specific encoding uses in their
   correct order.

   :param LabelEncoding encoding: The specific label-encoding.

   :return: The unique labels in the form of a vector. In the case
            of two labels, the first element will represent the
            positive label and the second element the negative label
            respectively.

.. code-block:: jlcon

   julia> label(LabelEnc.MarginBased())
   2-element Array{Float64,1}:
     1.0
    -1.0

We can also query the number of labels that a concrete encoding uses
to represent the classes.

.. function:: nlabel(encoding) -> Int

   Returns the number of labels that a specific encoding uses.

   :param LabelEncoding encoding: The specific label-encoding.

.. code-block:: jlcon

   julia> nlabel(LabelEnc.NativeLabels([:a,:b,:c]))
   3

More interestingly, we can query the type on an encoding for
the number of labels it can represent. This allows for some compile
time decisions.

.. function:: nlabel(type) -> Int

   Note that this function will fail if the number of labels can
   not be inferred from the given type.

   :param DataType type: Some subtype of :class:`LabelEncoding{T,K,M}`

.. code-block:: jlcon

   julia> nlabel(LabelEnc.ZeroOne)
   2

   julia> nlabel(LabelEnc.OneOfK)
   ERROR: ArgumentError: number of labels could not be inferred for the given type


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

