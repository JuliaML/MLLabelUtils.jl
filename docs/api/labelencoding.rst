Supported Encodings
======================

The design of this packages revolves around a number of immutable
types, each of which representing a specific label-encoding.
These types are contained within their own namespace ``LabelEnc``.
The reason for the namespace is mainly convenience, as it allows
for a simple form of auto-completion and also more concise names that
could otherwise be considered to be too ambiguous.

Abstract LabelEncoding
-------------------------

We offer a number of different encodings that can best be
described in terms of two orthogonal properties. The first
property is the number of classes it represents, and the
second property is the number of array dimensions it operates on.

.. type:: LabelEncoding{T,K,M}

   Abstract super-type of all label encodings. Mainly intended for
   dispatch. As such this type is not exported.
   It defines three type-parameters that are useful to divide the
   different encodings into groups.

   .. attribute:: T

      The label-type of the encoding, which specifies which concrete
      type all label of that particular encoding have.

   .. attribute:: K

      The number of labels that the label-encoding can deal with.
      So for binary encodings this will be the constant ``2``

   .. attribute:: M

      The number of array dimensions that the encoding works
      with.  For most encodings this will be ``1``, meaning that
      a target array of that encoding is expected to be some
      vector. In contrast to this does the encoding
      :class:`LabelEnc.OneOfK` defined ``M=2``, because it
      represents the target array as a matrix.


TrueFalse
-----------

.. type:: LabelEnc.TrueFalse

   Denotes the classes as boolean values, for which ``true``
   corresponds to the positive class, and ``false`` to the
   negative class.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.TrueFalse)
      MLLabelUtils.LabelEncoding{Bool,2,1}

   It belongs to the family of binary vector-based encodings, and
   as such represents the targets as a vector that is using only
   two distinct values for its elements. That implies that it is
   per defintion always binary and as such the number of labels
   can be inferred at compile time.

   .. code-block:: jlcon

      julia> nlabel(LabelEnc.TrueFalse)
      2

.. function:: TrueFalse() -> LabelEnc.TrueFalse

   Returns the singleton that represents the encoding.
   All information about the encoding is already contained
   withing the type. As such there is no need to specify
   additional parameters.

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [false, true, true, false];

   julia> labelenc(true_targets)
   MLLabelUtils.LabelEnc.TrueFalse()

   julia> label(LabelEnc.TrueFalse())
   2-element Array{Bool,1}:
     true
    false

   julia> nlabel(LabelEnc.TrueFalse())
   2

ZeroOne
-----------

.. type:: LabelEnc.ZeroOne

   Denotes the classes as numeric values, for which ``1``
   corresponds to the positive class, and ``0`` to the
   negative class. This type of encoding is often used
   when the predictions denote a probabilty.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.ZeroOne)
      MLLabelUtils.LabelEncoding{T<:Number,2,1}

   It belongs to the family of binary numeric vector-based
   encodings, and as such represents the targets as a vector that
   is using only two distinct values for its elements. In fact,
   it is by definition always binary and as such the number of
   labels can be inferred at compile time.

   .. code-block:: jlcon

      julia> nlabel(LabelEnc.ZeroOne)
      2

   This type also comes with support for classification (see
   :func:`classify`).
   It assumes that the raw predictions (often called
   :math:`\hat{y}`) are in the closed interval :math:`[0, 1]` and
   represent something resembling a probabilty (or some degree of
   certainty) that the observation is of the positive class. That
   means that in order to classify a raw prediction to either
   positive or negative, one needs to decide on a "threshold"
   parameter, which determines at which degree of certainty a
   prediction is "good enough" to classify as positive.

   .. attribute:: threshold

      A real number between 0 and 1 that defines the "cutoff"
      point for classification. Any prediction less than this
      value will be classified as negative and any prediction
      equal to or greater than this value will be classified as
      a positive prediction.


.. function:: ZeroOne([labeltype], [threshold]) -> LabelEnc.ZeroOne

   Creates a new label-encoding of the :class:`LabelEnc.ZeroOne`
   family.

   :param DataType labeltype: The type that should be used to
                              represent the labels. Has to be a
                              subtype of ``Number``.
                              Defaults to ``Float64``.

   :param Number threshold: The classification threshold that
                            should be used in :func:`classify`.
                            Defaults to ``0.5``.

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> LabelEnc.ZeroOne(Int, 0.3) # threshold = 0.3
   MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.3)

   julia> true_targets = [0, 1, 1, 0];

   julia> labelenc(true_targets)
   MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.5)

   julia> label(LabelEnc.ZeroOne())
   2-element Array{Float64,1}:
    1.0
    0.0

   julia> nlabel(LabelEnc.ZeroOne())
   2

MarginBased
------------

.. type:: LabelEnc.MarginBased

   Denotes the classes as numeric values, for which ``1``
   corresponds to the positive class, and ``-1`` to the
   negative class. This type of encoding is very prominent
   for margin-based classifier, in particular SVMs.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.MarginBased)
      MLLabelUtils.LabelEncoding{T<:Number,2,1}

   It belongs to the family of binary numeric vector-based
   encodings, and as such represents the targets as a vector that
   is using only two distinct values for its elements. In fact,
   it is by definition always binary and as such the number of
   labels can be inferred at compile time.

   .. code-block:: jlcon

      julia> nlabel(LabelEnc.MarginBased)
      2

   This type also comes with support for classification (see
   :func:`classify`).
   It expects the raw predictions to be real numbers of arbitrary
   value. The decision boundary between classifying into a
   negative or a positive label is predefined at zero. More
   precisely a raw prediction greater than or equal to zero is
   considered a positive prediction, while any strictly negative
   raw prediction is considered a negative prediction.

.. function:: MarginBased([labeltype]) -> LabelEnc.MarginBased

   Creates a new label-encoding of the
   :class:`LabelEnc.MarginBased` family.

   :param DataType labeltype: The type that should be used to
                              represent the labels. Has to be a
                              subtype of ``Number``.
                              Defaults to ``Float64``.

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [-1, 1, 1, -1];

   julia> labelenc(true_targets)
   MLLabelUtils.LabelEnc.MarginBased{Int64}()

   julia> label(LabelEnc.MarginBased())
   2-element Array{Float64,1}:
     1.0
    -1.0

   julia> nlabel(LabelEnc.MarginBased())
   2

OneVsRest
------------

.. type:: LabelEnc.OneVsRest

   This is a special type of binary encoding that allows to
   convert a multi-class problem into a binary one. It does so by
   only "caring" about what the positive label is, and treating
   everything that is not equal to it as negative.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.OneVsRest)
      MLLabelUtils.LabelEncoding{T,2,1}

   It belongs to the family of binary vector-based encodings.
   It is by definition always binary and as such the number of
   labels can be inferred at compile time.

   .. code-block:: jlcon

      julia> nlabel(LabelEnc.OneVsRest)
      2

   While this encoding only uses to positive label to assert
   class membership, it still needs to have a placeholder-value
   of the same type for a negative label in order for
   :func:`convertlabel` to work.

   .. attribute:: poslabel

      The value that will be used to represent the positive
      class. This value will be used to determine if a given
      value is positive (if it is equal) or negative.

   .. attribute:: neglabel

      Placeholder to represent the negative class. This value
      will not be used to determine membership, but simply to
      impute a reasonable value when converting to such an
      encoding.

.. function:: OneVsRest(poslabel, [neglabel]) -> LabelEnc.OneVsRest

   Creates a new label-encoding of the one-vs-rest family.  While
   both a positive and a negative label have to be known to the
   encoding, only the positive label is used for comparision and
   asserting class membership. Note that both parameter have to
   be of the same type.

   :param Any poslabel: The label of interest.

   :param Any neglabel: The negative label. It is optional for
                        the common types, such as symbol, number,
                        or string. For label-types other than
                        that it has to be provided explicitly.

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [:yes, :no, :maybe, :yes];

   julia> convertlabel(LabelEnc.OneVsRest(:yes), true_targets)
   4-element Array{Symbol,1}:
    :yes
    :not_yes
    :not_yes
    :yes

   julia> convertlabel(LabelEnc.MarginBased, true_targets, LabelEnc.OneVsRest(:yes))
   4-element Array{Float64,1}:
     1.0
    -1.0
    -1.0
     1.0

   julia> label(LabelEnc.OneVsRest(:yes))
   2-element Array{Symbol,1}:
    :yes
    :not_yes

   julia> nlabel(LabelEnc.OneVsRest(:yes))
   2

Indices
------------

.. type:: LabelEnc.Indices

   A multiclass encoding that uses the integer numbers in
   :math:`\{1, 2, ..., K\}` as label to denote the classes.
   While these "indices" are integers in terms of their values,
   they don't need to be ``Int`` as a type.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.Indices)
      MLLabelUtils.LabelEncoding{T<:Number,K,1}

   It belongs to the family of numeric vector-based encodings and
   can encode any number of classes. As such the number of labels
   ``K`` is a free type-parameter.
   It is considered a binary encoding if and only if ``K = 2``

.. function:: Indices([labeltype], k) -> LabelEnc.Incides

   Creates a new label-encoding of the
   :class:`LabelEnc.Indices` family.

   :param DataType labeltype: The type that should be used to
                              represent the labels. Has to be a
                              subtype of ``Number``.
                              Defaults to ``Int``.

   :param Int k: The number of classes that the concoding
                 should represent. This parameter can be
                 specified as an ``Int`` or in type-stable manner
                 as ``Val{k}``

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [1, 2, 1, 3, 1, 2];

   julia> labelenc(true_targets)
   MLLabelUtils.LabelEnc.Indices{Int64,3}()

   julia> label(LabelEnc.Indices(3))
   3-element Array{Int64,1}:
    1
    2
    3

   julia> label(LabelEnc.Indices(Float32,4))
   4-element Array{Float32,1}:
    1.0
    2.0
    3.0
    4.0

   julia> nlabel(LabelEnc.Indices(Val{5})) # type-stable
   5

OneOfK
-------------

.. type:: LabelEnc.OneOfK

   A multi-class encoding that uses one of the two matrix
   dimensions to denote the label. More precisely other words it
   uses an indicator-encoding to explicitly state what class an
   observation represents and what it does not represent, by
   only setting one element of each observation to ``1`` and the
   rest to ``0``

   .. code-block:: jlcon

      julia> supertype(LabelEnc.OneOfK)
      MLLabelUtils.LabelEncoding{T<:Number,K,2}

   It belongs to the family of numeric matrix-based encodings and
   can encode any number of classes. As such the number of labels
   ``K`` is a free type-parameter.
   It is considered a binary encoding if and only if ``K = 2``

.. function:: OneOfK([labeltype], k) -> LabelEnc.OneOfK

   Creates a new label-encoding of the matrix-based
   :class:`LabelEnc.OneOfK` family.

   :param DataType labeltype: The type that should be used to
                              represent the labels. Has to be a
                              subtype of ``Number``.
                              Defaults to ``Int``.

   :param Int k: The number of classes that the concoding
                 should represent. This parameter can be
                 specified as an ``Int`` or in type-stable manner
                 as ``Val{k}``

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [0 1 0 0; 1 0 1 0; 0 0 0 1]
   3×4 Array{Int64,2}:
    0  1  0  0
    1  0  1  0
    0  0  0  1

   julia> labelenc(true_targets)
   MLLabelUtils.LabelEnc.OneOfK{Int64,3}()

   julia> label(LabelEnc.OneOfK(Float32, 4)) # returns the indices
   4-element Array{Int64,1}:
    1
    2
    3
    4

   julia> ind2label(3, LabelEnc.OneOfK(Float32, 4))
   4-element Array{Float32,1}:
    0.0
    0.0
    1.0
    0.0

   julia> nlabel(LabelEnc.OneOfK(Val{4}))
   4

NativeLabels
-------------

.. type:: LabelEnc.NativeLabels

   A multi-class encoding that can use any abritrary values to
   represent any number of labels. It does so by mapping each
   label-index to a class label. The class labels can be of
   arbitrary type as long as the type is consistent for all
   labels. Furthermore, all labels have to be specified
   explicitly.

   .. code-block:: jlcon

      julia> supertype(LabelEnc.NativeLabels)
      MLLabelUtils.LabelEncoding{T,K,1}

   It belongs to the family of vector-based encodings that can
   encode any number of classes. As such the number of labels
   ``K`` is a free type-parameter. It is considered a binary
   encoding if and only if ``k = 2``

   .. attribute:: label

      A vector that contains all the used labels in their defined
      order. If it only contains two values, then the first value
      will be interpreted as the positive label and the second
      value as the negative label.

   .. attribute:: invlabel

      A Dict that maps each label to their index in the vector
      `label`. This map is used for fast lookup and generated
      automatically.

.. function:: NativeLabels(label, [k]) -> LabelEnc.NativeLabels

   Creates a new vector-based label-encoding for the given
   values in `label`. The values in `label` are expected to be
   distinct.

   :param Vector label: The label that the encoding should use in
                        their intended order

   :param DataType k: The number of labels in `label`. This
                      paramater is optional and will be computed
                      from `label` if omited. However, if the
                      number of labels is known at compile time
                      this parmater can be provided using
                      ``Val{k}``

For more information on how to use such an encoding, please look
at the corresponding parts of the documentation.

.. code-block:: jlcon

   julia> true_targets = [:a, :b, :a, :c, :b, :a];

   julia> le = labelenc(true_targets)
   MLLabelUtils.LabelEnc.NativeLabels{Symbol,3}(Symbol[:a,:b,:c],Dict(:c=>3,:a=>1,:b=>2))

   julia> label(le)
   3-element Array{Symbol,1}:
    :a
    :b
    :c

   julia> nlabel(le)
   3

   julia> LabelEnc.NativeLabels([:yes, :no, :maybe], Val{3}) # type inferrable
   MLLabelUtils.LabelEnc.NativeLabels{Symbol,3}(Symbol[:yes,:no,:maybe],Dict(:yes=>1,:maybe=>3,:no=>2))

FuzzyBinary
-------------

.. type:: LabelEnc.FuzzyBinary

   A vector-based binary label interpretation without a specific
   labeltype. It is primarily intended for fuzzy comparision of
   binary true targets and predicted targets.
   It basically assumes that the encoding is either `TrueFalse`,
   `ZeroOne`, or `MarginBased` by treating all non-negative values
   as positive outputs.

