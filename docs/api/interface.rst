Working with Encodings
========================

Now that we have an understanding of how to extract the label-related
information from our targets, let us consider how to instantiate (or
infer) a label-encoding, and what we can do with it once we have one.
In particular, these encodings will enable us to transform the targets
from one representation into another without losing the ability to
convert them back afterwards.

Inferring the Encoding
--------------------------------------

In many cases we may not want to just simply assume or guess the
particular encoding that some user-provided targets are in.
Instead we would rather let the targets themself inform us what
encoding they are using.
To that end we provide the function :func:`labelenc`.

.. function:: labelenc(vec) -> LabelEncoding

   Tries to determine the most approriate label-encoding to describe
   the given vector `vec`, based on the result of ``label(vec)``.
   Note that in most cases this function is not typestable, because
   the eltype of `vec` is usually not enough to infer the encoding
   or number of labels reliably.

   :param AbstractVector vec: The classification targets in vector form.

   :return: The label-encoding that is deemed most approriate
            to describe the values found in `vec`.

.. code-block:: jlcon

   julia> labelenc([:yes,:no,:no,:maybe,:yes,:no])
   MLLabelUtils.LabelEnc.NativeLabels{Symbol,3}(Symbol[:yes,:no,:maybe],Dict(:yes=>1,:maybe=>3,:no=>2))

   julia> labelenc([-1,1,1,-1,1])
   MLLabelUtils.LabelEnc.MarginBased{Int64}()

   julia> labelenc(UInt8[0,1,1,0,1])
   MLLabelUtils.LabelEnc.ZeroOne{UInt8,Float64}(0.5)

   julia> labelenc([false,true,true,false,true])
   MLLabelUtils.LabelEnc.TrueFalse()

For matrices we allow an additional (but optional) parameter with
which the user can specify the array dimension that denotes the
observations.

.. function:: labelenc(mat, [obsdim]) -> LabelEncoding

   Computes the concrete matrix-based label-encoding that is used,
   by determining the size of the matrix for the dimension that is
   **not** used for denoting the observations.

   :param AbstractMatrix mat: An numeric matrix that is assumed to be in
                              the form of a one-hot encoding or similar.

   :param ObsDimension obsdim: Optional. Denotes which of the two array
                               dimensions of `mat` denotes the
                               observations. It can be specified as
                               a type-stable positional argument or
                               a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: The label-encoding that is deemed most approriate
            to describe the structure and values found in `mat`.

.. code-block:: jlcon

   julia> labelenc([0 1 0 0; 1 0 1 0; 0 0 0 1])
   MLLabelUtils.LabelEnc.OneOfK{Int64,3}()

   julia> labelenc(Float32[0 1; 1 0; 0 1; 0 1], obsdim = 1)
   MLLabelUtils.LabelEnc.OneOfK{Float32,2}()

Asserting Assumptions
--------------------------------------

When writing a function that requires the classification targets to
be in a specific encoding (for example :math:`\{1, -1\}` in the case
of SVMs), it can be useful to check if the user-provided targets are
already in the appropriate encoding, of if they first have to be
converted.
To check if the targets are of a specific encoding, or family of
encodings, we provide the function :func:`islabelenc`.

.. function:: islabelenc(vec, encoding) -> Bool

   Checks if the given values in `vec` can be described as being
   produced by the given `encoding`. This function does not only
   check the values but also for the correct type.
   Furthermore it also checks if the total number of labels is
   appropriate for what the `encoding` expects it to be.

   :param AbstractVector vec: The classification targets in vector form.

   :param LabelEncoding encoding: A concrete instance of a
                                  label-encoding that one wants to work
                                  with.

   :return: True, if both the values in `vec` as well as their types
            are consistent with the given `encoding`.

.. code-block:: jlcon

   julia> islabelenc([0,1,1,0,1], LabelEnc.ZeroOne(Int))
   true

   julia> islabelenc([0,1,1,0,1], LabelEnc.ZeroOne(Float64))
   false

   julia> islabelenc([0,1,1,0,1], LabelEnc.MarginBased(Int))
   false

   julia> islabelenc(Int8[-1,1,1,-1,1], LabelEnc.MarginBased(Int8))
   true

   julia> islabelenc(Int8[-1,1,1,-1,1], LabelEnc.MarginBased(Int16))
   false

   julia> islabelenc([2,1,2,3,1], LabelEnc.Indices(Int,3))
   true

   julia> islabelenc([2,1,2,3,1], LabelEnc.Indices(Int,4)) # it allows missing labels
   true

   julia> islabelenc([2,1,2,3,1], LabelEnc.Indices(Int,2)) # more labels than expected
   false

Similar to :func:`label` we treat matrices in a special way to
account for the fact that information about the number of labels
is contained in the size of a matrix and not its values.
Additionally the user has the freedom to choose which matrix
dimension denotes the observations.

.. function:: islabelenc(mat, encoding, [obsdim]) -> Bool

   Checks if the values and the structure of the given matrix `mat`
   is consistent with the specified `encoding`.
   This functions also checks for the correct type and dimensions.

   :param AbstractMatrix mat: The classification targets in matrix form.

   :param LabelEncoding encoding: A concrete instance of a matrix-based
                                  label-encoding that one wants to work
                                  with.

   :param ObsDimension obsdim: Optional. Denotes which of the two array
                               dimensions of `mat` denotes the
                               observations. It can be specified as
                               a type-stable positional argument or
                               a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: True, if the values in `mat`, its eltype, and the
            shape of `mat` is consistent with the given `encoding`.

.. code-block:: jlcon

   julia> islabelenc([0 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK(Int,3))
   true

   julia> islabelenc([0 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK(Int8,3))
   false

   julia> islabelenc([1 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK(Int,3)) # matrix is not one-hot
   false

   julia> islabelenc([0 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK(Int,4)) # only 3 rows
   false

   julia> islabelenc([0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK(Int,2), obsdim = 1)
   true

   julia> islabelenc(UInt8[0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK(Int,2), obsdim = 1)
   false

   julia> islabelenc(UInt8[0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK(UInt8,2), obsdim = 1)
   true

So far :func:`islabelenc` was very restrictive concerning the element
types of the given target array. In many cases, however, we may not
actually care too much about the concrete numeric type but only if
the encoding-scheme itself is followed. In fact we usually don't
want to be restrictive about concrete types at all, since we
have Julia's multiple-dispatch system to take care of that later on.
In other words we may be more interested in asserting if the labels
of the given targets belong to a **family** of possible label-encodings.

.. function:: islabelenc(vec, type) -> Bool

   Checks is the given values in `vec` can be described as being
   produced by any possible instance of the given `type`.
   In other word this function checks if the labels in `vec` can
   be described as being consistent with the family of label-encodings
   specified by `type`.
   This means that the check is much more tolerant concerning the
   eltype and the total number of labels, since some families of
   encodings are approriate for any number of labels.

   :param AbstractVector vec: The classification targets in vector form.

   :param DataType type: Any subtype of :class:`LabelEncoding{T,K,1}`

   :return: True, if the values in `vec` are consistent with the
            given family of encodings specified by `type`.

.. code-block:: jlcon

   julia> islabelenc([0,1,1,0,1], LabelEnc.ZeroOne)
   true

   julia> islabelenc(UInt8[0,1,1,0,1], LabelEnc.ZeroOne)
   true

   julia> islabelenc([0,1,1,0,1], LabelEnc.MarginBased)
   false

   julia> islabelenc(Float32[-1,1,1,-1,1], LabelEnc.MarginBased)
   true

   julia> islabelenc(Int8[-1,1,1,-1,1], LabelEnc.MarginBased)
   true

   julia> islabelenc([2,1,2,3,1], LabelEnc.Indices)
   true

   julia> islabelenc(Int8[2,1,2,3,1], LabelEnc.Indices)
   true

   julia> islabelenc(Int8[2,1,2,3,1], LabelEnc.Indices{Int}) # restrict type but not nlabels
   false

We again provide a special version for matrices.

.. function:: islabelenc(mat, type, [obsdim]) -> Bool

   Checks is the values and the structure of the given matrix `mat`
   can be described as being produced by any possible instance of
   the given `type`.
   This means that the check is much more tolerant concerning the
   eltype and the size of the matrix, since some families of
   encodings are approriate for any number of labels.

   :param AbstractMatrix mat: The classification targets in matrix form.

   :param DataType type: Any subtype of :class:`LabelEncoding{T,K,2}`

   :param ObsDimension obsdim: Optional. Denotes which of the two array
                               dimensions of `mat` denotes the
                               observations. It can be specified as
                               a type-stable positional argument or
                               a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: True, if the values in `mat` are consistent with the
            given family of encodings specified by `type`.

.. code-block:: jlcon

   julia> islabelenc([0 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK)
   true

   julia> islabelenc(Int8[0 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK)
   true

   julia> islabelenc([1 1 0 0; 1 0 1 0; 0 0 0 1], LabelEnc.OneOfK) # matrix is not one-hot
   false

   julia> islabelenc([0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK, obsdim = 1)
   true

   julia> islabelenc(UInt8[0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK, obsdim = 1)
   true

   julia> islabelenc(UInt8[0 1; 1 0; 0 1; 0 1], LabelEnc.OneOfK{Int32}, obsdim = 1) # restrict type but not nlabels
   false

Properties of an Encoding
--------------------------------------

Once we have an instance of some label-encoding, we can compute
a number of useful properties about it.
For example we can query all the labels that an encoding uses
to represent the classes.

.. function:: label(encoding) -> Vector

   Returns all the labels that a specific encoding uses in their
   approriate order.

   :param LabelEncoding encoding: The specific label-encoding.

   :return: The unique labels in the form of a vector. In the case
            of two labels, the first element will represent the
            positive label and the second element the negative label
            respectively.

.. code-block:: jlcon

   julia> label(LabelEnc.ZeroOne(UInt8))
   2-element Array{UInt8,1}:
    0x01
    0x00

   julia> label(LabelEnc.MarginBased())
   2-element Array{Float64,1}:
     1.0
    -1.0

   julia> label(LabelEnc.Indices(Float32,5))
   5-element Array{Float32,1}:
    1.0
    2.0
    3.0
    4.0
    5.0

For convenience one can also just query for the label that
corresponds to the positive class or the negative class respectively.
These helper functions are only defined for binary label-encoding and
will throw an ``MethodError`` for multi-class encodings.

.. function:: poslabel(encoding)

   If the encoding is binary it will return the positive label of it.
   The function will throw an error otherwise.

   :param LabelEncoding encoding: The specific label-encoding.

   :return: The value representing the positive label of the given
            `encoding` in the approriate type.

.. code-block:: jlcon

   julia> poslabel(LabelEnc.ZeroOne(UInt8))
   0x01

   julia> poslabel(LabelEnc.MarginBased())
   1.0

   julia> poslabel(LabelEnc.Indices(Float32,2))
   1.0f0

   julia> poslabel(LabelEnc.Indices(Float32,5))
   ERROR: MethodError: no method matching poslabel(::MLLabelUtils.LabelEnc.Indices{Float32,5})

.. function:: neglabel(encoding)

   If the encoding is binary it will return the negative label of it.
   The function will throw an error otherwise.

   :param LabelEncoding encoding: The specific label-encoding.

   :return: The value representing the negative label of the given
            `encoding` in the approriate type.

.. code-block:: jlcon

   julia> neglabel(LabelEnc.ZeroOne(UInt8))
   0x00

   julia> neglabel(LabelEnc.MarginBased())
   -1.0

   julia> neglabel(LabelEnc.Indices(Float32,2))
   2.0f0

   julia> neglabel(LabelEnc.Indices(Float32,5))
   ERROR: MethodError: no method matching neglabel(::MLLabelUtils.LabelEnc.Indices{Float32,5})

We can also query the number of labels that a concrete encoding uses.
In other words we can query the number of classes the given
label-encoding is able to represent.

.. function:: nlabel(encoding) -> Int

   Returns the number of labels that a specific encoding uses.

   :param LabelEncoding encoding: The specific label-encoding.

.. code-block:: jlcon

   julia> nlabel(LabelEnc.ZeroOne(UInt8))
   2

   julia> nlabel(LabelEnc.NativeLabels([:a,:b,:c]))
   3

More interestingly, we can infer the number of labels for a family
of encodings. This allows for some compile time decisions, but only
work for some types of encodings (i.e. binary).

.. function:: nlabel(type) -> Int

   Returns the number of labels that the family of encodings `type`
   can describe.
   Note that this function will fail if the number of labels can
   not be inferred from the given type.

   :param DataType type: Some subtype of :class:`LabelEncoding{T,K,M}`
                         with a fixed ``K``

   :return: The type-parameter ``K`` of `type`.

.. code-block:: jlcon

   julia> nlabel(LabelEnc.ZeroOne)
   2

   julia> nlabel(LabelEnc.NativeLabels)
   ERROR: ArgumentError: number of labels could not be inferred for the given type

We can also query a family of encodings for their label-type.
In this case we decided to not throw an error if the type can not
be inferred but instead return the most specific abstract type.

.. function:: labeltype(type) -> DataType

   Determine the type of the labels represented by the given
   family of label-encoding. If the type can not be inferred than
   ``Any`` is returned.

   :param DataType type: Some subtype of :class:`LabelEncoding{T,K,M}`

   :return: The type-parameter ``T`` of `type` if specified,
            or the most specific abstract type otherwise.

.. code-block:: jlcon

   julia> labeltype(LabelEnc.TrueFalse)
   Bool

   julia> labeltype(LabelEnc.ZeroOne{Int})
   Int64

   julia> labeltype(LabelEnc.ZeroOne)
   Number

   julia> labeltype(LabelEnc.NativeLabels)
   Any

Converting to/from Indices
--------------------------------------

As stated before, the order of the of :func:`label` matters.
In a binary setting, for example, the first label is interpreted as
the positive class and the second label as the negative class.
This is simply the arbitrary convention that we follow.
That said, even in a multi-class setting it is important to be
consistent with the ordering. This is crucial in order to make sure
that converting to a different encoding and then converting back
yields the original values.

Every encoding understands the concept of a **label-index**,
which is a unique representation of a class that all encodings share.
For example the positive label of a binary label-encoding always
has the label-index ``1`` and the negative ``2`` respectively.

To convert a label-index into the label that a specific encoding uses
to represent the underlying class we provide the function
:func:`ind2label`.

.. function:: ind2label(index, encoding)

   Converts the given `index` into the corresponding label defined
   by the `encoding`. Note that in the binary case, ``index = 1``
   represents the positive label and ``index = 2`` the negative label.

   This function supports broadcasting.

   :param Int index: Index of the desired label. This variable can
                     be specified either as an ``Int`` or as a ``Val``.
                     Note that indices are one-based.

   :param LabelEncoding encoding: The encoding one wants to get the
                                  label from.

   :return: The label of the specified `index` for the specified
            `encoding`.

.. code-block:: jlcon

   julia> ind2label(1, LabelEnc.MarginBased(Float32))
   1.0f0

   julia> ind2label(Val{1}, LabelEnc.MarginBased(Float32))
   1.0f0

   julia> ind2label(2, LabelEnc.MarginBased(Float32))
   -1.0f0

   julia> ind2label(3, LabelEnc.OneOfK(Int8,4))
   4-element Array{Int8,1}:
    0
    0
    1
    0

   julia> ind2label(3, LabelEnc.NativeLabels([:a,:b,:c,:d]))
   :c

   julia> ind2label.([1,2,2,1], LabelEnc.ZeroOne(UInt8)) # broadcast support
   4-element Array{UInt8,1}:
    0x01
    0x00
    0x00
    0x01

We also provide inverse function for converting a label of a specific
encoding into the corresponding label-index.
Note that this function does not check if the given label is of the
expected type, but simply that it is of the appropriate value.

.. function:: label2ind(label, encoding) -> Int

   Converts the given `label` into the corresponding index defined
   by the `encoding`. Note that in the binary case, the positive label
   will result in the index `1` and the negative label in the index
   `2` respectively.

   This function supports broadcasting.

   :param Any label: A label in the format familiar to the `encoding`.

   :param LabelEncoding encoding: The encoding to compute the
                                  label-index with.

   :return: The index of the specified `label` for the specified
            `encoding`.

.. code-block:: jlcon

   julia> label2ind(1.0, LabelEnc.MarginBased())
   1

   julia> label2ind(-1.0, LabelEnc.MarginBased())
   2

   julia> label2ind([0,0,1,0], LabelEnc.OneOfK(4))
   3

   julia> label2ind(:c, LabelEnc.NativeLabels([:a,:b,:c,:d]))
   3

   julia> label2ind.([1,0,0,1], LabelEnc.ZeroOne()) # broadcast support
   4-element Array{Int64,1}:
    1
    2
    2
    1

Converting between Encodings
------------------------------

In the case that the given targets are not in the encoding that your
algorithm expects them to be in, you may want to convert them into the
format you require.
For that purpose we expose the function :func:`convertlabel`.

.. function:: convertlabel(dst_encoding, src_label, src_encoding)

   Converts the given input label `src_label` from `src_encoding`
   into the corresponding label described by the desired output
   encoding `dst_encoding`.

   Note that both encodings are expected to be vector-based, meaning
   that this method does not work for :class:`LabelEnc.OneOfK`.
   It does, however, support broadcasting.

   :param LabelEncoding dst_encoding: The vector-based label-encoding
                                      that should be used to produce
                                      the output label.

   :param Any src_label: The input label one wants to convert. It is
                         expected to be consistent with `src_encoding`.

   :param LabelEncoding src_encoding: A vector-based label-encoding
                                      that is assumed to have produced
                                      the given `src_label`.

   :return: The label from `dst_encoding` that corresponds to
            `src_label` in `src_encoding`

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneOfK(2), -1, LabelEnc.MarginBased()) # OneOfK is not vector-based
   ERROR: MethodError: no method matching [...]

   julia> convertlabel(LabelEnc.NativeLabels([:a,:b,:c,:d]), 3, LabelEnc.Indices(4))
   :c

   julia> convertlabel(LabelEnc.ZeroOne(), :yes, LabelEnc.NativeLabels([:yes,:no]))
   1.0

   julia> convertlabel(LabelEnc.ZeroOne(), :no, LabelEnc.NativeLabels([:yes,:no]))
   0.0

   julia> convertlabel(LabelEnc.MarginBased(Int), 0, LabelEnc.ZeroOne())
   -1

   julia> convertlabel(LabelEnc.NativeLabels([:a,:b]), -1, LabelEnc.MarginBased())
   :b

   julia> convertlabel.(LabelEnc.NativeLabels([:a,:b]), [-1,1,1,-1], LabelEnc.MarginBased()) # broadcast support
   4-element Array{Symbol,1}:
    :b
    :a
    :a
    :b

Aside from the one broadcast-able method that is implemented for
converting single labels, we provide a range of methods that work on
whole arrays.
These are more flexible because by having an array as input these
methods have more information available to make reasonable
decisions. As a consequence of that can we consider the
"source-encoding" parameter optional, because these methods can
now make use of :func:`labelenc` internally to infer it
automatically.

.. function:: convertlabel(dst_encoding, arr, [src_encoding], [obsdim])

   Converts the given array `arr` from the `src_encoding` into the
   `dst_encoding`. If `src_encoding` is not specified it will be
   inferred automaticaly using the function :func:`labelenc`.
   This should not negatively influence type-inference.

   Note that both encodings should have the same number of labels,
   or a MethodError will be thrown in most cases.

   :param LabelEncoding dst_encoding: The desired output format.

   :param AbstractArray arr: The input targets that should be
                             converted into the encoding specified
                             by `dst_encoding`.

   :param LabelEncoding src_encoding: The input encoding that `arr`
                                      is expected to be in.

   :param ObsDimension obsdim: Optional. Only possible if one of the
                               two encodings is a matrix-based encoding.
                               Defines which of the two array
                               dimensions denotes the observations.
                               It can be specified as a type-stable
                               positional argument or a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: A converted version of `arr` using the specified
            output encoding `dst_encoding`.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.NativeLabels([:yes,:no]), [-1,1,-1,1,1,-1])
   6-element Array{Symbol,1}:
    :no
    :yes
    :no
    :yes
    :yes
    :no

   julia> convertlabel(LabelEnc.OneOfK(Float32,2), [-1,1,-1,1,1,-1])
   2×6 Array{Float32,2}:
    0.0  1.0  0.0  1.0  1.0  0.0
    1.0  0.0  1.0  0.0  0.0  1.0

   julia> convertlabel(LabelEnc.TrueFalse(), [-1,1,-1,1,1,-1])
   6-element Array{Bool,1}:
    false
     true
    false
     true
     true
    false

   julia> convertlabel(LabelEnc.Indices(3), [:no,:maybe,:yes,:no], LabelEnc.NativeLabels([:yes,:maybe,:no]))
   4-element Array{Int64,1}:
    3
    2
    1
    3

It may be interesting to point out explicitly that we provide
special treatment for :class:`LabelEnc.OneVsRest` to conveniently
convert a multi-class problem into a two-class problem.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneVsRest(:yes), [:yes,:no,:no,:maybe,:yes,:yes])
   6-element Array{Symbol,1}:
    :yes
    :not_yes
    :not_yes
    :not_yes
    :yes
    :yes

   julia> convertlabel(LabelEnc.ZeroOne(Float64), [:yes,:no,:no,:maybe,:yes,:yes], LabelEnc.OneVsRest(:yes))
   6-element Array{Float64,1}:
    1.0
    0.0
    0.0
    0.0
    1.0
    1.0


We also allow a more concise way to specify that your are using a
:class:`LabelEnc.NativeLabels` encoding by just passing the
label-vector directly, that you would normally pass to its
constructor.

.. code-block:: jlcon

   julia> convertlabel([:yes,:no], [-1,1,-1,1,1,-1])
   6-element Array{Symbol,1}:
    :no
    :yes
    :no
    :yes
    :yes
    :no

   julia> convertlabel(LabelEnc.Indices(3), [:no,:maybe,:yes,:no], [:yes,:maybe,:no])
   4-element Array{Int64,1}:
    3
    2
    1
    3


In many cases it can be inconvenient that one has to explicitly
specify the label-type and number of labels for the desired
output-encoding. To that end we also allow the output-encoding
to be specified in terms of an encoding-family (i.e. as ``DataType``).

.. function:: convertlabel(dst_family, arr, [src_encoding], [obsdim])

   Converts the given array `arr` from the `src_encoding` into
   some concrete label-encoding that is a subtype of `dst_family`.
   This way the method tries to preserve the eltype of `arr`
   if it is numeric. Furthermore, the concrete number of labels
   need not be specified explicitly, but will instead be inferred
   from `src_encoding`.

   If `src_encoding` is not specified it will be
   inferred automaticaly using the function :func:`labelenc`.
   This should not negatively influence type-inference.

   :param DataType dst_family: Any subtype of
                               :class:`LabelEncoding{T,K,M}`.
                               It denotes the desired family of
                               label-encodings that one wants
                               the return value to be in.

   :param AbstractArray arr: The input targets that should be
                             converted into some encoding specified
                             by the type `dst_family`.

   :param LabelEncoding src_encoding: The input encoding that `arr`
                                      is expected to be in.

   :param ObsDimension obsdim: Optional. Only possible if one of the
                               two encodings is a matrix-based encoding.
                               Defines which of the two array
                               dimensions denotes the observations.
                               It can be specified as a type-stable
                               positional argument or a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: A converted version of `arr` using a label-encoding
            that is member of the encoding-family `dst_family`.

.. code-block:: jlcon

   julia> convertlabel(LabelEnc.OneOfK, Int8[-1,1,-1,1,1,-1])
   2×6 Array{Int8,2}:
    0  1  0  1  1  0
    1  0  1  0  0  1

   julia> convertlabel(LabelEnc.OneOfK{Float32}, Int8[-1,1,-1,1,1,-1], obsdim = 1)
   6×2 Array{Float32,2}:
    0.0  1.0
    1.0  0.0
    0.0  1.0
    1.0  0.0
    1.0  0.0
    0.0  1.0

   julia> convertlabel(LabelEnc.TrueFalse, [-1,1,-1,1,1,-1])
   6-element Array{Bool,1}:
    false
     true
    false
     true
     true
    false

   julia> convertlabel(LabelEnc.Indices, [:no,:maybe,:yes,:no], LabelEnc.NativeLabels([:yes,:maybe,:no]))
   4-element Array{Int64,1}:
    3
    2
    1
    3

For vector-based encodings (which means all except
:class:`LabelEnc.OneOfK`), we provide a lazy version of
:func:`convertlabel` that does not allocate a new array for the
outputs, but instead creates a
`MappedArray <https://github.com/JuliaArrays/MappedArrays.jl>`_
into the original targets.

.. function:: convertlabelview(dst_encoding, vec, [src_encoding])

   Creates a ``MappedArray`` that provides a lazy view into `vec`,
   that makes it look like the values are actually of the provided
   output encoding `new_encoding`. This means that the convertion
   happens on the fly when an element of the resulting mapped array
   is accessed.
   This resulting mapped array will even be writeable, unless
   `src_encoding` is :class:`LabelEnc.OneVsRest`.

   Note that both encodings are expected to be vector-based, meaning
   that this method does not work for :class:`LabelEnc.OneOfK`.

   :param LabelEncoding dst_encoding: The desired vector-based output
                                      encoding.

   :param AbstractVector vec: The input targets that one wants to
                              convert using `dst_encoding`.
                              It is expected to be consistent with
                              `src_encoding`.

   :param LabelEncoding src_encoding: A vector-based label-encoding
                                      that is assumed to have produced
                                      the values in `vec`.

   :return: A ``MappedArray`` or ``ReadonlyMappedArray`` that makes
            `vec` look like it is in the encoding specified by
            `new_encoding`

.. code-block:: jlcon

   julia> true_targets = [-1,1,-1,1,1,-1]
   6-element Array{Int64,1}:
    -1
     1
    -1
     1
     1
    -1

   julia> A = convertlabelview(LabelEnc.NativeLabels([:yes,:no]), true_targets)
   6-element MappedArrays.MappedArray{Symbol,1,...}:
    :no
    :yes
    :no
    :yes
    :yes
    :no

   julia> A[2] = :no
   julia> A
   6-element MappedArrays.MappedArray{Symbol,1,...}:
    :no
    :no
    :no
    :yes
    :yes
    :no

   julia> true_targets
   6-element Array{Int64,1}:
    -1
    -1
    -1
     1
     1
    -1

Classifying Predictions
-------------------------

Some encodings come with an implicit interpretation of how the
**raw predictions** of some model (often denoted as :math:`\hat{y}`,
written ``yhat``) should look like and how they can be classified
into a predicted class-label.
For that purpose we provide the function :func:`classify` and its
mutating version :func:`classify!`.

.. function:: classify(yhat, encoding)

   Returns the classified version of `yhat` given the `encoding`.
   That means that if `yhat` can be interpreted as a positive label,
   the positive label of `encoding` is returned.
   If `yhat` can not be interpreted as a positive value then the
   negative label is returned.

   This methods supports broadcasting.

   :param Number yhat: The numeric prediction that should be
                       classified into either the label representing
                       the positive class or the label representing
                       the negative class

   :param LabelEncoding encoding: A concrete instance of a
                                  label-encoding that one wants to
                                  work with.

   :return: The label that the encoding uses to represent the class
            that `yhat` is classified into.

For :class:`LabelEnc.MarginBased` the decision boundary between
classifying into a negative or a positive label is predefined at zero.
More precisely a raw prediction greater than or equal to zero
is considered a positive prediction, while any strictly negative raw
prediction is considered a negative prediction.

.. code-block:: jlcon

   julia> classify(-0.3f0, LabelEnc.MarginBased()) # defaults to Float64
   -1.0

   julia> classify.([-2.3,6.5], LabelEnc.MarginBased(Int))
   2-element Array{Int64,1}:
    -1
     1


For :class:`LabelEnc.ZeroOne` the assumption is that the raw
prediction is in the closed interval :math:`[0, 1]` and represents
a degree of certainty that the observation is of the positive class.
That means that in order to classify a raw prediction to either
positive or negative, one needs to decide on a "threshold" parameter,
which determines at which degree of certainty a prediction is
"good enough" to classify as positive.

.. code-block:: jlcon

   julia> classify(0.3f0, LabelEnc.ZeroOne(0.5)) # defaults to Float64
   0.0

   julia> classify(0.3f0, LabelEnc.ZeroOne(Int,0.2))
   1

   julia> classify.([0.3,0.5], LabelEnc.ZeroOne(Int,0.4))
   2-element Array{Int64,1}:
    0
    1

We recognize that such a probabilistic interpretation of the raw
predicted value is fairly common. So much so that we provide a
convenience method for when one is working under the assumption of
a :class:`LabelEnc.ZeroOne` encoding.

.. function:: classify(yhat, threshold)

   Returns the classified version of `yhat` given the decision margin
   `threshold`. This method assumes that `yhat` denotes a probability
   and will either return ``zero(yhat)`` if `yhat` is below
   `threshold`, or ``one(yhat)`` otherwise.

   This methods supports broadcasting.

   :param Number yhat: The numeric prediction. It is assumed be a
                       value between 0 and 1.

   :param Number threshold: The threshold below which `yhat` will be
                            classified as ``0``.

   :return: The classified version of `yhat` of the same type.

.. code-block:: jlcon

   julia> classify(0.3f0, 0.5)
   0.0f0

   julia> classify(0.3f0, 0.2)
   1.0f0

   julia> classify.([0.3,0.5], 0.4)
   2-element Array{Float64,1}:
    0.0
    1.0

For matrix-based encodings, such as :class:`LabelEnc.OneOfK` we
provide a special method that allows to optionally specify the
dimension of the matrix that denote the observations.

.. function:: classify(yhat, encoding, [obsdim])

   If `yhat` is a vector (i.e. a single observation), this function
   returns the index of the element that has the largest value.
   If `yhat` is a matrix, this function returns a vector of
   indices for each observation in `yhat`.

   :param AbstractArray yhat: The numeric predictions in the form of
                              either a vector or a matrix.

   :param LabelEncoding encoding: A concrete instance of a
                                  matrix-based label-encoding that
                                  one wants to work with.

   :param ObsDimension obsdim: Optional iff `yhat` is a matrix.
                               Denotes which of the two array
                               dimensions of `yhat` denotes the
                               observations. It can be specified as
                               a type-stable positional argument or
                               a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: The classified version of `yhat`. This will either be
            an integer or a vector of indices.

.. code-block:: jlcon

   julia> pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
   3×4 Array{Float64,2}:
    0.1  0.4  0.3  0.2
    0.8  0.3  0.6  0.2
    0.1  0.3  0.1  0.6

   julia> classify(pred_output, LabelEnc.OneOfK(3))
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify(pred_output', LabelEnc.OneOfK(3), obsdim=1) # note the transpose
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK(4)) # single observation
   3

Similar to other functions we expose a version that can be called
with a family of encodings (i.e. a type with free type parameters)
instead of a concrete instance.

.. function:: classify(yhat, type)

   Returns the classified version of `yhat` given the family of
   encodings specified by `type`.
   That means that if `yhat` can be interpreted as a positive label,
   the positive label of that family is returned (and the negative
   otherwise). Furthermore, the type of `yhat` is preserved.

   This method supports broadcasting.

   :param Number yhat: The numeric prediction that should be
                       classified into either the label representing
                       the positive class or the label representing
                       the negative class

   :param DataType type: Any subtype of :class:`LabelEncoding{T,K,1}`

   :return: The classified version of `yhat` of the same type.

.. code-block:: jlcon

   julia> classify(0.3f0, LabelEnc.ZeroOne) # threshold fixed at 0.5
   0.0f0

   julia> classify(0.3, LabelEnc.ZeroOne)
   0.0

   julia> classify(4f0, LabelEnc.MarginBased)
   1.0f0

   julia> classify(-4, LabelEnc.MarginBased)
   -1

.. function:: classify(yhat, type, [obsdim])

   If `yhat` is a vector (i.e. a single observation), this function
   returns the index of the element that has the largest value.
   If `yhat` is a matrix, this function returns a vector of
   indices for each observation in `yhat`.

   :param AbstractArray yhat: The numeric predictions in the form of
                              either a vector or a matrix.

   :param DataType type: Any subtype of :class:`LabelEncoding{T,K,2}`

   :param ObsDimension obsdim: Optional iff `yhat` is a matrix.
                               Denotes which of the two array
                               dimensions of `yhat` denotes the
                               observations. It can be specified as
                               a type-stable positional argument or
                               a smart keyword.
                               Defaults to ``Obsdim.Last()``.
                               see ``?ObsDim`` for more information.

   :return: The classified version of `yhat`. This will either be
            an integer or a vector of indices.

.. code-block:: jlcon

   julia> pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
   3×4 Array{Float64,2}:
    0.1  0.4  0.3  0.2
    0.8  0.3  0.6  0.2
    0.1  0.3  0.1  0.6

   julia> classify(pred_output, LabelEnc.OneOfK)
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify(pred_output', LabelEnc.OneOfK, obsdim=1) # note the transpose
   4-element Array{Int64,1}:
    2
    1
    2
    3

   julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK) # single observation
   3


We also provide a mutating version. This is mainly of interest
when working with :func:`LabelEnc.OneOfK`, in which case broadcast
is not defined on the previous methods.

.. function:: classify!(out, arr, encoding, [obsdim])

   Same as `classify`, but uses `out` to store the result.
   In the case of a vector-based encoding this will use
   broadcast internally.
   It is mainly provided to offer a consistent API between
   vector-based and matrix-based encodings.

For convenience we also provide boolean version that assert if the
given raw prediction could be interpreted as either a positive or
a negative prediction.

.. function:: isposlabel(yhat, encoding) -> Bool

   Checks if the given value `yhat` can be interpreted as the positive
   label given the `encoding`. This function takes potential
   classification rules into account.

.. code-block:: jlcon

   julia> isposlabel([1,0], LabelEnc.OneOfK(2))
   true

   julia> isposlabel([0,1], LabelEnc.OneOfK(2))
   false

   julia> isposlabel(-5, LabelEnc.MarginBased())
   false

   julia> isposlabel(2, LabelEnc.MarginBased())
   true

   julia> isposlabel(0.3f0, LabelEnc.ZeroOne(0.5))
   false

   julia> isposlabel(0.3f0, LabelEnc.ZeroOne(0.2))
   true

.. function:: isneglabel(yhat, encoding) -> Bool

   Checks if the given value `yhat` can be interpreted as the negative
   label given the `encoding`. This function takes potential
   classification rules into account.

.. code-block:: jlcon

   julia> isneglabel([1,0], LabelEnc.OneOfK(2))
   false

   julia> isneglabel([0,1], LabelEnc.OneOfK(2))
   true

   julia> isneglabel(-5, LabelEnc.MarginBased())
   true

   julia> isneglabel(2, LabelEnc.MarginBased())
   false

   julia> isneglabel(0.3f0, LabelEnc.ZeroOne(0.5))
   true

   julia> isneglabel(0.3f0, LabelEnc.ZeroOne(0.2))
   false

