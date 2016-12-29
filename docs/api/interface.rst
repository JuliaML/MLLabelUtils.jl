Shared Interface
=================

Computing the labels
--------------------------

.. function:: label(iter) -> Vector

   Returns the labels represented in the given iterator `iter`.
   Note that the order of the labels matters. In the case of two
   labels, the first element represents the positive label and the
   second element the negative label respectively.

.. function:: nlabel(iter) -> Int

   Returns the number of labels represented in the given iterator
   `iter`.

.. function:: labeltype(type) -> DataType

   Determine the type of the labels represented by the given
   type of label-encoding.

.. function:: poslabel(encoding) -> labeltype(typeof(encoding))

   If the encoding is binary it will return the positive label of it.
   The function will throw an error otherwise.

.. function:: neglabel(encoding) -> labeltype(typeof(encoding))

   If the encoding is binary it will return the negative label of it.
   The function will throw an error otherwise.

.. function:: labelmap(obj) -> Dict

   Computes a mapping from the labels in `obj` to all the individual
   element-indices in `obj` that correspond to that label

.. function:: labelmap!(dict, idx, elem) -> Dict

   Updates the given label-map `dict` with the new element `elem`,
   which is assumed to be associated with the index `idx`.

.. function:: labelfreq(obj) -> Dict

   Computes the absolute frequencies for each label in `obj`.

.. function:: labelfreq!(dict, obj) -> Dict

   Updates the given label-frequency-map `dict` with the absolute
   frequencies for each label in `obj`

Deriving the encoding
--------------------------------------

.. function:: labelenc(obj) -> LabelEncoding

   Tries to determine the most approriate label-encoding to describe
   the given object `obj` based on the result of `label(obj)`.
   Note that in most cases this function is not typestable.

.. function:: islabelenc(obj, encoding) -> Bool

   Checks is the given object `obj` can be described as being produced
   by the given `encoding` in which case the function returns true,
   or false otherwise.

.. function:: isposlabel(x, encoding) -> Bool

   Checks if the given value `x` can be interpreted as the positive
   label given the `encoding`. This function takes potential
   classification rules into account.

.. function:: isneglabel(x, encoding) -> Bool

   Checks if the given value `x` can be interpreted as the negative
   label given the `encoding`. This function takes potential
   classification rules into account.


Converting to/from indices
--------------------------------------

.. function:: ind2label(index, encoding) -> labeltype(typeof(encoding))

   Converts the given `index` into the corresponding label defined
   by the `encoding`. Note that in the binary case, `index == 1`
   represents the positive label and `index == 2` the negative label.

   :param Int index: Index of the desired label. Note that indices are one-based.

   :param LabelEncoding encoding: The encoding one wants to get the label from.

   :return: The label of the specified index for the specified encoding.

.. function:: label2ind(label, encoding) -> Int

   Converts the given `label` into the corresponding index defined
   by the encoding. Note that in the binary case, the positive label
   will result in the index `1` and the negative label in the index
   `2` respectively.

   :param Any label: A label in the format familiar to the encoding.

   :param LabelEncoding encoding: The encoding to compute the label-index with.

   :return: The index of the specified label for the specified encoding.

Converting between encodings
------------------------------

.. function:: convertlabel(new_encoding, x, [old_encoding])

   Converts the given value/array `x` from the `old_encoding` into the
   `new_encoding`. Note that if `old_encoding` is not specified it will
   be derived automaticaly using `labelenc`.

.. function:: convertlabel(new_encoding, x, [old_encoding], [obsdim])

   When working with `LabelEnc.OneOfK` one can additionally specifify
   which dimension of the array denotes the observations using `obsdim`

Classifying predictions
-------------------------

.. function:: classify(x, encoding) -> labeltype(typeof(encoding))

   Returns the classified version of `x` given the `encoding`.
   Which means that if `x` can be interpreted as a positive label,
   the positive label of `encoding` is returned; the negative otherwise.

.. function:: classify!(out, x, encoding) -> labeltype(typeof(encoding))

   Same as `classify`, but uses `out` to store the result.

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


