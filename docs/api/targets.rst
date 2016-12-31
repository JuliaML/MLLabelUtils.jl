Classification Targets
=========================

In this section we will outline the functionality that this package
provides in order to work with classification targets.
We will start by discussion the terms we use and how they are
used in the context of this package.

Terms and Definitions
-----------------------

In a classification setting one usually treats the desired output
variable (also called *ground truths*, or *targets*) as a
discrete categorical variable. That is true even if the values
themself are of numerical type, which they often are for
practical reasons.

We use the term **targets** when we talk about concrete data.
Concretely, targets are the desired output of some dataset and further
themself also part of the dataset. If a dataset includes targets we
call it labeled data.
In a labeled dataset, each observation has its own target.
Thus we have as many targets as we have observations, as the target
is treated as a part of each observation.

.. tip::

   Let us look at an example of what targets could look like and how
   they relate to some dataset, or in this case data subset.
   The following code snipped loads the first 3 observations
   of the iris dataset using the
   `RDatasets <https://github.com/johnmyleswhite/RDatasets.jl>`_
   package.

   .. code-block:: jlcon

      julia> using RDatasets
      julia> iris = head(dataset("datasets", "iris"), 3)
      3×5 DataFrames.DataFrame
      │ Row │ SepalLength │ SepalWidth │ PetalLength │ PetalWidth │ Species  │
      ├─────┼─────────────┼────────────┼─────────────┼────────────┼──────────┤
      │ 1   │ 5.1         │ 3.5        │ 1.4         │ 0.2        │ "setosa" │
      │ 2   │ 4.9         │ 3.0        │ 1.4         │ 0.2        │ "setosa" │
      │ 3   │ 4.7         │ 3.2        │ 1.3         │ 0.2        │ "setosa" │

   For this data subset the targets would be
   ``["setosa","setosa","setosa"]``.
   Note how only one of the three available classes of the dataset
   is represented here.

The term "target" itself applies for both regression and
classification scenarios. In a classification setting (which is the
domain that this package operates in) the targets are treated as a
discrete categorical variable. If the classification targets can just
take one of two values, we call the classification problem **binary**,
or **two-class**.

For our purposes we treat the term "class" as an abstract concept
with little to no practical appearance in the functionality
provided by this package. In essence we think about a **class**
as the abstract interpretation behind some concrete value.  For
example: Let's say we try to predict if a tumor is malignant or
benign. The two classes could then be described as "malignant
tumor" and "benign tumor". One could argue that we could
translate these abstract concepts into a string or symbol quite
easily and thus make it concrete, but that is not the point. The
point is, that the concrete interpretation behind the prediction
targets is of little consequence for the library and as such it
should not talk about it.

Instead, this library cares about representation. The
representation can vary a lot between one model to another, while
the "class" remains the same.  For example, some models require
the targets in the form of numbers in the set :math:`\{1,0\}`,
other in :math:`\{1,-1\}` etc.

We call a concrete and consistent representation of a single class a
**label**. That implies that each class should consistently be
represented by a single label respectively.
How a label looks like is completely up to the user, but there
are some forms that are more common than others.
A convention of what labels to use to represent a fixed number of
classes will be referred to as a **label-encoding**, or short encoding.

.. note::

   To be fair, the term "class-encoding" would be more appropriate.
   However, when considering that we need to use the defined terms for
   naming the functions and types, it seemed more reasonable (and
   user-friendly) to keep the list of utilized domain-specific words
   small and consistent.

Determine the Labels
---------------------------

Now that we settled on the terminology, let us investigate what kind
of functionality this package provides to work with classification
targets. The first thing we may be interested in is determining what
kind of labels we are working with when presented with some targets.

In general we try to make little assumptions about the *type* of
the object containing the targets, just that it supports ``unique``.
The functions listed here do, however, expect the *object* containing
the targets to include all possible labels of the classification
scenario.

.. function:: label(iter) -> Vector

   Returns the labels represented in the given iterator `iter`.
   Note that the order of the resulting labels matters in general,
   because other functions expect the first label to denote the
   positive label for binary classification problems.
   Thus, for consistency reason there are some heuristics involved
   that try to guarantee this for the commons encodings.

   :param Any iter: Any object for which the type either implements
                    the iterator interface, or which provides a custom
                    implementation for ``unique``.

   :return: The unique labels in the form of a vector. In the case
            of two labels, the first element will represent the
            positive label and the second element the negative label
            respectively.

.. code-block:: jlcon

   julia> label([:yes,:no,:no,:maybe,:yes,:no])
   3-element Array{Symbol,1}:
    :yes
    :no
    :maybe

   julia> label([-1,1,1,-1,1])
   2-element Array{Int64,1}:
     1
    -1

As described above, we may mutate the order of the result of
``unique`` for consistency reasons in those cases that they describe
a common binary label-encoding. The reason for this is that we want
the first element to denote the positive label.
The following example highlights the different results for
``unique`` and :func:`label` in the case of targets in "zero-one" form.

.. code-block:: jlcon

   julia> unique([0,1,0,0,1])
   2-element Array{Int64,1}:
    0
    1

   julia> label([0,1,0,0,1])
   2-element Array{Int64,1}:
    1
    0

While the generic iterator implementation covers most cases, we
do selectively treat some iterators (such as ``Dict``),
differently, or even disallow some completely (such as any
``AbstractArray`` that has more than two dimensions).

.. function:: label(dict) -> Vector

   Returns the keys of the dictionary in the form of a vector.
   The reasoning behind this convention for how to interpret the
   content of a `Dict` is that we utilize dictionaries to store
   label-specific information, such as the class-frequency
   (see :func:`labelfreq`).

   Note again, that for consistency reasons there are heuristics
   in place that try to enforce the correct label-order for
   numeric label-vectors that have exactly two elements.

   :param Dict dict: Any julia dictionary.

   :return: The unique labels in the form of a vector. In the case
            of two labels, the first element will represent the
            positive label and the second element the negative label
            respectively.

We also treat matrices in a special way. The reason for this is that
for our purposes it is not their values that encode the information
about the labels, but their structure.

.. function:: label(mat, [obsdim]) -> Vector

   Returns a vector that enumerates the dimension of the given matrix
   `mat` that does **not** denote the observations. In other words it
   returns the indices of that dimension.

   :param AbstractMatrix mat: An numeric array that is assumed to be in
                              the form of a one-hot encoding or similar.

   :param ObsDimension obsdim: Optional. Denotes which of the two
        array dimensions of `mat` denotes the observations. It
        can be specified as a type-stable positional argument or
        a smart keyword (Note: for this method the return-value
        will type-stable either way). Defaults to
        ``Obsdim.Last()``.  see ``?ObsDim`` for more information.

   :return: A vector of indices that enumerate the particular
            dimension of `mat` that does not denote the
            observations.

.. code-block:: jlcon

   julia> label([0 1 0 0; 1 0 1 0; 0 0 0 1])
   3-element Array{Int64,1}:
    1
    2
    3

   julia> label([0 1 0; 1 0 0; 0 1 0; 0 0 1], obsdim = 1)
   3-element Array{Int64,1}:
    1
    2
    3

   julia> label([0 1 0; 1 0 0; 0 1 0; 0 0 1], ObsDim.First()) # positional obsdim
   3-element Array{Int64,1}:
    1
    2
    3

For convenience one can also just query for the label that
corresponds to the positive class or the negative class respectively.
These helper functions check if the given targets contain exactly two
unique labels and will throw an ``ArgumentError`` if this assumption
is violated.

.. function:: poslabel(iter) -> eltype(iter)

   If :func:`label` returns a vector of length = 2, then this
   function will return the first element of it, which denotes
   the positive label. Otherwise an error will be thrown.

.. code-block:: jlcon

   julia> poslabel([-1,1,1,-1,1])
   1

   julia> poslabel([:yes,:no,:no,:maybe,:yes,:no])
   ERROR: ArgumentError: The given object has more or less than two labels, thus poslabel is not defined.

.. function:: neglabel(iter) -> eltype(iter)

   If :func:`label` returns a vector of length = 2, then this
   function will return the second element of it, which denotes
   the negative label. Otherwise an error will be thrown.

.. code-block:: jlcon

   julia> neglabel([-1,1,1,-1,1])
   -1

   julia> neglabel([:yes,:no,:no,:maybe,:yes,:no])
   ERROR: ArgumentError: The given object has more or less than two labels, thus neglabel is not defined.

Number of Labels
--------------------

We can compute the number of unique labels using :func:`nlabel`.
It works by first computing the labels and then counting them.
As such it has the same restrictions as :func:`label`.

.. function:: nlabel(iter) -> Int

   Returns the number of labels represented in the given iterator
   `iter`. It uses the function :func:`label` internally, so the
   same properties and restrictions apply.

   :param Any iter: Any object for which the function :func:`label`
                    is implemented.

.. code-block:: jlcon

   julia> nlabel([:yes,:no,:no,:maybe,:yes,:no])
   3

   julia> nlabel([-1,1,1,-1,1])
   2


Mapping Labels to Observations
---------------------------------

In many classification scenarios we have to deal with what is called
an imbalanced class distribution. In essence that means that some
classes are represented more often in a given dataset than the other
classes. While we won't go into detail about the implications of such
a scenario, the key takeaway is that there exist strategies to deal
with those situations by using information about how the class-label
are distributed. More importantly even, some require a mapping from
each label to all the observations that have that label as target.
We call such a mapping from labels to observation-indices a
**label-map**.

.. function:: labelmap(iter) -> Dict

   Computes a mapping from the labels in `iter` to all the individual
   element-indices in `iter` that correspond to that label.
   Note that there is actually no check or requirement that `iter`
   must implement `length` or `getindex`. Instead, it is assumed that
   the first element of the iterator has the index ``1`` and the
   indices are incremented by ``1`` with each element of the iterator.

   :param Any iter: Any object for which the type implements the
                    iterator interface

   :return: A dictionary that for each label as key, has a vector
            as value that contains all indices of the observations
            that observed that label.

.. code-block:: jlcon

   julia> labelmap([0, 1, 1, 0, 0])
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5]
     1 => [2,3]

   julia> labelmap([:yes,:no,:no,:maybe,:yes,:no])
   Dict{Symbol,Array{Int64,1}} with 3 entries:
     :yes   => [1,5]
     :maybe => [4]
     :no    => [2,3,6]

We also provide a mutating version to update an existing label-map.
In those cases we also have to specify the index/indices of that new
observation(s).

.. function:: labelmap!(dict, idx, elem) -> Dict

   Updates the given label-map `dict` with the new element `elem`,
   which is assumed to be associated with the index `idx`.
   Note that the given index is not checked for being a duplicate.

   :param Dict dict: The dictionary that may or may not already
                     contain existing label-mapping.
                     It will be updated with the new element.

   :param Int idx: The observation-index that `elem` corresponds
                   to in the context of the overall dataset.

   :param Any elem: The new target of the observation denoted by `idx`.
                    It is expected to be in the form of a label.

   :return: Returns the mutated `dict` for convenience.

.. code-block:: jlcon

   julia> lm = labelmap([0, 1, 1, 0, 0])
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5]
     1 => [2,3]

   julia> labelmap!(lm, 6, 0)
   Dict{Int64,Array{Int64,1}} with 2 entries:
     0 => [1,4,5,6]
     1 => [2,3]

.. function:: labelmap!(dict, indices, iter) -> Dict

   Updates the given label-map `dict` with the new elements in the
   given iterator `iter`. Each element in `iter` is assumed to be
   associated with the corresponding index in `indices`.
   This implies that both, `iter` and `indices`, must provide the same
   amount of elements.
   Note that the given indices are not checked for being duplicates.

   :param Dict dict: The dictionary that may or may not already
                     contain existing label-mapping. It will be
                     updated with the new elements in `iter`.

   :param AbstractVector{Int} indices: The indices for each element in
                                       `iter`.

   :param Any iter: Any object for which the type implements the
                    iterator interface.

   :return: Returns the mutated `dict` for convenience.

.. code-block:: jlcon

   julia> lm = labelmap([:yes,:no,:no,:maybe,:yes,:no])
   Dict{Symbol,Array{Int64,1}} with 3 entries:
     :yes   => [1,5]
     :maybe => [4]
     :no    => [2,3,6]

   julia> labelmap!(lm, 7:8, [:no,:maybe])
   Dict{Symbol,Array{Int64,1}} with 3 entries:
     :yes   => [1,5]
     :maybe => [4,8]
     :no    => [2,3,6,7]

Frequency of Labels
------------------------

Another useful information to compute is the absolute frequency of
each label in the dataset of interest. In contrast to :func:`labelmap`,
this function does not care about indices but instead simply counts
occurrences. We call such a dictionary a **frequency-map**.

.. function:: labelfreq(iter) -> Dict

   Computes the absolute frequencies for each label in `iter` and
   adds it as a key (label) value (count) pair to the resulting
   dictionary.

   :param Any iter: Any object for which the type implements the
                    iterator interface

   :return: A dictionary that for each label as key, has an Int
            as value that denotes how often the corresponding label
            was encountered in `iter`

.. code-block:: jlcon

   julia> labelfreq([0, 1, 1, 0, 0])
   Dict{Int64,Int64} with 2 entries:
     0 => 3
     1 => 2

   julia> labelfreq([:yes,:no,:no,:maybe,:yes,:no])
   Dict{Symbol,Int64} with 3 entries:
     :yes   => 2
     :maybe => 1
     :no    => 3

If you have already created a mapping using :func:`labelmap`, then you
can reuse that dictionary to compute the frequencies more efficiently.

.. function:: labelfreq(dict) -> Dict

   Converts a label-map to a frequency map by counting the number
   of indices associated with each label.

   :param Dict dict: A dictionary produced by :func:`labelmap`.

   :return: A dictionary that for each label as key, has an Int
            as value that denotes how many indices were stored in
            `dict` for the corresponding label.

.. code-block:: jlcon

   julia> lm = labelmap([:yes,:no,:no,:maybe,:yes,:no])
   Dict{Symbol,Array{Int64,1}} with 3 entries:
     :yes   => [1,5]
     :maybe => [4]
     :no    => [2,3,6]

   julia> labelfreq(lm)
   Dict{Symbol,Int64} with 3 entries:
     :yes   => 2
     :maybe => 1
     :no    => 3

For some data sources it may not be useful or even possible to
associate an observation with an index (e.g. streaming data).
For such cases it may still prove useful to continuously keep track
of the number of times each label was encountered.
To that end we provide a mutating version that updates a
frequency-map in-place.

.. function:: labelfreq!(dict, iter) -> Dict

   Updates the given frequency-map `dict` with the number of times
   each label occurs in the given iterator `iter`.
   Note that these occurances are added to the current values.

   :param Dict dict: The dictionary that may or may not already
                     contain existing frequency information. It will
                     be updated with the new elements in `iter`.

   :param Any iter: Any object for which the type implements the
                    iterator interface.

   :return: Returns the mutated `dict` for convenience.

.. code-block:: jlcon

   julia> lf = labelfreq([:yes,:no,:no,:maybe,:yes,:no])
   Dict{Symbol,Int64} with 3 entries:
     :yes   => 2
     :maybe => 1
     :no    => 3

   julia> labelfreq!(lf, [:no,:maybe])
   Dict{Symbol,Int64} with 3 entries:
     :yes   => 2
     :maybe => 2
     :no    => 4

