MLLabelUtils.jl's documentation
=================================

This package represents a community effort to provide common
functionality for interpreting and transforming classification targets.
It does so by defining a set of commonly used class-label encodings.
It is part of the `JuliaML <https://github.com/JuliaML>`_ ecosystem.

The main intend of this package is to be a light-weight back-end for
other JuliaML packages that deal with classification problems.
In particular, this library is designed with package developers in
mind that require their classification-targets to be in a specific
format. As such, the core focus of this package is to provide all
the tools needed to deal with classification targets of arbitrary
format.

From an end-user's perspective one normally does not need to import
this package directly. That said, some functionality (in particular
:func:`convertlabels`) can also be useful to end-users who code
their own special Machine Learning scripts.

Where to begin?
----------------

If this is the first time you consider using MLLabelUtils for your
machine learning related experiments or packages, make sure to check
out the "Getting Started" section; specifically "How to ...?", which
lists some of most common scenarios and links to the appropriate places
that should guide you on how to approach these scenarios using the
functionality provided by this or other packages.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted


API Documentation
--------------------

This section gives a more detailed treatment of all the exposed
functions and their available methods.
We start by discussing what we understand under terms such as
"classification targets" and the available functionality to compute
properties about them.

.. toctree::
   :maxdepth: 2

   api/targets

Next we focus on label-encodings. We will show how to create them
and how they can be used to transfrom classification targets from
one encoding-convention to another.

.. toctree::
   :maxdepth: 2

   api/interface

Lastly we provide an organized list of the implemented label-encoding
that this package exposes.

.. toctree::
   :maxdepth: 2

   api/labelencoding


Indices and tables
==================

.. toctree::
   :hidden:
   :maxdepth: 2

   about/license

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

