MLLabelUtils.jl's documentation
=================================

This package represents a community effort to provide common
functionality to interpret and transform classification targets.
It does so by defining a set of commonly used class-label encodings.
This package is part of the JuliaML ecosystem.

The main intend of this package is to be a light-weight back-end for
other JuliaML packages that deal with classification problems.
As such, one normally does not need to import this package directly.
That said, some functionality (in particular :func:`convertlabels`)
can also be useful to end-users who code their own low-level
Machine Learning functionality.

Where to begin?
----------------

If this is the first time you consider using MLLabelUtils for your
machine learning related experiments, make sure to check out the
"Getting Started" section; specifically "How to ...?", which
lists some of most common scenarios and links to the appropriate places
that should guide you on how to approach these scenarios using the
functionality provided by this or other packages.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted


API Documentation
--------------------

.. toctree::
   :maxdepth: 2

   api/interface
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

