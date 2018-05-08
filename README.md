# MLLabelUtils

_Utility package for working with classification targets. As such, this package provides the necessary functionality for interpreting class-predictions, as well as converting classification targets from one encoding to another._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mllabelutilsjl.readthedocs.io/en/latest/?badge=latest) | [![MLLabelUtils 0.5](http://pkg.julialang.org/badges/MLLabelUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) [![MLLabelUtils 0.6](http://pkg.julialang.org/badges/MLLabelUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) | [![Build Status](https://travis-ci.org/JuliaML/MLLabelUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLLabelUtils.jl) [![Build status](https://ci.appveyor.com/api/projects/status/do24mf2pojqx6tai?svg=true)](https://ci.appveyor.com/project/Evizero/mllabelutils-jl) [![Coverage Status](https://coveralls.io/repos/JuliaML/MLLabelUtils.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaML/MLLabelUtils.jl?branch=master) |

## Introduction

In a classification setting, one usually treats the desired
output variable (also called *ground truths*, or *targets*) as a
discrete categorical variable. That is true even if the values
themself are of numerical type, which they often are for
practical reasons.

In fact, it is a common requirement in Machine Learning related
experiments to encode the classification targets of some
supervised dataset in a very specific way.
There are multiple conventions that all have their own merits
and reasons to exist. Some models, such as the probabilistic
version of logistic regression, require the targets in the form
of numbers in the set {1,0}. On the other hand, margin-based
classifier, such as SVMs, expect the targets to be in the set
{1,−1}.

This package provides the functionality needed to deal will these
different scenarios in an efficient, consistent, and convenient
manner. In particular, this library is designed with package
developers in mind, that require their classification-targets to
be in a specific format. To that end, the core focus of this
package is to provide all the tools needed to deal with
classification targets of arbitrary format. This includes
asserting if the targets are of a desired encoding, inferring the
concrete encoding the targets are in and how many classes they
represent, and converting from their native encoding to the
desired one.

## Example

The following code snippets show a simple "hello world" scenario
of how this package can be used to work with classification targets.

```julia
using MLLabelUtils
```

We can automatically derive the used encoding from the targets using
`labelenc`. This function looks at all elements and tries to determine
which specific encoding best describes the target array.

```julia
julia> true_targets = Int8[0, 1, 0, 1, 1];

julia> le = labelenc(true_targets)
# MLLabelUtils.LabelEnc.ZeroOne{Int8,Float64}(0.5)
```

To just determine if a specific encoding is approriate one can use
the function `islabelenc`.

```julia
julia> islabelenc(true_targets, LabelEnc.ZeroOne)
# true

julia> islabelenc(true_targets, LabelEnc.MarginBased)
# false
```

Furthermore we can compute a label map, which computes the indices
of all elements that belong to each class. This information is useful
for resampling strategies, such as stratified sampling

```julia
julia> true_targets = [:yes,:no,:maybe,:yes];

julia> labelmap(true_targets)
# Dict{Symbol,Array{Int64,1}} with 3 entries:
#   :yes   => [1,4]
#   :maybe => [3]
#   :no    => [2]
```

If need be we can convert to other encodings. Note that unless
explicitly specified, we try to preserve the `eltype` of the
input. However, this behaviour only comes to play in the case of
numbers.

```julia
julia> true_targets = Int8[0, 1, 0, 1, 1];

julia> convertlabel([:yes,:no], true_targets) # Equivalent to LabelEnc.NativeLabels([:yes,:no])
# 5-element Array{Symbol,1}:
#  :no
#  :yes
#  :no
#  :yes
#  :yes

julia> convertlabel(LabelEnc.MarginBased, true_targets) # Preserves eltype
# 5-element Array{Int8,1}:
#  -1
#   1
#  -1
#   1
#   1

julia> convertlabel(LabelEnc.MarginBased(Float32), true_targets) # Force new eltype
# 5-element Array{Float32,1}:
#  -1.0
#   1.0
#  -1.0
#   1.0
#   1.0
```

For encodings that can be multiclass, the number of classes can
be inferred from the targets, or specified explicitly.

```julia
julia> convertlabel(LabelEnc.Indices{Int}, true_targets) # number of classes inferred
# 5-element Array{Int64,1}:
#  2
#  1
#  2
#  1
#  1

julia> convertlabel(LabelEnc.Indices(Int,2), true_targets)
# 5-element Array{Int64,1}:
#  2
#  1
#  2
#  1
#  1

julia> convertlabel(LabelEnc.OneOfK{Bool}, true_targets)
# 2×5 Array{Bool,2}:
#  false   true  false   true   true
#   true  false   true  false  false
```

Note that the `OneOfK` encoding is special in that as a matrix-based
encoding it supports `ObsDim`, which can be used to specify which
dimension of the array denotes the observations.

```julia
julia> convertlabel(LabelEnc.OneOfK{Int}, true_targets, obsdim = 1)
# 5×2 Array{Int64,2}:
#  0  1
#  1  0
#  0  1
#  1  0
#  1  0
```

We also provide a `OneVsRest` encoding, which allows to transform
a multiclass problem into a binary one

```julia
julia> true_targets = [:yes,:no,:maybe,:yes];

julia> convertlabel(LabelEnc.OneVsRest(:yes), true_targets)
# 4-element Array{Symbol,1}:
#  :yes
#  :not_yes
#  :not_yes
#  :yes

julia> convertlabel(LabelEnc.TrueFalse, true_targets, LabelEnc.OneVsRest(:yes))
# 4-element Array{Bool,1}:
#   true
#  false
#  false
#   true
```

`NativeLabels` maps between data of an arbitary type (e.g. Strings) and
the other label types (Normally `LabelEnc.Indices{Int}` for an integer index).
When using it, you should always save the encoding in a variable,
and pass it as an argument to `convertlabel`; as otherwise the encoding will 
be inferred each time, so will normally encode differently for different inputs.

```julia
julia> enc = LabelEnc.NativeLabels(["copper", "tin", "gold"])
MLLabelUtils.LabelEnc.NativeLabels{String,3}(String["copper", "tin", "gold"], Dict("gold"=>3,"copper"=>1,"tin"=>2))

julia> convertlabel(LabelEnc.Indices, ["gold", "copper"], enc)
2-element Array{Int64,1}:
 3
 1
```

Encodings such as `ZeroOne`, `MarginBased`, and `OneOfK` also provide
a `classify` function.

`ZeroOne` has a threshold parameter which represents the decision
boundary.

```julia
julia> classify(0.3, 0.5)
# 0.0

julia> classify(0.3, LabelEnc.ZeroOne) # equivalent to before
# 0.0

julia> classify(0.3, LabelEnc.ZeroOne(0.2)) # custom threshold
# 1.0

julia> classify(0.3, LabelEnc.ZeroOne(Int,0.2)) # custom type
# 1

julia> classify.([0.3,0.5], LabelEnc.ZeroOne(Int,0.4)) # broadcast support
# 2-element Array{Int64,1}:
#  0
#  1
```

`MarginBased` uses the sign to determine the class.

```julia
julia> classify(-5, LabelEnc.MarginBased)
# -1

julia> classify(0.2, LabelEnc.MarginBased)
# 1.0

julia> classify(-5, LabelEnc.MarginBased(Float64))
# -1.0

julia> classify.([-5,5], LabelEnc.MarginBased(Float64))
# 2-element Array{Float64,1}:
#  -1.0
#   1.0
```

`OneOfK` determines which index is the largest element.

```julia
julia> pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
# 3×4 Array{Float64,2}:
#  0.1  0.4  0.3  0.2
#  0.8  0.3  0.6  0.2
#  0.1  0.3  0.1  0.6

julia> classify(pred_output, LabelEnc.OneOfK)
# 4-element Array{Int64,1}:
#  2
#  1
#  2
#  3

julia> classify(pred_output', LabelEnc.OneOfK, obsdim = 1) # note the transpose
# 4-element Array{Int64,1}:
#  2
#  1
#  2
#  3

julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK) # single observation
# 3
```

## Documentation

For a much more detailed treatment check out the
[latest documentation](http://mllabelutilsjl.readthedocs.io/en/latest/)

Additionally, you can make use of Julia's native docsystem. The
following example shows how to get additional information on
`convertlabel` within Julia's REPL:

```
?convertlabel
```

## Installation

This package is registered in `METADATA.jl` and can be installed
as usual. Just start up Julia and type the following code-snipped
into the REPL. It makes use of the native Julia package manger.

```julia
Pkg.add("MLLabelUtils")
```

Additionally, for example if you encounter any sudden issues, or
in the case you would like to contribute to the package, you can
manually choose to be on the latest (untagged) version.

```Julia
Pkg.checkout("MLLabelUtils")
```

## License

This code is free to use under the terms of the MIT license

