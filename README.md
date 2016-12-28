# MLLabelUtils

_Utility package for interpreting and transforming classification targets for the most commonly used class-label encodings in Machine Learning. As such, this package provides functionality to derive or assert properties about some label-encoding or targets array, as well as the functions needed to convert some given targets array into a different label encoding._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mllabelutilsjl.readthedocs.io/en/latest/?badge=latest) | [![MLLabelUtils 0.5](http://pkg.julialang.org/badges/MLLabelUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) [![MLLabelUtils 0.6](http://pkg.julialang.org/badges/MLLabelUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) | [![Build Status](https://travis-ci.org/JuliaML/MLLabelUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLLabelUtils.jl) [![Build status](https://ci.appveyor.com/api/projects/status/do24mf2pojqx6tai?svg=true)](https://ci.appveyor.com/project/Evizero/mllabelutils-jl) [![Coverage Status](https://coveralls.io/repos/JuliaML/MLLabelUtils.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaML/MLLabelUtils.jl?branch=master) |

## Introduction

It is a common requirement in Machine Learning related experiments
to encode the classification targets of some supervised dataset in
one way or the other.


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
true_targets = Int8[0, 1, 0, 1, 1]
le = labelenc(true_targets)
```
```
MLLabelUtils.LabelEnc.ZeroOne{Int8,Float64}(0.5)
```

To just determine if a specific encoding is approriate one can use
the function `islabelenc`.

```julia
@assert islabelenc(true_targets, LabelEnc.MarginBased) == false
```

Furthermore we can compute a label map, which computes the indices
of all elements that belong to each class. This information is useful
for resampling strategies, such as stratified sampling

```julia
labelmap(true_targets)
```
```
Dict{Symbol,Array{Int64,1}} with 3 entries:
  :yes   => [1,4]
  :maybe => [3]
  :no    => [2]
```

If need be we can convert to other encodings. Note that unless
otherwise specified, the method tries to preserve the `eltype` of the
input. This only comes to play in the case of numbers.

```julia
# Specify new arbitrary output labels directly.
# Equivalent to using LabelEnc.NativeLabels([:yes,:no])
@assert convertlabel([:yes,:no], true_targets) == [:no,:yes,:no,:yes,:yes]
# Preserve eltype of input array
@assert convertlabel(LabelEnc.MarginBased, true_targets) == Int8[-1,1,-1,1,1]
# Specify new eltype for output array
@assert convertlabel(LabelEnc.MarginBased(Float32), true_targets) == Float32[-1,1,-1,1,1]
# For encodings that can be multiclass, the number of classes can be inferred
@assert convertlabel(LabelEnc.Indices{Int},   true_targets) == Int[2,1,2,1,1]
@assert convertlabel(LabelEnc.Indices(Int,2), true_targets) == Int[2,1,2,1,1]
@assert convertlabel(LabelEnc.OneOfK{Bool},   true_targets) == [false true false true true; true false true false false]
```

Note that the `OneOfK` encoding is special in that as a matrix-based
encoding it supports `ObsDim`, which can be used to specify which
dimension of the array denotes the observations.

```julia
@assert convertlabel(LabelEnc.OneOfK{Int}, true_targets, obsdim = 1) == [0 1; 1 0; 0 1; 1 0; 1 0]
```

We also provide a `OneVsRest` encoding, which allows to transform
a multiclass problem into a binary one

```julia
true_targets = [:yes,:no,:maybe,:yes]
@assert convertlabel(LabelEnc.OneVsRest(:yes), true_targets) == [:yes, :not_yes, :not_yes, :yes]
@assert convertlabel(LabelEnc.TrueFalse, true_targets, LabelEnc.OneVsRest(:yes)) == [true, false, false, true]
```

Encodings such as `ZeroOne`, `MarginBased`, and `OneOfK` also provide
a `classify` function.

`ZeroOne` has a cutoff parameter which represents the decision boundary

```julia
@assert classify(0.3, 0.5) === 0.
@assert classify(0.3, LabelEnc.ZeroOne) === 0. # equivalent to before
@assert classify(0.3, LabelEnc.ZeroOne(0.5)) === 0. # equivalent to before
@assert classify(0.3, LabelEnc.ZeroOne(Int,0.2)) === 1
@assert classify.([0.3,0.5], LabelEnc.ZeroOne(Int,0.4)) == [0,1]
```

`MarginBased` uses the sign to determine the class

```julia
@assert classify(-5, LabelEnc.MarginBased) === -1
@assert classify(0.2, LabelEnc.MarginBased) === 1.
@assert classify(-5, LabelEnc.MarginBased(Float64)) === -1.
@assert classify.([-5,5], LabelEnc.MarginBased(Float64)) == [-1.,1.]
```

`OneOfK` determines which index is the largest element

```julia
pred_output = [0.1 0.4 0.3 0.2; 0.8 0.3 0.6 0.2; 0.1 0.3 0.1 0.6]
@assert classify(pred_output, LabelEnc.OneOfK) == [2,1,2,3]
@assert classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK) === 3
```

## Documentation

check out the [latest documentation](http://mllabelutilsjl.readthedocs.io/en/latest/)

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

