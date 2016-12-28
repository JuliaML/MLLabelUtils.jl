# MLLabelUtils

_Utility package for interpreting and transforming classification targets in Machine Learning. Most notably this library provides a set of functions to convert target arrays from one representation to another._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://mllabelutilsjl.readthedocs.io/en/latest/?badge=latest) | [![MLLabelUtils 0.5](http://pkg.julialang.org/badges/MLLabelUtils_0.5.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) [![MLLabelUtils 0.6](http://pkg.julialang.org/badges/MLLabelUtils_0.6.svg)](http://pkg.julialang.org/?pkg=MLLabelUtils) | [![Build Status](https://travis-ci.org/JuliaML/MLLabelUtils.jl.svg?branch=master)](https://travis-ci.org/JuliaML/MLLabelUtils.jl) [![Build status](https://ci.appveyor.com/api/projects/status/do24mf2pojqx6tai?svg=true)](https://ci.appveyor.com/project/Evizero/mllabelutils-jl) [![Coverage Status](https://coveralls.io/repos/JuliaML/MLLabelUtils.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaML/MLLabelUtils.jl?branch=master) |

## Example

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

