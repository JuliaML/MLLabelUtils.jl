# everything that should move to learnbase

abstract LabelEncoding{T,K,M} # Eltype, Labelcount, Arraydimensions
typealias BinaryLabelEncoding{T,M} LabelEncoding{T,2,M}
typealias VectorLabelEncoding{T,K} LabelEncoding{T,K,1}
typealias MatrixLabelEncoding{T,K} LabelEncoding{T,K,2}

Base.size(::LabelEncoding) = ()
Base.getindex(lm::LabelEncoding, idx) = lm

Base.size{T<:LabelEncoding}(::Type{T}) = ()
Base.getindex{T<:LabelEncoding}(t::Type{T}, idx) = t

"""
    nlabel(obj) -> Int

Returns the number of labels represented in the given object `obj`.

    julia> nlabel([:yes,:no,:yes,:yes])
    2
"""
nlabel{T<:BinaryLabelEncoding}(::Type{T}) = 2
nlabel{T,K,M}(::Type{LabelEncoding{T,K,M}}) = Int(K)
nlabel(::Type{Any}) = error("nlabel not defined for the given type")
nlabel{T}(::Type{T}) = nlabel(supertype(T))
nlabel{T,K}(::LabelEncoding{T,K}) = Int(K)
nlabel(itr) = length(unique(itr))

"""
    label(obj) -> Vector

Returns the labels represented in the given object `obj`.
Note that the order of the labels matters.
In the case of two labels, the first element represents the positive
label and the second element the negative label.

    julia> label([:yes,:no,:yes,:yes])
    2-element Array{Symbol,1}:
     :yes
     :no

    julia> label(LabelEnc.ZeroOne())
    2-element Array{Float64,1}:
     1.0
     0.0
"""
label(itr) = _arrange_label(unique(itr))

# make sure pos label is first
_arrange_label(lbl::Vector) = lbl
_arrange_label{T<:Bool}(lbl::Vector{T}) = [true,false]
function _arrange_label{T<:Number}(lbl::Vector{T})
    if length(lbl) == 2
        if minimum(lbl) == 0 && maximum(lbl) == 1
            lbl[1] = T(1)
            lbl[2] = T(0)
        elseif minimum(lbl) == -1 && maximum(lbl) == 1
            lbl[1] = T(1)
            lbl[2] = T(-1)
        elseif minimum(lbl) == 1 && maximum(lbl) == 2
            lbl[1] = T(1)
            lbl[2] = T(2)
        end
    end
    lbl
end


"""
    labeltype{T<:LabelEncoding}(::Type{T}) -> DataType

Determine the type of the labels represented by the label encoding `T`
"""
labeltype{T}(::Type{MatrixLabelEncoding{T}}) = T
labeltype{T}(::Type{VectorLabelEncoding{T}}) = T
labeltype{T,K,M}(::Type{LabelEncoding{T,K,M}}) = T
labeltype(::Type{Any}) = Any
labeltype{T}(::Type{T}) = labeltype(supertype(T))
labeltype{T}(lm::LabelEncoding{T}) = T

"""
    ind2label(index, encoding)

Converts the given `index` into the corresponding label defined
by the `encoding`. Note that in the binary case, `index == 1`
represents the positive label and `index == 2` the negative label.

    julia> ind2label(1, LabelEnc.ZeroOne(Float32))
    1.0f0

    julia> ind2label(2, LabelEnc.ZeroOne(Float32))
    0.0f0
"""
function ind2label end

"""
    label2ind(label, encoding) -> Int

Converts the given `label` into the corresponding index defined
by the encoding. Note that in the binary case, the positive label
will result in the index `1` and the negative label in the index `2`
respectively

    julia> label2ind(:no, LabelEnc.NativeLabels([:yes,:no]))
    2

    julia> label2ind(1, LabelEnc.MarginBased())
    1

    julia> label2ind(-1, LabelEnc.MarginBased())
    2
"""
function label2ind end

"""
    poslabel(encoding)

If the encoding is binary it will return the positive label of it.
The function will throw an error otherwise.

    julia> poslabel(LabelEnc.MarginBased(Float32))
    1.0f0
"""
function poslabel end

"""
    neglabel(encoding)

If the encoding is binary it will return the negative label of it.
The function will throw an error otherwise.

    julia> neglabel(LabelEnc.MarginBased(Float32))
    -1.0f0
"""
function neglabel end

"""
    labelenc(obj) -> LabelEncoding

Tries to determine the most approriate label-encoding to describe the
given object `obj` based on the result of `label(obj)`. Note that in
most cases this function is not typestable.

    julia> labelenc([:yes,:no,:no,:yes,:maybe])
    MLLabelUtils.LabelEnc.NativeLabels{Symbol,3}(Symbol[:yes,:no,:maybe],Dict(:yes=>1,:maybe=>3,:no=>2))

    julia> labelenc([1,0,0,1,0,1])
    MLLabelUtils.LabelEnc.ZeroOne{Int64,Float64}(0.5)

    julia> labelenc(Int8[1,-1,-1,1,-1,1])
    MLLabelUtils.LabelEnc.MarginBased{Int8}()
"""
function labelenc end

"""
    islabelenc(obj, encoding) -> Bool

Checks is the given object `obj` can be described as being produced
by the given `encoding` in which case the function returns true,
or false otherwise.

    julia> islabelenc([1,0,1], LabelEnc.ZeroOne)
    true

    julia> islabelenc([1,-1,1], LabelEnc.ZeroOne)
    false
"""
function islabelenc end

"""
    isposlabel(x, encoding) -> Bool

Checks if the given value `x` can be interpreted as the positive label
given the `encoding`. This function takes potential classification
rules into account.

    julia> isposlabel(0.6, LabelEnc.ZeroOne(0.5))
    true

    julia> isposlabel(0.4, LabelEnc.ZeroOne(0.5))
    false

    julia> isposlabel(:b, LabelEnc.NativeLabels([:a,:b]))
    false
"""
function isposlabel end

"""
    isneglabel(x, encoding) -> Bool

Checks if the given value `x` can be interpreted as the negative label
given the `encoding`. This function takes potential classification
rules into account.

    julia> isneglabel(0.6, LabelEnc.ZeroOne(0.5))
    false

    julia> isneglabel(0.4, LabelEnc.ZeroOne(0.5))
    true

    julia> isneglabel(:b, LabelEnc.NativeLabels([:a,:b]))
    true
"""
function isneglabel end

"""
    classify(x, encoding)

Returns the classified version of `x` given the `encoding`.
Which means that if `x` can be interpreted as a positive label,
the positive label of `encoding` is returned; the negative otherwise.

    julia> classify(0.6, LabelEnc.ZeroOne(UInt8))
    0x01

    julia> classify(0.4, LabelEnc.ZeroOne(UInt8))
    0x00

    julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK)
    3
"""
function classify end

"""
    classify!(out, x, encoding)

Same as `classify`, but uses `out` to store the result.

    julia> buffer = zeros(2);
    julia> classify!(buffer, [0.4,0.6], LabelEnc.ZeroOne)
    2-element Array{Float64,1}:
     0.0
     1.0
"""
function classify! end

"""
    convertlabel(new_encoding, x, [old_encoding])

Converts the given value/array `x` from the `old_encoding` into the
`new_encoding`. Note that if `old_encoding` is not specified it will
be derived automaticaly using `labelenc`.

    julia> convertlabel(LabelEnc.MarginBased, [0, 1, 1, 0, 0])
    5-element Array{Int64,1}:
     -1
      1
      1
     -1
     -1

    julia> convertlabel([:yes,:no], [0, 1, 1, 0, 0])
    5-element Array{Symbol,1}:
     :no
     :yes
     :yes
     :no
     :no

For more information on the available encodings, see `?LabelEnc`.

    convertlabel(new_encoding, x, [old_encoding], [obsdim])

When working with `OneOfK` one can additionally specifify which
dimension of the array denotes the observations using `obsdim`

    julia> convertlabel(LabelEnc.OneOfK, [0, 1, 1, 0, 0], obsdim = 2)
    2×5 Array{Int64,2}:
     0  1  1  0  0
     1  0  0  1  1
"""
function convertlabel end
function convertlabel! end

"""
    labelmap(obj) -> Dict

Computes a mapping from the labels in `obj` to all the individual
element-indices in `obj` that correspond to that label

    julia> labelmap([0, 1, 1, 0, 0])
    Dict{Int64,Array{Int64,1}} with 2 entries:
      0 => [1,4,5]
      1 => [2,3]
"""
function labelmap end

"""
    labelmap!(dict, idx, elem) -> Dict

Updates the given label-map `dict` with the new element `elem`,
which is assumed to be associated with the index `idx`.

    julia> lm = labelmap([0, 1, 1, 0, 0])
    Dict{Int64,Array{Int64,1}} with 2 entries:
      0 => [1,4,5]
      1 => [2,3]

    julia> labelmap!(lm, 6, 0)
    Dict{Int64,Array{Int64,1}} with 2 entries:
      0 => [1,4,5,6]
      1 => [2,3]

    julia> labelmap!(lm, 7:8, [1,0])
    Dict{Int64,Array{Int64,1}} with 2 entries:
      0 => [1,4,5,6,8]
      1 => [2,3,7]
"""
function labelmap! end

"""
    labelfreq(obj) -> Dict

Computes the absolute frequencies for each label in `obj`.

    julia> labelfreq([0, 1, 1, 0, 0])
    Dict{Int64,Int64} with 2 entries:
      0 => 3
      1 => 2
"""
function labelfreq end

"""
    labelfreq!(dict, obj) -> Dict

updates the given label-frequency-map `dict` with the absolute
frequencies for each label in `obj`

    julia> ld = labelfreq([0, 1, 1, 0, 0])
    Dict{Int64,Int64} with 2 entries:
      0 => 3
      1 => 2

    julia> labelfreq!(ld, [1,0,0])
    Dict{Int64,Int64} with 2 entries:
      0 => 5
      1 => 3
"""
function labelfreq! end

