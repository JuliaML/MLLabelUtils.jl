"""
The submodule `MLLabelUtils.LabelEnc` contains a selection of
popular label encodings:

Strictly binary label encodings:

- [`LabelEnc.FuzzyBinary`](@ref)
- [`LabelEnc.TrueFalse`](@ref)
- [`LabelEnc.ZeroOne`](@ref)
- [`LabelEnc.MarginBased`](@ref)

Multiclass label encodings:

- [`LabelEnc.Indices`](@ref)
- [`LabelEnc.NativeLabels`](@ref)
- [`LabelEnc.OneOfK`](@ref)

Multiclass to binary:

- [`LabelEnc.OneVsRest`](@ref)
"""
module LabelEnc
    import ..MLLabelUtils.LabelEncoding
    import ..MLLabelUtils.BinaryLabelEncoding

    """
        LabelEnc.FuzzyBinary <: LabelEncoding{Any,2,1}

    A vector-based binary label encoding without a specific labeltype.

    Primarily used for fuzzy comparision of binary true targets
    and predicted targets. It basically assumes that the encoding
    is either `TrueFalse`, `ZeroOne`, or `MarginBased` by treating
    all non-negative values as positive outputs.
    """
    struct FuzzyBinary <: BinaryLabelEncoding{Any,1} end

    """
        LabelEnc.TrueFalse <: LabelEncoding{Bool,2,1}

    A vector-based binary label encoding with labeltype `Bool`.

    Represents binary classification-labels as boolean, in which `true`
    represents the positive class, and `false` the negative class.
    """
    struct TrueFalse <: BinaryLabelEncoding{Bool,1} end

    """
        LabelEnc.ZeroOne{T<:Number} <: LabelEncoding{T,2,1}

    A vector-based binary label encoding with a numeric labeltype.

    Represents binary classification-labels as numbers, in which
    `one(T)` represents the positive class, and `zero(T)` the
    negative class.
    """
    struct ZeroOne{T<:Number,R<:Number} <: BinaryLabelEncoding{T,1}
        cutoff::R
        function ZeroOne{T,R}(cutoff::R = R(0.5)) where {T,R}
            @assert 0 <= cutoff <= 1
            new{T,R}(cutoff)
        end
    end
    ZeroOne(t::Type{T} = Float64, cutoff::R = 0.5) where {T<:Number,R<:Number} = ZeroOne{T,R}(cutoff)
    ZeroOne(cutoff::R) where {R<:Number} = ZeroOne(R, cutoff)

    """
        LabelEnc.MarginBased{T<:Number} <: LabelEncoding{T,2,1}

    A vector-based binary label encoding with a numeric labeltype.

    Represents binary classification-labels as numbers, in which
    `one(T)` represents the positive class, and `-one(T)` the
    negative class.
    """
    struct MarginBased{T<:Number} <: BinaryLabelEncoding{T,1} end
    MarginBased(::Type{T} = Float64) where {T<:Number} = MarginBased{T}()

    """
        LabelEnc.OneVsRest{T} <: LabelEncoding{T,2,1}

    A vector-based binary label encoding with a arbitrary labeltype.

    A special label-encoding in that it only uses a positive
    label to determine which class some element belongs to.
    If some value matches the specified positive label, then it is
    considered positive, otherwise it is considered negative.

    That said `OneVsRest` requires a negative label to be specified
    in order to be able to denote it somehow. Note, however,
    that the specified negative label is purely for asthetic
    reasons and is not used to determine class membership
    """
    struct OneVsRest{T} <: BinaryLabelEncoding{T,1}
        poslabel::T
        neglabel::T
    end
    OneVsRest(poslabel::Bool) = OneVsRest{Bool}(poslabel, !poslabel)
    OneVsRest(poslabel::String) = OneVsRest{String}(poslabel, "not_$(poslabel)")
    OneVsRest(poslabel::Symbol) = OneVsRest{Symbol}(poslabel, Symbol(:not_, poslabel))
    OneVsRest(poslabel::T) where {T<:Number} = OneVsRest{T}(poslabel, ifelse(poslabel == 0, poslabel+T(1), T(0)))

    """
        LabelEnc.Indices{T,K<:Number} <: LabelEncoding{T,K,1}

    A vector-based multi-label encoding with a numeric labeltype.

    Represents the class-labels as indices starting from `T(1)` up to
    (including) `T(K)`.
    In a binary setting `T(1)` corresponds to te positive class and
    `T(2)` corresponds to the negative class.
    """
    struct Indices{T<:Number,K} <: LabelEncoding{T,K,1}
        function Indices{T,K}() where {T,K}
            typeof(K) <: Int || throw(TypeError(:Indices,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            new{T,K}()
        end
    end
    Indices(::Type{T}, ::Type{Val{K}}) where {T,K} = Indices{T,K}()
    Indices(::Type{Val{K}}) where {K} = Indices(Int,Val{K})
    Indices(::Type{T}, K::Number) where {T} = Indices(T,Val{Int(K)})
    Indices(K::Number) = Indices(Int,Val{Int(K)})

    """
        LabelEnc.OneOfK{T,K<:Number} <: LabelEncoding{T,K,2}

    A matrix-based multi-label encoding with a numeric labeltype.

    Represents the class labels in a one-hot encoding scheme.
    That means a matrix in which for every observation one of
    `K` elements is set to 1.
    """
    struct OneOfK{T<:Number,K} <: LabelEncoding{T,K,2}
        function OneOfK{T,K}() where {T,K}
            typeof(K) <: Int || throw(TypeError(:OneOfK,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            new{T,K}()
        end
    end
    OneOfK(::Type{T}, ::Type{Val{K}}) where {T,K} = OneOfK{T,K}()
    OneOfK(::Type{Val{K}}) where {K} = OneOfK(Int,Val{K})
    OneOfK(::Type{T}, K::Number) where {T} = OneOfK(T,Val{Int(K)})
    OneOfK(K::Number) = OneOfK(Int,Val{Int(K)})


    """
        LabelEnc.NativeLabels{T,K,F} <: LabelEncoding{T,K,1}

    A vector-based multi-label encoding with a arbitrary labeltype.

    Represents arbitrary class labels by storing the possible values
    as a member variable.

    In a binary setting the first element of the stored class labels
    represents the positive label and the second element the negative
    label.
    """
    struct NativeLabels{T,K,F} <: LabelEncoding{T,K,1}
        getfallbacklabel::F
        label::Vector{T}
        invlabel::Dict{T,Int}
        function NativeLabels{T,K,F}(getfallbacklabel::F, label::Vector{T}) where {T,K,F<:Function}
            typeof(K) <: Int || throw(TypeError(:NativeLabels,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            @assert length(label) == length(unique(label)) == K
            new{T,K,F}(getfallbacklabel, label, Dict(zip(label,1:K)))
        end
    end
    const default_getfallbacklabel = identity # by default leave label as is, so parent can change

    NativeLabels{T,K}(getfallbacklabel::F, label::Vector{T}) where {T,K,F<:Function} = NativeLabels{T,K,F}(getfallbacklabel, label)
    NativeLabels{T,K}(fallbacklabel, label::Vector{T}) where {T,K} = NativeLabels{T,K}(oov->fallbacklabel, label)
    NativeLabels{T,K}(label::Vector{T}) where {T,K} = NativeLabels{T,K}(default_getfallbacklabel, label)

    NativeLabels{T,K}(getfallbacklabel::F, label::AbstractVector{T}) where {F,T,K} = NativeLabels{T,K}(getfallbacklabel, collect(label))
    NativeLabels{T,K}(fallbacklabel::T, label::AbstractVector{T}) where {T,K} = NativeLabels{T,K}(oov->fallbacklabel, label)
    NativeLabels{T,K}(label::AbstractVector{T}) where {T,K} = NativeLabels{T,K}(default_getfallbacklabel, label)
    
    NativeLabels(getfallbacklabel::Function, label::AbstractVector{T}, ::Type{Val{K}})  where {T,K} = NativeLabels{T,K}(getfallbacklabel, label)
    NativeLabels(fallbacklabel, label::AbstractVector{T}, ::Type{Val{K}})  where {T,K} = NativeLabels(oov->fallbacklabel, label, Val{K})
    NativeLabels(label::AbstractVector, ::Type{Val{K}})  where {K} = NativeLabels(default_getfallbacklabel, label, Val{K})
    
    NativeLabels(getfallbacklabel, label) = NativeLabels(getfallbacklabel, label, Val{length(label)})
    NativeLabels(label) = NativeLabels(default_getfallbacklabel, label)

    Base.hash(a::NativeLabels, h::UInt) = hash(a.getfallbacklabel, hash(a.label, hash(:NativeLabels, h)))
    Base.:(==)(a::NativeLabels, b::NativeLabels) = isequal(a.label, b.label) && isequal(a.getfallbacklabel, b.getfallbacklabel)

end # submodule

standardize_label(lbl, lm::LabelEnc.NativeLabels) = haskey(lm.invlabel, lbl) ? lbl : lm.getfallbacklabel(lbl)

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelEncoding manually."))

# Query the index
label2ind(lbl, lm::BinaryLabelEncoding) = ifelse(isposlabel(lbl, lm), 1, 2)
label2ind(lbl::Union{Number,T}, lm::LabelEnc.Indices{T}) where {T} = Int(lbl)
label2ind(lbl::Union{Number,T}, lm::LabelEnc.OneOfK{T}) where {T} = Int(lbl)
label2ind(lbl::AbstractVector, lm::LabelEnc.OneOfK) = indmax(lbl)
function label2ind(lbl, lm::LabelEnc.NativeLabels)
    std_lbl = standardize_label(lbl, lm)
    Int(lm.invlabel[std_lbl])
end

# Query the label
ind2label(i::Integer, lm::BinaryLabelEncoding) = ifelse(i == 1, ind2label(Val{1},lm), ind2label(Val{2},lm))
ind2label(i::Integer, ::LabelEnc.Indices{T}) where {T} = T(i)
ind2label(i::Integer, ::LabelEnc.OneOfK{T,K}) where {T,K} = (x = zeros(T,K); x[i] = T(1); x)
ind2label(i::Integer, lm::LabelEnc.NativeLabels) = lm.label[Int(i)]

ind2label(::Type{Val{1}}, ::LabelEnc.TrueFalse) = true
ind2label(::Type{Val{2}}, ::LabelEnc.TrueFalse) = false
ind2label(::Type{Val{1}}, ::LabelEnc.ZeroOne{T}) where {T} = T(1)
ind2label(::Type{Val{2}}, ::LabelEnc.ZeroOne{T}) where {T} = T(0)
ind2label(::Type{Val{1}}, ::LabelEnc.MarginBased{T}) where {T} = T(1)
ind2label(::Type{Val{2}}, ::LabelEnc.MarginBased{T}) where {T} = -T(1)
ind2label(::Type{Val{1}}, ovr::LabelEnc.OneVsRest) = ovr.poslabel
ind2label(::Type{Val{2}}, ovr::LabelEnc.OneVsRest) = ovr.neglabel
ind2label(::Type{Val{K}}, ::LabelEnc.Indices{T}) where {T,K} = T(K)
ind2label(::Type{Val{I}}, ::LabelEnc.OneOfK{T,K}) where {T,I,K} = (x = zeros(T,K); x[I] = T(1); x)
ind2label(::Type{Val{K}}, lm::LabelEnc.NativeLabels) where {K} = lm.label[K]

poslabel(lm::LabelEnc.BinaryLabelEncoding) = ind2label(Val{1}, lm)
neglabel(lm::LabelEnc.BinaryLabelEncoding) = ind2label(Val{2}, lm)

label(lm::LabelEnc.BinaryLabelEncoding{T,1}) where {T} = [poslabel(lm), neglabel(lm)]
label(::LabelEnc.Indices{T,K}) where {T,K} = collect(T(1):T(K))
label(::LabelEnc.OneOfK{T,K}) where {T,K} = collect(1:K)
label(lm::LabelEnc.NativeLabels) = lm.label

labeltype(::Type{LabelEnc.ZeroOne}) = Number
labeltype(::Type{LabelEnc.MarginBased}) = Number
labeltype(::Type{LabelEnc.Indices}) = Number
labeltype(::Type{LabelEnc.OneOfK}) = Number

# What it means to be a positive label
isposlabel(value, ::LabelEnc.FuzzyBinary) = _ambiguous()
isposlabel(value, ovr::LabelEnc.OneVsRest) = (value == ovr.poslabel)
isposlabel(value::Bool, ::LabelEnc.FuzzyBinary) = value
isposlabel(value::Bool, ::LabelEnc.TrueFalse)   = value
isposlabel(value::Bool, ::LabelEnc.MarginBased) = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelEnc.Indices)     = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelEnc.OneOfK)      = throw(MethodError(isposlabel,(value,)))
isposlabel(value::T, ::LabelEnc.FuzzyBinary) where {T<:Number} = (value > T(0))
isposlabel(value::T, zo::LabelEnc.ZeroOne)   where {T<:Number} = (value >= zo.cutoff)
isposlabel(value::T, ::LabelEnc.MarginBased) where {T<:Number} = (sign(value) >= T(0))
isposlabel(value::Number, lm::LabelEnc.Indices{T,2}) where {T} = value == poslabel(lm)
isposlabel(value::Number, lm::LabelEnc.OneOfK{T,2})  where {T} = value == T(1)
isposlabel(value::AbstractVector{<:Number}, lm::LabelEnc.OneOfK{T,2}) where {T} = indmax(value) == 1
isposlabel(value, lm::LabelEnc.NativeLabels{T,2}) where {T} = standardize_label(value, lm) == poslabel(lm)

# What it means to be a negative label
isneglabel(value, ::LabelEnc.FuzzyBinary) = _ambiguous()
isneglabel(value, ovr::LabelEnc.OneVsRest) = (value != ovr.poslabel)
isneglabel(value::Bool, ::LabelEnc.FuzzyBinary) = !value
isneglabel(value::Bool, ::LabelEnc.TrueFalse)   = !value
isneglabel(value::Bool, ::LabelEnc.MarginBased) = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelEnc.Indices)     = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelEnc.OneOfK)      = throw(MethodError(isneglabel,(value,)))
isneglabel(value::T, ::LabelEnc.FuzzyBinary) where {T<:Number} = (value <= T(0))
isneglabel(value::T, zo::LabelEnc.ZeroOne)   where {T<:Number} = (value < zo.cutoff)
isneglabel(value::T, ::LabelEnc.MarginBased) where {T<:Number} = (sign(value) == -1)
isneglabel(value::Number, lm::LabelEnc.Indices{T,2}) where {T} = value == neglabel(lm)
isneglabel(value::Number, lm::LabelEnc.OneOfK{T,2})  where {T} = value == T(2)
isneglabel(value::AbstractVector{<:Number}, lm::LabelEnc.OneOfK{T,2}) where {T} = indmax(value) == 2
isneglabel(value, lm::LabelEnc.NativeLabels{T,2}) where {T} = standardize_label(value, lm) == neglabel(lm)

# Check if the encoding is appropriate
islabelenc(targets::AbstractArray, args...) = false
islabelenc(targets::AbstractVector{T}, ::LabelEnc.FuzzyBinary)       where {T<:Union{Number,Bool}} = true
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.FuzzyBinary}) where {T<:Union{Number,Bool}} = true
islabelenc(targets::AbstractVector{T}, ::LabelEnc.ZeroOne{T})        where {T<:Union{Number,Bool}} = all(x == 0 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.ZeroOne{T}})  where {T<:Union{Number,Bool}} = all(x == 0 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.ZeroOne})     where {T<:Union{Number,Bool}} = all(x == 0 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{T}, ::LabelEnc.MarginBased{T})       where {T<:Number} = all(x == -1 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.MarginBased{T}}) where {T<:Number} = all(x == -1 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.MarginBased})    where {T<:Number} = all(x == -1 || x == 1 for x in targets)
islabelenc(targets::AbstractVector{Bool}, ::LabelEnc.TrueFalse) = true
islabelenc(targets::AbstractVector{Bool}, ::Type{LabelEnc.TrueFalse}) = true
islabelenc(targets::AbstractVector{T}, lm::LabelEnc.OneVsRest{T})   where {T}           = any(x == lm.poslabel for x in targets)
islabelenc(targets::AbstractVector{T}, ::LabelEnc.Indices{T,K})     where {T<:Number,K} = all(0 < x <= T(K) && isinteger(x) for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.Indices{T}}) where {T<:Number}   = all(0 < x && isinteger(x) for x in targets)
islabelenc(targets::AbstractVector{T}, ::Type{LabelEnc.Indices})    where {T<:Number}   = all(0 < x && isinteger(x) for x in targets)

function islabelenc(targets::AbstractVector, lm::LabelEnc.NativeLabels; strict=true)
    if strict
        all(haskey(lm.invlabel, x) for x in targets)
    else
        all(haskey(lm.invlabel, standardize_label(x, lm)) for x in targets)
    end
end

function islabelenc(targets::AbstractMatrix{<:Union{Bool,Number}}, lm; obsdim = LearnBase.default_obsdim(targets))
    islabelenc(targets, lm, convert(LearnBase.ObsDimension,obsdim))
end

function islabelenc(targets::AbstractMatrix{<:Union{Bool,Number}}, lm::LabelEnc.OneOfK, obsdim)
    islabelenc(targets, typeof(lm), obsdim)
end

function islabelenc(targets::AbstractMatrix{T}, ::Type{LabelEnc.OneOfK{T}}, obsdim) where {T<:Union{Bool,Number}}
    islabelenc(targets, LabelEnc.OneOfK, obsdim)
end

function islabelenc(targets::AbstractMatrix{T}, ::Type{LabelEnc.OneOfK{T,K}}, ::Union{ObsDim.Last,ObsDim.Constant{2}}) where {T<:Union{Bool,Number},K}
    k, n = size(targets)
    ifelse(k != K, false, islabelenc(targets, LabelEnc.OneOfK, ObsDim.Last()))
end

function islabelenc(targets::AbstractMatrix{T}, ::Type{LabelEnc.OneOfK{T,K}}, ::ObsDim.First) where {T<:Union{Bool,Number},K}
    n, k = size(targets)
    ifelse(k != K, false, islabelenc(targets, LabelEnc.OneOfK, ObsDim.First()))
end

function islabelenc(targets::AbstractMatrix{T}, ::Type{LabelEnc.OneOfK}, ::Union{ObsDim.Last,ObsDim.Constant{2}}) where {T<:Union{Bool,Number}}
    k, n = size(targets)
    @inbounds for i in 1:n
        found = false
        for j in 1:k
            tcur = targets[j,i]
            if tcur == T(1)
                if found
                    return false
                end
                found = true
            elseif tcur == T(0)
                # this is fine
            else
                return false
            end
        end
        if !found
            return false
        end
    end
    return true
end

function islabelenc(targets::AbstractMatrix{T}, ::Type{LabelEnc.OneOfK}, ::ObsDim.First) where {T<:Union{Bool,Number}}
    n, k = size(targets)
    @inbounds for i in 1:n
        found = false
        for j in 1:k
            tcur = targets[i,j]
            if tcur == T(1)
                if found
                    return false
                end
                found = true
            elseif tcur == T(0)
                # this is fine
            else
                return false
            end
        end
        if !found
            return false
        end
    end
    return true
end

# Automatic determination of label mode
labelenc(target) = _ambiguous()
labelenc(targets::AbstractVector{Bool}) = LabelEnc.TrueFalse()

function labelenc(targets::AbstractVector)
    lbls = label(targets)
    LabelEnc.NativeLabels(lbls)
end

function labelenc(targets::AbstractVector{T}) where {T<:Number}
    lbls = label(targets)
    if length(lbls) == 1
        if lbls[1] == 0
            LabelEnc.ZeroOne(T)
        elseif lbls[1] == -1
            LabelEnc.MarginBased(T)
        else
            _ambiguous() # could be ZeroOne or MarginBased
        end
    elseif length(lbls) == 2
        mi, ma = extrema(lbls)
        if mi == 0 && ma == 1
            LabelEnc.ZeroOne(T)
        elseif mi == -1 && ma == 1
            LabelEnc.MarginBased(T)
        elseif mi == 1 && ma == 2
            LabelEnc.Indices(T, Val{2})
        else
            LabelEnc.NativeLabels([ma, mi])
        end
    elseif all(x > 0 && isinteger(x) for x in lbls)
        LabelEnc.Indices(T, maximum(lbls))
    elseif length(lbls) > 10 && any(!isinteger(x) for x in lbls)
        warn("The number of distinct floating point numbers (including at least one that is non-integer!) in the label vector is quite large. Are you sure you want to perform classification and not regression?")
        LabelEnc.NativeLabels(lbls)
    else
        LabelEnc.NativeLabels(lbls)
    end
end

# TODO: Multilabel (Matrix as targets)
function labelenc(targets::AbstractMatrix{<:Number}; obsdim = LearnBase.default_obsdim(targets))
    labelenc(targets, convert(LearnBase.ObsDimension,obsdim))
end

function labelenc(targets::AbstractMatrix{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}}) where {T<:Number}
    LabelEnc.OneOfK(T,size(targets,1))
end

function labelenc(targets::AbstractMatrix{T}, ::ObsDim.First) where {T<:Number}
    LabelEnc.OneOfK(T,size(targets,2))
end
