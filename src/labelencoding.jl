"""
TODO
"""
module LabelEnc
    import ..MLLabelUtils.LabelEncoding
    import ..MLLabelUtils.BinaryLabelEncoding

    """
    TODO
    """
    immutable FuzzyBinary <: BinaryLabelEncoding{Any,1} end

    """
    TODO
    """
    immutable TrueFalse <: BinaryLabelEncoding{Bool,1} end

    """
    TODO
    """
    immutable ZeroOne{T<:Number,R<:Number} <: BinaryLabelEncoding{T,1}
        cutoff::R
        function ZeroOne(cutoff::R = R(0.5))
            @assert 0 <= cutoff <= 1
            new(cutoff)
        end
    end
    ZeroOne{T<:Number,R<:Number}(t::Type{T} = Float64, cutoff::R = 0.5) = ZeroOne{T,R}(cutoff)
    ZeroOne{R<:Number}(cutoff::R) = ZeroOne(R, cutoff)

    """
    TODO
    """
    immutable MarginBased{T<:Number} <: BinaryLabelEncoding{T,1} end
    MarginBased{T<:Number}(::Type{T} = Float64) = MarginBased{T}()

    """
    TODO
    """
    immutable OneVsRest{T} <: BinaryLabelEncoding{T,1}
        poslabel::T
        neglabel::T
    end
    OneVsRest(poslabel::Bool) = OneVsRest{Bool}(poslabel, !poslabel)
    OneVsRest(poslabel::String) = OneVsRest{String}(poslabel, "not_$(poslabel)")
    OneVsRest(poslabel::Symbol) = OneVsRest{Symbol}(poslabel, Symbol(:not_, poslabel))
    OneVsRest{T<:Number}(poslabel::T) = OneVsRest{T}(poslabel, poslabel == 0 ? poslabel+one(T) : zero(T))

    """
    TODO
    """
    immutable Indices{T<:Number,K} <: LabelEncoding{T,K,1}
        function Indices()
            typeof(K) <: Int || throw(TypeError(:Indices,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            new()
        end
    end
    Indices{T,K}(::Type{T}, ::Type{Val{K}}) = Indices{T,K}()
    Indices{K}(::Type{Val{K}}) = Indices(Int,Val{K})
    Indices{T}(::Type{T}, K::Number) = Indices(T,Val{Int(K)})
    Indices(K::Number) = Indices(Int,Val{Int(K)})

    """
    TODO
    """
    immutable OneOfK{T<:Number,K} <: LabelEncoding{T,K,2}
        function OneOfK()
            typeof(K) <: Int || throw(TypeError(:OneOfK,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            new()
        end
    end
    OneOfK{T,K}(::Type{T}, ::Type{Val{K}}) = OneOfK{T,K}()
    OneOfK{K}(::Type{Val{K}}) = OneOfK(Int,Val{K})
    OneOfK{T}(::Type{T}, K::Number) = OneOfK(T,Val{Int(K)})
    OneOfK(K::Number) = OneOfK(Int,Val{Int(K)})


    """
    TODO
    """
    immutable NativeLabels{T,K} <: LabelEncoding{T,K,1}
        label::Vector{T}
        invlabel::Dict{T,Int}
        function NativeLabels(label::Vector{T})
            typeof(K) <: Int || throw(TypeError(:NativeLabels,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            @assert length(label) == length(unique(label)) == K
            new(label, Dict(zip(label,1:K)))
        end
    end
    NativeLabels{T,K}(label::Vector{T}, ::Type{Val{K}}) = NativeLabels{T,K}(label)
    NativeLabels(label::Vector) = NativeLabels(label, Val{length(label)})

end # submodule

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelEncoding manually."))

# Query the index
label2ind(lbl, lm::BinaryLabelEncoding) = isposlabel(lbl, lm) ? 1 : 2
label2ind{T}(lbl::T, lm::LabelEnc.NativeLabels{T}) = Int(lm.invlabel[lbl])
label2ind{T}(lbl::Union{Number,T}, lm::LabelEnc.Indices{T}) = Int(lbl)
label2ind{T}(lbl::Union{Number,T}, lm::LabelEnc.OneOfK{T}) = Int(lbl)
label2ind(lbl::AbstractVector, lm::LabelEnc.OneOfK) = indmax(lbl)

# Query the label
ind2label(i::Int, lm::BinaryLabelEncoding) = i == 1 ? ind2label(Val{1},lm) : ind2label(Val{2},lm)
ind2label{T}(i::Int, ::LabelEnc.Indices{T}) = T(i)
ind2label{T,K}(i::Int, ::LabelEnc.OneOfK{T,K}) = (x = zeros(T,K); x[i] = one(T); x)
ind2label{T}(i::Int, lm::LabelEnc.NativeLabels{T}) = lm.label[i]

ind2label(::Type{Val{1}}, ::LabelEnc.TrueFalse) = true
ind2label(::Type{Val{2}}, ::LabelEnc.TrueFalse) = false
ind2label{T}(::Type{Val{1}}, ::LabelEnc.ZeroOne{T}) = one(T)
ind2label{T}(::Type{Val{2}}, ::LabelEnc.ZeroOne{T}) = zero(T)
ind2label{T}(::Type{Val{1}}, ::LabelEnc.MarginBased{T}) = one(T)
ind2label{T}(::Type{Val{2}}, ::LabelEnc.MarginBased{T}) = -one(T)
ind2label(::Type{Val{1}}, ovr::LabelEnc.OneVsRest) = ovr.poslabel
ind2label(::Type{Val{2}}, ovr::LabelEnc.OneVsRest) = ovr.neglabel
ind2label{T,K}(::Type{Val{K}}, ::LabelEnc.Indices{T}) = T(K)
ind2label{T,I,K}(::Type{Val{I}}, ::LabelEnc.OneOfK{T,K}) = (x = zeros(T,K); x[I] = one(T); x)
ind2label{T,K}(::Type{Val{K}}, lm::LabelEnc.NativeLabels{T}) = lm.label[K]

poslabel(lm::LabelEnc.BinaryLabelEncoding) = ind2label(Val{1}, lm)
neglabel(lm::LabelEnc.BinaryLabelEncoding) = ind2label(Val{2}, lm)

label{T}(lm::LabelEnc.BinaryLabelEncoding{T,1}) = [poslabel(lm), neglabel(lm)]
label{T,K}(::LabelEnc.Indices{T,K}) = collect(one(T):T(K))
label{T,K}(::LabelEnc.OneOfK{T,K})  = collect(1:K)
label(lm::LabelEnc.NativeLabels) = lm.label

# What it means to be a positive label
isposlabel(value, ::LabelEnc.FuzzyBinary) = _ambiguous()
isposlabel(value, ovr::LabelEnc.OneVsRest) = (value == ovr.poslabel)
isposlabel(value::Bool, ::LabelEnc.FuzzyBinary) = value
isposlabel(value::Bool, ::LabelEnc.TrueFalse)   = value
isposlabel(value::Bool, ::LabelEnc.MarginBased) = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelEnc.Indices)     = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelEnc.OneOfK)      = throw(MethodError(isposlabel,(value,)))
isposlabel{T<:Number}(value::T, ::LabelEnc.FuzzyBinary)  = (value > zero(T))
isposlabel{T<:Number}(value::T, zo::LabelEnc.ZeroOne)    = (value >= zo.cutoff)
isposlabel{T<:Number}(value::T, ::LabelEnc.MarginBased)  = (sign(value) >= zero(T))
isposlabel{T}(value::Number, lm::LabelEnc.Indices{T,2})  = value == poslabel(lm)
isposlabel{T}(value::Number, lm::LabelEnc.OneOfK{T,2})   = value == 1
isposlabel{R<:Number,T}(value::AbstractVector{R}, lm::LabelEnc.OneOfK{T,2}) = indmax(value) == 1
isposlabel{T}(value, lm::LabelEnc.NativeLabels{T,2})     = value == poslabel(lm)

# What it means to be a negative label
isneglabel(value, ::LabelEnc.FuzzyBinary) = _ambiguous()
isneglabel(value, ovr::LabelEnc.OneVsRest) = (value != ovr.poslabel)
isneglabel(value::Bool, ::LabelEnc.FuzzyBinary) = !value
isneglabel(value::Bool, ::LabelEnc.TrueFalse)   = !value
isneglabel(value::Bool, ::LabelEnc.MarginBased) = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelEnc.Indices)     = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelEnc.OneOfK)      = throw(MethodError(isneglabel,(value,)))
isneglabel{T<:Number}(value::T, ::LabelEnc.FuzzyBinary)  = (value <= zero(T))
isneglabel{T<:Number}(value::T, zo::LabelEnc.ZeroOne)    = (value < zo.cutoff)
isneglabel{T<:Number}(value::T, ::LabelEnc.MarginBased)  = (sign(value) == -one(T))
isneglabel{T}(value::Number, lm::LabelEnc.Indices{T,2})  = value == neglabel(lm)
isneglabel{T}(value::Number, lm::LabelEnc.OneOfK{T,2})   = value == 2
isneglabel{R<:Number,T}(value::AbstractVector{R}, lm::LabelEnc.OneOfK{T,2}) = indmax(value) == 2
isneglabel{T}(value, lm::LabelEnc.NativeLabels{T,2})     = value == neglabel(lm)

# Automatic determination of label mode
labelenc(target) = _ambiguous()
labelenc(targets::AbstractVector{Bool}) = LabelEnc.TrueFalse()

function labelenc(targets::AbstractVector)
    lbls = label(targets)
    LabelEnc.NativeLabels(lbls)
end

function labelenc{T<:Number}(targets::AbstractVector{T})
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
function labelenc{T<:Number}(targets::AbstractMatrix{T}; obsdim = LearnBase.default_obsdim(targets))
    labelenc(targets, LearnBase.obs_dim(obsdim))
end

function labelenc{T<:Number}(targets::AbstractMatrix{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    LabelEnc.OneOfK(T,size(targets,1))
end

function labelenc{T<:Number}(targets::AbstractMatrix{T}, ::ObsDim.First)
    LabelEnc.OneOfK(T,size(targets,2))
end

