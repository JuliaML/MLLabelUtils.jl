"""
TODO
"""
module LabelModes
    import ..MLLabelUtils.LabelMode
    import ..MLLabelUtils.BinaryLabelMode

    """
    TODO
    """
    immutable FuzzyBinary <: BinaryLabelMode{Any,1} end

    """
    TODO
    """
    immutable TrueFalse <: BinaryLabelMode{Bool,1} end

    """
    TODO
    """
    immutable ZeroOne{T<:Number,R<:Number} <: BinaryLabelMode{T,1}
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
    immutable MarginBased{T<:Number} <: BinaryLabelMode{T,1} end
    MarginBased{T<:Number}(::Type{T} = Float64) = MarginBased{T}()

    """
    TODO
    """
    immutable OneVsRest{T} <: BinaryLabelMode{T,1}
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
    immutable Indices{T<:Number,K} <: LabelMode{T,K,1}
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
    immutable OneOfK{T<:Number,K} <: LabelMode{T,K,2}
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
    immutable NativeLabels{T,K} <: LabelMode{T,K,1}
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

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelMode manually."))

# Query the index
label2ind(lbl, lm::MLLabelUtils.BinaryLabelMode) = isposlabel(lbl, lm) ? 1 : 2
label2ind{T}(lbl::T, lm::LabelModes.NativeLabels{T}) = Int(lm.invlabel[lbl])
label2ind{T}(lbl::Union{Number,T}, lm::LabelModes.Indices{T}) = Int(lbl)
label2ind{T}(lbl::Union{Number,T}, lm::LabelModes.OneOfK{T}) = Int(lbl)
label2ind(lbl::AbstractVector, lm::LabelModes.OneOfK) = indmax(lbl)

# Query the label
ind2label(i::Int, lm::MLLabelUtils.BinaryLabelMode) = i == 1 ? ind2label(Val{1},lm) : ind2label(Val{2},lm)
ind2label{T}(i::Int, ::LabelModes.Indices{T}) = T(i)
ind2label{T}(i::Int, ::LabelModes.OneOfK{T}) = Int(i)
ind2label{T}(i::Int, lm::LabelModes.NativeLabels{T}) = lm.label[i]

ind2label(::Type{Val{1}}, ::LabelModes.TrueFalse) = true
ind2label(::Type{Val{2}}, ::LabelModes.TrueFalse) = false
ind2label{T}(::Type{Val{1}}, ::LabelModes.ZeroOne{T}) = one(T)
ind2label{T}(::Type{Val{2}}, ::LabelModes.ZeroOne{T}) = zero(T)
ind2label{T}(::Type{Val{1}}, ::LabelModes.MarginBased{T}) = one(T)
ind2label{T}(::Type{Val{2}}, ::LabelModes.MarginBased{T}) = -one(T)
ind2label(::Type{Val{1}}, ovr::LabelModes.OneVsRest) = ovr.poslabel
ind2label(::Type{Val{2}}, ovr::LabelModes.OneVsRest) = ovr.neglabel
ind2label{T,K}(::Type{Val{K}}, ::LabelModes.Indices{T}) = T(K)
ind2label{T,K}(::Type{Val{K}}, ::LabelModes.OneOfK{T}) = Int(K)
ind2label{T,K}(::Type{Val{K}}, lm::LabelModes.NativeLabels{T}) = lm.label[K]

poslabel(lm::LabelModes.BinaryLabelMode) = ind2label(Val{1}, lm)
neglabel(lm::LabelModes.BinaryLabelMode) = ind2label(Val{2}, lm)

label(lm::LabelModes.BinaryLabelMode) = [poslabel(lm), neglabel(lm)]
label{T,K}(::LabelModes.Indices{T,K}) = collect(one(T):T(K))
label{T,K}(::LabelModes.OneOfK{T,K})  = collect(1:K)
label(lm::LabelModes.NativeLabels) = lm.label

# What it means to be a positive label
isposlabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
isposlabel(value, ovr::LabelModes.OneVsRest) = (value == ovr.poslabel)
isposlabel(value::Bool, ::LabelModes.FuzzyBinary) = value
isposlabel(value::Bool, ::LabelModes.TrueFalse)   = value
isposlabel(value::Bool, ::LabelModes.MarginBased) = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelModes.Indices)     = throw(MethodError(isposlabel,(value,)))
isposlabel(value::Bool, ::LabelModes.OneOfK)      = throw(MethodError(isposlabel,(value,)))
isposlabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value > zero(T))
isposlabel{T<:Number}(value::T, zo::LabelModes.ZeroOne)    = (value >= zo.cutoff)
isposlabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) >= zero(T))
isposlabel{T}(value::Number, lm::LabelModes.Indices{T,2})  = value == poslabel(lm)
isposlabel{T}(value::Number, lm::LabelModes.OneOfK{T,2})   = value == poslabel(lm)
isposlabel{R<:Number,T}(value::AbstractVector{R}, lm::LabelModes.OneOfK{T,2}) = indmax(value) == poslabel(lm)
isposlabel{T}(value, lm::LabelModes.NativeLabels{T,2})     = value == poslabel(lm)

# What it means to be a negative label
isneglabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
isneglabel(value, ovr::LabelModes.OneVsRest) = (value != ovr.poslabel)
isneglabel(value::Bool, ::LabelModes.FuzzyBinary) = !value
isneglabel(value::Bool, ::LabelModes.TrueFalse)   = !value
isneglabel(value::Bool, ::LabelModes.MarginBased) = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelModes.Indices)     = throw(MethodError(isneglabel,(value,)))
isneglabel(value::Bool, ::LabelModes.OneOfK)      = throw(MethodError(isneglabel,(value,)))
isneglabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value <= zero(T))
isneglabel{T<:Number}(value::T, zo::LabelModes.ZeroOne)    = (value < zo.cutoff)
isneglabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) == -one(T))
isneglabel{T}(value::Number, lm::LabelModes.Indices{T,2})  = value == neglabel(lm)
isneglabel{T}(value::Number, lm::LabelModes.OneOfK{T,2})   = value == neglabel(lm)
isneglabel{R<:Number,T}(value::AbstractVector{R}, lm::LabelModes.OneOfK{T,2}) = indmax(value) == neglabel(lm)
isneglabel{T}(value, lm::LabelModes.NativeLabels{T,2})     = value == neglabel(lm)

# Automatic determination of label mode
labelmode(target) = _ambiguous()
labelmode(targets::AbstractVector{Bool}) = LabelModes.TrueFalse()

function labelmode(targets::AbstractVector)
    lbls = label(targets)
    LabelModes.NativeLabels(lbls)
end

function labelmode{T<:Number}(targets::AbstractVector{T})
    lbls = label(targets)
    if length(lbls) == 1
        if lbls[1] == 0
            LabelModes.ZeroOne(T)
        elseif lbls[1] == -1
            LabelModes.MarginBased(T)
        else
            _ambiguous() # could be ZeroOne or MarginBased
        end
    elseif length(lbls) == 2
        mi, ma = extrema(lbls)
        if mi == 0 && ma == 1
            LabelModes.ZeroOne(T)
        elseif mi == -1 && ma == 1
            LabelModes.MarginBased(T)
        elseif mi == 1 && ma == 2
            LabelModes.Indices(T, Val{2})
        else
            LabelModes.NativeLabels([ma, mi])
        end
    elseif all(x > 0 && isinteger(x) for x in lbls)
        LabelModes.Indices(T, maximum(lbls))
    elseif length(lbls) > 10 && any(!isinteger(x) for x in lbls)
        warn("The number of distinct floating point numbers (including at least one that is non-integer!) in the label vector is quite large. Are you sure you want to perform classification and not regression?")
        LabelModes.NativeLabels(lbls)
    else
        LabelModes.NativeLabels(lbls)
    end
end

# TODO: Multilabel (Matrix as targets)
function labelmode{T<:Number}(targets::AbstractMatrix{T}; obsdim = LearnBase.default_obsdim(targets))
    labelmode(targets, LearnBase.obs_dim(obsdim))
end

function labelmode{T<:Number}(targets::AbstractMatrix{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    LabelModes.OneOfK(T,size(targets,1))
end

function labelmode{T<:Number}(targets::AbstractMatrix{T}, ::ObsDim.First)
    LabelModes.OneOfK(T,size(targets,2))
end

