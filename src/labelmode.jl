"""
TODO
"""
module LabelModes
    import ..MLLabelUtils.LabelMode
    import ..MLLabelUtils.BinaryLabelMode

    """
    TODO
    """
    immutable FuzzyBinary <: BinaryLabelMode end

    """
    TODO
    """
    immutable TrueFalse <: BinaryLabelMode end

    """
    TODO
    """
    immutable ZeroOne{T<:Number,R<:Number} <: BinaryLabelMode
        cutoff::R
        function ZeroOne(cutoff::R)
            @assert 0 <= cutoff <= 1
            new(cutoff)
        end
    end
    ZeroOne{T<:Number,R<:Number}(t::Type{T} = Float64, cutoff::R = 0.5) = ZeroOne{T,R}(cutoff)
    ZeroOne{R<:Number}(cutoff::R) = ZeroOne(R, cutoff)

    """
    TODO
    """
    immutable MarginBased{T<:Number} <: BinaryLabelMode end
    MarginBased{T<:Number}(::Type{T} = Float64) = MarginBased{T}()

    """
    TODO
    """
    immutable OneVsRest{T} <: BinaryLabelMode
        poslabel::T
        neglabel::T
    end
    OneVsRest(poslabel::Bool) = OneVsRest{Bool}(poslabel, !poslabel)
    OneVsRest(poslabel::String) = OneVsRest{String}(poslabel, "not_$(poslabel)")
    OneVsRest(poslabel::Symbol) = OneVsRest{Symbol}(poslabel, Symbol(:not_, poslabel))
    OneVsRest{T<:Number}(poslabel::T) = OneVsRest{T}(poslabel, poslabel == 0 ? poslabel+one(T) : zero(T))

    for KIND in (:Indices, :OneOfK)
        @eval begin
            immutable ($KIND){T<:Number,K} <: LabelMode{K}
                function ($KIND)()
                    typeof(K) <: Int || throw(TypeError(Symbol($(string(KIND))),"constructor when checking typeof(K)",Type{Int},typeof(K)))
                    new()
                end
            end
            ($KIND){T,K}(::Type{T}, ::Type{Val{K}}) = ($KIND){T,K}()
            ($KIND){K}(::Type{Val{K}}) = ($KIND)(Int,Val{K})
            ($KIND){T}(::Type{T}, K::Number) = ($KIND)(T,Val{Int(K)})
            ($KIND)(K::Number) = ($KIND)(Int,Val{Int(K)})
        end
    end

    @doc """
    TODO
    """ ->
    Indices

    @doc """
    TODO
    """ ->
    OneOfK

    """
    TODO
    """
    immutable NativeLabels{T,K} <: LabelMode{K}
        labels::Vector{T}
        invlabels::Dict{T,Int}
        function NativeLabels(labels::Vector{T})
            typeof(K) <: Int || throw(TypeError(:NativeLabels,"constructor when checking typeof(K)",Type{Int},typeof(K)))
            @assert length(labels) == length(unique(labels)) == K
            new(labels, Dict(zip(labels,1:K)))
        end
    end
    NativeLabels{T,K}(labels::Vector{T}, ::Type{Val{K}}) = NativeLabels{T,K}(labels)
    NativeLabels(labels::Vector) = NativeLabels(labels, Val{length(labels)})

end # submodule

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelMode manually."))

# Query the labels
poslabel(::LabelModes.TrueFalse) = true
neglabel(::LabelModes.TrueFalse) = false
poslabel{T}(::LabelModes.ZeroOne{T}) = one(T)
neglabel{T}(::LabelModes.ZeroOne{T}) = zero(T)
poslabel{T}(::LabelModes.MarginBased{T}) = one(T)
neglabel{T}(::LabelModes.MarginBased{T}) = -one(T)
poslabel(ovr::LabelModes.OneVsRest) = ovr.poslabel
neglabel(ovr::LabelModes.OneVsRest) = ovr.neglabel
poslabel{T}(::LabelModes.Indices{T,2}) = T(1)
neglabel{T}(::LabelModes.Indices{T,2}) = T(2)
poslabel{T}(::LabelModes.OneOfK{T,2}) = 1
neglabel{T}(::LabelModes.OneOfK{T,2}) = 2
poslabel{T}(lm::LabelModes.NativeLabels{T,2}) = lm.labels[1]
neglabel{T}(lm::LabelModes.NativeLabels{T,2}) = lm.labels[2]

labels(lm::LabelModes.BinaryLabelMode) = [poslabel(lm), neglabel(lm)]
labels{T,K}(::LabelModes.Indices{T,K}) = collect(one(T):T(K))
labels{T,K}(::LabelModes.OneOfK{T,K})  = collect(1:K)
labels(lm::LabelModes.NativeLabels) = lm.labels

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
    lbls = labels(targets)
    LabelModes.NativeLabels(lbls)
end

function labelmode{T<:Number}(targets::AbstractVector{T})
    lbls = labels(targets)
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

