"""
TODO
"""
module LabelModes
    import ..MLLabelUtils.LabelMode
    import ..MLLabelUtils.BinaryLabelMode

    # Binary label modes
    immutable FuzzyBinary  <: BinaryLabelMode end
    immutable TrueFalse    <: BinaryLabelMode end
    immutable ZeroOne      <: BinaryLabelMode end
    immutable MarginBased  <: BinaryLabelMode end
    immutable OneVsRest{T} <: BinaryLabelMode
        pos_label::T
    end

    # Multiclass label modes
    immutable Indices{K} <: LabelMode{K}
        Indices() = typeof(K) <: Number && isinteger(K) ? new() : error("Invalid K=$K. It has to be a number and equal to an integer")
    end
    Indices{K}(::Type{Val{K}}) = Indices{K}()
    Indices(K::Real) = Indices(Val{K})

    for KIND in (:NativeLabels, :OneOfK)
        @eval begin
            immutable ($KIND){T,K} <: LabelMode{K}
                labels::Vector{T}
                invlabels::Dict{T,Int}
                function ($KIND)(labels::Vector{T})
                    @assert typeof(K) <: Int
                    @assert length(labels) == length(unique(labels)) == K
                    new(labels, Dict(zip(labels,1:K)))
                end
            end
            ($KIND){T,K}(labels::Vector{T}, ::Type{Val{K}}) = $KIND{T,K}(labels)
            ($KIND)(labels::Vector) = $KIND(labels, Val{length(labels)})
        end
    end

end # submodule

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelMode manually."))

# Automatic determination of label mode
labelmode(target) = _ambiguous()
labelmode(target::Bool) = LabelModes.TrueFalse()
labelmode(targets::AbstractVector{Bool}) = LabelModes.TrueFalse()

function labelmode(targets::AbstractVector)
    labels = unique(targets)
    if 1 <= length(labels) <= 2
        LabelModes.OneVsRest(labels[1])
    else
        LabelModes.NativeLabels(labels)
    end
end

function labelmode{T<:Real}(targets::AbstractVector{T})
    labels = unique(targets)
    if length(labels) == 1
        if labels[1] == 0
            LabelModes.ZeroOne()
        elseif labels[1] == -1
            LabelModes.MarginBased()
        else
            _ambiguous() # could be ZeroOne or MarginBased
        end
    elseif length(labels) == 2
        if minimum(labels) == -1 && maximum(labels) == 1
            LabelModes.MarginBased()
        elseif minimum(labels) == 0 && maximum(labels) == 1
            LabelModes.ZeroOne()
        else
            LabelModes.OneVsRest(maximum(labels))
        end
    elseif all(x > 0 && isinteger(x) for x in labels)
        LabelModes.Indices(maximum(labels))
    elseif length(labels) > 10 && any(!isinteger(x) for x in labels)
        warn("The number of distinct non-integer elements in the label vextor is quite large. Are you sure you want to do classification and not regression?")
        LabelModes.NativeLabels(labels)
    else
        LabelModes.NativeLabels(labels)
    end
end

# TODO: Multilabel (Matrix as targets)

# What it means to be a positive label
@inline isposlabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
@inline isposlabel(value::Bool, ::LabelModes.FuzzyBinary) = value
@inline isposlabel(value::Bool, ::LabelModes.TrueFalse)   = value
@inline isposlabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value > zero(T))
@inline isposlabel{T<:Number}(value::T, ::LabelModes.ZeroOne)      = (value == one(T))
@inline isposlabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) == one(T))
@inline isposlabel{T<:Number}(value::T, ovr::LabelModes.OneVsRest) = (value == ovr.pos_label)

# What it means to be a negative label
@inline isneglabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
@inline isneglabel(value::Bool, ::LabelModes.FuzzyBinary) = !value
@inline isneglabel(value::Bool, ::LabelModes.TrueFalse)   = !value
@inline isneglabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value <= zero(T))
@inline isneglabel{T<:Number}(value::T, ::LabelModes.ZeroOne)      = (value == zero(T))
@inline isneglabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) == -one(T))
@inline isneglabel{T<:Number}(value::T, ovr::LabelModes.OneVsRest) = (value != ovr.pos_label)

