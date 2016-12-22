"""
TODO
"""
module LabelModes
    import ..MLLabelUtils.LabelMode
    import ..MLLabelUtils.BinaryLabelMode
    import ..MLLabelUtils.labels
    import ..MLLabelUtils.nlabels

    immutable FuzzyBinary  <: BinaryLabelMode end

    immutable TrueFalse    <: BinaryLabelMode end
    labels(::TrueFalse) = [true, false]

    # Binary label modes
    immutable ZeroOne{T<:Number,R<:Number} <: BinaryLabelMode
        cutoff::R
        function ZeroOne(cutoff::R)
            @assert 0 <= cutoff <= 1
            new(cutoff)
        end
    end
    ZeroOne{R<:Number,T<:Number}(cutoff::R = 0.5, t::Type{T} = Int) = ZeroOne{T,R}(cutoff)
    ZeroOne{T<:Number}(::Type{T}) = ZeroOne(0.5, T)
    labels{T}(::ZeroOne{T}) = [one(T), zero(T)]

    immutable MarginBased{T<:Number} <: BinaryLabelMode end
    MarginBased{T<:Number}(::Type{T} = Int) = MarginBased{T}()
    labels{T}(::MarginBased{T}) = [one(T), -one(T)]

    immutable OneVsRest{T} <: BinaryLabelMode
        pos_label::T
    end
    labels(ovr::OneVsRest{Bool}) = [ovr.pos_label, !ovr.pos_label]
    labels{T<:Number}(ovr::OneVsRest{T}) = [ovr.pos_label, ovr.pos_label == 0 ? ovr.pos_label+one(T) : zero(T)]
    labels(ovr::OneVsRest{String}) = [ovr.pos_label, "not_$(ovr.pos_label)"]
    labels(ovr::OneVsRest{Symbol}) = [ovr.pos_label, Symbol(:not_, ovr.pos_label)]

    # Multiclass label modes
    immutable Indices{K} <: LabelMode{K}
        Indices() = typeof(K) <: Number && isinteger(K) ? new() : error("Invalid K=$K. It has to be a number and equal to an integer")
    end
    Indices{K}(::Type{Val{K}}) = Indices{K}()
    Indices(K::Number) = Indices(Val{K})
    labels{K}(::Indices{K}) = collect(one(K):K)

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
            labels(lm::$KIND) = lm.labels
        end
    end

end # submodule

_ambiguous() = throw(ArgumentError("Can't infer the label meaning because argument types or values are ambiguous. Please specify the desired LabelMode manually."))

# Automatic determination of label mode
labelmode(target) = _ambiguous()
labelmode(targets::AbstractVector{Bool}) = LabelModes.TrueFalse()

function labelmode(targets::AbstractVector)
    lbls = unique(targets)
    if 1 <= length(labels) <= 2
        LabelModes.OneVsRest(labels[1])
    else
        LabelModes.NativeLabels(labels)
    end
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
        if minimum(lbls) == -1 && maximum(lbls) == 1
            LabelModes.MarginBased(T)
        elseif minimum(lbls) == 0 && maximum(lbls) == 1
            LabelModes.ZeroOne(T)
        else
            LabelModes.OneVsRest(maximum(lbls))
        end
    elseif all(x > 0 && isinteger(x) for x in lbls)
        LabelModes.Indices(maximum(lbls))
    elseif length(lbls) > 10 && any(!isinteger(x) for x in lbls)
        warn("The number of distinct non-integer elements in the label vextor is quite large. Are you sure you want to do classification and not regression?")
        LabelModes.NativeLabels(lbls)
    else
        LabelModes.NativeLabels(lbls)
    end
end

# TODO: Multilabel (Matrix as targets)

# What it means to be a positive label
isposlabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
isposlabel(value::Bool, ::LabelModes.FuzzyBinary) = value
isposlabel(value::Bool, ::LabelModes.TrueFalse)   = value
isposlabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value > zero(T))
isposlabel{T<:Number}(value::T, zo::LabelModes.ZeroOne)    = (value >= zo.cutoff)
isposlabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) == one(T))
isposlabel{T<:Number}(value::T, ovr::LabelModes.OneVsRest) = (value == ovr.pos_label)

# What it means to be a negative label
isneglabel(value, ::LabelModes.FuzzyBinary) = _ambiguous()
isneglabel(value::Bool, ::LabelModes.FuzzyBinary) = !value
isneglabel(value::Bool, ::LabelModes.TrueFalse)   = !value
isneglabel{T<:Number}(value::T, ::LabelModes.FuzzyBinary)  = (value <= zero(T))
isneglabel{T<:Number}(value::T, zo::LabelModes.ZeroOne)    = (value < zo.cutoff)
isneglabel{T<:Number}(value::T, ::LabelModes.MarginBased)  = (sign(value) == -one(T))
isneglabel{T<:Number}(value::T, ovr::LabelModes.OneVsRest) = (value != ovr.pos_label)

