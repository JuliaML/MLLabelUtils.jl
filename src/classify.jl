## ZeroOne

function classify{T<:Number}(value::T, cutoff::Number)
    value >= cutoff ? one(T) : zero(T)
end

function classify{T<:Number}(value::T, ::Type{LabelModes.ZeroOne})
    classify(value, 0.5)
end

function classify{T<:Number,R}(value::T, ::Type{LabelModes.ZeroOne{R}})
    R(classify(value, 0.5))
end

function classify{T<:Number,R}(value::T, lm::LabelModes.ZeroOne{R})
    R(classify(value, lm.cutoff))
end

## Margin

_sign{T}(value::T)::T = signbit(value) ? -one(T) : one(T)

function classify{T<:Number}(value::T, ::Type{LabelModes.MarginBased})
    _sign(value)
end

function classify{T<:Number,R}(value::T, ::Type{LabelModes.MarginBased{R}})
    R(_sign(value))
end

function classify{T<:Number,R}(value::T, lm::LabelModes.MarginBased{R})
    R(_sign(value))
end

## broadcast

function classify{T}(values::AbstractVector{T}, cutoff::Number)
    classify.(values, number)::Vector{T}
end

for KIND in (:(LabelModes.MarginBased), :(LabelModes.ZeroOne))
    @eval begin
        function classify{T}(values::AbstractVector{T}, ::Type{($KIND)})
            classify.(values, ($KIND))::Vector{T}
        end
        function classify{T,R}(values::AbstractVector{T}, ::Type{($KIND){R}})
            classify.(values, ($KIND){R})::Vector{labeltype(($KIND)(R))}
        end
        function classify{T,L<:($KIND)}(values::AbstractVector{T}, lm::L)
            classify.(values, lm)::Vector{labeltype(L)}
        end
    end
end

## OneOfK

function classify{T<:LabelModes.OneOfK}(values::AbstractVector, ::Type{T})
    indmax(values)
end

function classify(values::AbstractVector, lm::LabelModes.OneOfK)
    classify(values, typeof(lm))
end

function classify{T<:LabelModes.OneOfK}(values::AbstractMatrix, ::Type{T}; obsdim = LearnBase.default_obsdim(values))
    classify(values, T, LearnBase.obs_dim(obsdim))
end

function classify(values::AbstractMatrix, lm::LabelModes.OneOfK; obsdim = LearnBase.default_obsdim(values))
    classify(values, typeof(lm), LearnBase.obs_dim(obsdim))
end

function classify{R<:Number,T<:LabelModes.OneOfK}(values::AbstractMatrix{R}, ::Type{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    K, N = size(values)
    res = Vector{Int}(N)
    @inbounds for n in 1:N
        imax = 0
        tmax = typemin(R)
        for k in 1:K
            tcur = values[k,n]
            if tcur > tmax
                imax = k
                tmax = tcur
            end
        end
        res[n] = imax
    end
    res
end

function classify{R<:Number,T<:LabelModes.OneOfK}(values::AbstractMatrix{R}, ::Type{T}, ::ObsDim.First)
    N, K = size(values)
    tmax = fill(typemin(R),N)
    res  = Vector{Int}(N)
    @inbounds for k in 1:K
        for n in 1:N
            tcur = values[n,k]
            if tcur > tmax[n]
                tmax[n] = tcur
                res[n] = k
            end
        end
    end
    res
end

function classify(values::AbstractMatrix, lm::LabelModes.OneOfK, obsdim::LearnBase.ObsDimension)
    classify(values, typeof(lm), obsdim)
end

