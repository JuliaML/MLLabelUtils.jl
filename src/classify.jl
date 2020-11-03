## ZeroOne

function classify(value::T, cutoff::Number) where {T<:Number}
    value >= cutoff ? T(1) : T(0)
end

function classify(value::Number, ::Type{LabelEnc.ZeroOne})
    classify(value, 0.5)
end

function classify(value::Number, lm::LabelEnc.ZeroOne{R}) where {R}
    R(classify(value, lm.cutoff))
end

## Margin

_sign(value::T) where {T} = ifelse(signbit(value), T(-1), T(1))::T

function classify(value::Number, ::Type{LabelEnc.MarginBased})
    _sign(value)
end

function classify(value::Number, lm::LabelEnc.MarginBased{R}) where {R}
    R(_sign(value))
end

## broadcast

function classify!(buffer::AbstractVector, values::AbstractVector, lm)
    buffer .= classify.(values, lm)
    buffer
end

function classify(values::AbstractVector{T}, cutoff::Number) where {T}
    classify.(values, cutoff)::Vector{T}
end

for KIND in (:(LabelEnc.MarginBased), :(LabelEnc.ZeroOne))
    @eval begin
        function classify(values::AbstractVector{T}, ::Type{($KIND)}) where {T}
            classify.(values, ($KIND))::Vector{T}
        end
        function classify(values::AbstractVector{T}, lm::L) where {T,L<:($KIND)}
            classify.(values, lm)::Vector{labeltype(L)}
        end
    end
end

## OneOfK

function classify(values::AbstractVector, ::Type{<:LabelEnc.OneOfK})
    argmax(values)
end

function classify(values::AbstractVector, lm::LabelEnc.OneOfK)
    classify(values, typeof(lm))
end

function classify!(buffer::T,
                   values::AbstractMatrix,
                   lm;
                   obsdim = LearnBase.default_obsdim(values)
                  ) where {T<:AbstractVector}
    classify!(buffer, values, lm, Val(obsdim))::T
end

function classify(values::AbstractMatrix,
                  lm;
                  obsdim = LearnBase.default_obsdim(values))
    classify(values, lm, Val(obsdim))
end

function classify!(buffer,
                   values::AbstractMatrix,
                   lm::LabelEnc.OneOfK,
                   obsdim)
    classify!(buffer, values, typeof(lm), obsdim)
end

function classify(values::AbstractMatrix,
                  lm::LabelEnc.OneOfK,
                  obsdim)
    classify(values, typeof(lm), obsdim)
end

function classify(values::AbstractMatrix,
                  ::Type{T},
                  ::Val{2}
                 ) where {T<:LabelEnc.OneOfK}
    K, N = size(values)
    buffer = Vector{Int}(undef, N)
    classify!(buffer, values, T, Val(2))
end

function classify!(buffer::AbstractVector,
                   values::AbstractMatrix{R},
                   ::Type{<:LabelEnc.OneOfK},
                   ::Val{2}
                  ) where {R<:Number}
    K, N = size(values)
    @assert length(buffer) == N
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
        buffer[n] = imax
    end
    buffer
end

function classify(values::AbstractMatrix,
                  ::Type{T},
                  ::Val{1}
                 ) where {T<:LabelEnc.OneOfK}
    N, K = size(values)
    buffer = Vector{Int}(undef, N)
    classify!(buffer, values, T, Val(1))
end

function classify!(buffer::AbstractVector,
                   values::AbstractMatrix{R},
                   ::Type{<:LabelEnc.OneOfK},
                   ::Val{1}
                  ) where {R<:Number}
    N, K = size(values)
    tmax = fill(typemin(R),N)
    @assert length(buffer) == N
    @inbounds for k in 1:K
        for n in 1:N
            tcur = values[n,k]
            if tcur > tmax[n]
                tmax[n] = tcur
                buffer[n] = k
            end
        end
    end
    buffer
end
