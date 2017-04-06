## ZeroOne

function classify{T<:Number}(value::T, cutoff::Number)
    value >= cutoff ? one(T) : zero(T)
end

function classify{T<:Number}(value::T, ::Type{LabelEnc.ZeroOne})
    classify(value, 0.5)
end

function classify{T<:Number,R}(value::T, lm::LabelEnc.ZeroOne{R})
    R(classify(value, lm.cutoff))
end

## Margin

_sign{T}(value::T)::T = signbit(value) ? -one(T) : one(T)

function classify{T<:Number}(value::T, ::Type{LabelEnc.MarginBased})
    _sign(value)
end

function classify{T<:Number,R}(value::T, lm::LabelEnc.MarginBased{R})
    R(_sign(value))
end

## broadcast

function classify!(buffer::AbstractVector, values::AbstractVector, lm)
    buffer .= classify.(values, lm)
    buffer
end

function classify{T}(values::AbstractVector{T}, cutoff::Number)
    classify.(values, cutoff)::Vector{T}
end

for KIND in (:(LabelEnc.MarginBased), :(LabelEnc.ZeroOne))
    @eval begin
        function classify{T}(values::AbstractVector{T}, ::Type{($KIND)})
            classify.(values, ($KIND))::Vector{T}
        end
        function classify{T,L<:($KIND)}(values::AbstractVector{T}, lm::L)
            classify.(values, lm)::Vector{labeltype(L)}
        end
    end
end

## OneOfK

function classify{T<:LabelEnc.OneOfK}(values::AbstractVector, ::Type{T})
    indmax(values)
end

function classify(values::AbstractVector, lm::LabelEnc.OneOfK)
    classify(values, typeof(lm))
end

function classify!{T<:AbstractVector}(buffer::T, values::AbstractMatrix, lm; obsdim = LearnBase.default_obsdim(values))
    classify!(buffer, values, lm, convert(LearnBase.ObsDimension,obsdim))::T
end

function classify(values::AbstractMatrix, lm; obsdim = LearnBase.default_obsdim(values))
    classify(values, lm, convert(LearnBase.ObsDimension,obsdim))
end

function classify!(buffer, values::AbstractMatrix, lm::LabelEnc.OneOfK, obsdim::LearnBase.ObsDimension)
    classify!(buffer, values, typeof(lm), obsdim)
end

function classify(values::AbstractMatrix, lm::LabelEnc.OneOfK, obsdim::LearnBase.ObsDimension)
    classify(values, typeof(lm), obsdim)
end

function classify{T<:LabelEnc.OneOfK}(values::AbstractMatrix, ::Type{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    K, N = size(values)
    buffer = Vector{Int}(N)
    classify!(buffer, values, T, ObsDim.Last())
end

function classify!{R<:Number,T<:LabelEnc.OneOfK}(buffer::AbstractVector, values::AbstractMatrix{R}, ::Type{T}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
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

function classify{T<:LabelEnc.OneOfK}(values::AbstractMatrix, ::Type{T}, ::ObsDim.First)
    N, K = size(values)
    buffer = Vector{Int}(N)
    classify!(buffer, values, T, ObsDim.First())
end

function classify!{R<:Number,T<:LabelEnc.OneOfK}(buffer::AbstractVector, values::AbstractMatrix{R}, ::Type{T}, ::ObsDim.First)
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

