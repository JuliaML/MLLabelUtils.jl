## ZeroOne

"""
    classify(x, encoding)

Returns the classified version of `x` given the `encoding`.
Which means that if `x` can be interpreted as a positive label,
the positive label of `encoding` is returned; the negative otherwise.

# Examples
```jldoctest
julia> classify(0.6, LabelEnc.ZeroOne(UInt8))
0x01

julia> classify(0.4, LabelEnc.ZeroOne(UInt8))
0x00

julia> classify([0.1,0.2,0.6,0.1], LabelEnc.OneOfK)
3
```
"""
LearnBase.classify(value::T, cutoff::Number) where {T<:Number} =
    (value >= cutoff) ? T(1) : T(0)
LearnBase.classify(value::Number, ::Type{LabelEnc.ZeroOne}) = classify(value, 0.5)
LearnBase.classify(value::Number, lm::LabelEnc.ZeroOne{R}) where {R} =
    R(classify(value, lm.cutoff))
LearnBase.classify(values::AbstractVector{T}, cutoff::Number) where {T} =
    classify.(values, cutoff)::Vector{T}

## Margin

_sign(value::T) where {T} = ifelse(signbit(value), T(-1), T(1))::T

LearnBase.classify(value::Number, ::Type{LabelEnc.MarginBased}) = _sign(value)
LearnBase.classify(value::Number, lm::LabelEnc.MarginBased{R}) where {R} = R(_sign(value))

for KIND in (:(LabelEnc.MarginBased), :(LabelEnc.ZeroOne))
    @eval begin
        function LearnBase.classify(values::AbstractVector{T}, ::Type{($KIND)}) where {T}
            classify.(values, ($KIND))::Vector{T}
        end
        function LearnBase.classify(values::AbstractVector{T}, lm::L) where {T,L<:($KIND)}
            classify.(values, lm)::Vector{labeltype(L)}
        end
    end
end

## OneOfK

LearnBase.classify(values::AbstractVector, ::Type{<:LabelEnc.OneOfK}) = argmax(values)
LearnBase.classify(values::AbstractVector, lm::LabelEnc.OneOfK) = classify(values, typeof(lm))

LearnBase.classify(values::AbstractMatrix, T::Type{<:LabelEnc.OneOfK}; obsdim = default_obsdim(values)) =
    classify!(Vector{Int}(undef, size(values, obsdim)), values, T; obsdim = obsdim)
LearnBase.classify(values::AbstractMatrix, lm::LabelEnc.OneOfK; obsdim = default_obsdim(values)) =
    classify(values, typeof(lm); obsdim=obsdim)

function LearnBase.classify!(buffer::AbstractVector, values::AbstractMatrix,
                             T::Type{<:LabelEnc.OneOfK}; obsdim = default_obsdim(values))
    for (i, v) in enumerate(eachslice(values; dims = obsdim))
        buffer[i] = classify(v, T)
    end

    return buffer
end
LearnBase.classify!(buffer, values::AbstractMatrix, lm::LabelEnc.OneOfK; obsdim = default_obsdim(obsdim)) =
    classify!(buffer, values, typeof(lm); obsdim=obsdim)

"""
    classify!(out, x, encoding)

Same as [`classify`](@ref), but uses `out` to store the result.

# Examples
```jldoctest
julia> buffer = zeros(2);
julia> classify!(buffer, [0.4,0.6], LabelEnc.ZeroOne)
2-element Array{Float64,1}:
    0.0
    1.0
```
"""
function LearnBase.classify!(buffer::AbstractVector, values::AbstractVector, lm)
    buffer .= classify.(values, lm)
    
    return buffer
end
