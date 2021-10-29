_lm(::Type{LabelEnc.TrueFalse},   ::Type{Val{2}}) = LabelEnc.TrueFalse()
_lm(::Type{LabelEnc.ZeroOne},     ::Type{Val{2}}) = LabelEnc.ZeroOne()
_lm(::Type{LabelEnc.MarginBased}, ::Type{Val{2}}) = LabelEnc.MarginBased()
_lm(::Type{LabelEnc.OneOfK},      ::Type{Val{K}}) where {K} = LabelEnc.OneOfK(Val{K})
_lm(::Type{LabelEnc.Indices},     ::Type{Val{K}}) where {K} = LabelEnc.Indices(Val{K})
_lm(::Type{LabelEnc.OneOfK{D}},   ::Type{Val{K}}) where {D,K} = LabelEnc.OneOfK(D,Val{K})
_lm(::Type{LabelEnc.Indices{D}},  ::Type{Val{K}}) where {D,K} = LabelEnc.Indices(D,Val{K})

_lm(::Type{LabelEnc.TrueFalse}, ::Type{T}, ::Type{Val{2}}) where {T} = LabelEnc.TrueFalse()
_lm(::Type{D}, ::Type, ::Type{Val{2}}) where {D<:LabelEnc.ZeroOne} = D()
_lm(::Type{D}, ::Type, ::Type{Val{2}}) where {D<:LabelEnc.MarginBased} = D()
_lm(::Type{LabelEnc.ZeroOne}, ::Type{T}, ::Type{Val{2}}) where {T<:Number} = LabelEnc.ZeroOne(T)
_lm(::Type{LabelEnc.MarginBased}, ::Type{T}, ::Type{Val{2}}) where {T<:Number} = LabelEnc.MarginBased(T)
_lm(::Type{LabelEnc.OneOfK},  ::Type{T}, ::Type{Val{K}}) where {T<:Number,K} = LabelEnc.OneOfK(T,Val{K})
_lm(::Type{LabelEnc.Indices}, ::Type{T}, ::Type{Val{K}}) where {T<:Number,K} = LabelEnc.Indices(T,Val{K})
_lm(::Type{LabelEnc.OneOfK},  ::Type, ::Type{Val{K}}) where {K} = LabelEnc.OneOfK(Val{K})
_lm(::Type{LabelEnc.Indices}, ::Type, ::Type{Val{K}}) where {K} = LabelEnc.Indices(Val{K})
_lm(::Type{LabelEnc.OneOfK{D}},  ::Type, ::Type{Val{K}}) where {D,K} = LabelEnc.OneOfK(D,Val{K})
_lm(::Type{LabelEnc.Indices{D}}, ::Type, ::Type{Val{K}}) where {D,K} = LabelEnc.Indices(D,Val{K})

@inline _array_type(::Type{T}, ::Type{Val{N}}) where {T,N} = Array{T,N}
@inline _array_type(::Type{Bool}, ::Type{Val{N}}) where {N} = BitArray{N}

## Convert views for Vector based using MappedArrays.

LearnBase.convertlabelview(dst::LabelEnc.FuzzyBinary, values, src::LearnBase.VectorLabelEncoding) =
    throw(MethodError(convertlabelview, (dst,values,src)))
LearnBase.convertlabelview(dst::LearnBase.VectorLabelEncoding,  values, src::LabelEnc.FuzzyBinary) =
    throw(MethodError(convertlabelview, (dst,values,src)))

function LearnBase.convertlabelview(dst::LearnBase.VectorLabelEncoding, values::AbstractVector)
    convertlabelview(dst, values, labelenc(values))
end

function LearnBase.convertlabelview(dst::LearnBase.VectorLabelEncoding{T,2},
                          values::AbstractVector{V},
                          src::LabelEnc.OneVsRest{V}) where {T,V}
    f = x -> convertlabel(dst, x, src)
    ReadonlyMappedArray{T,1,typeof(values),typeof(f)}(f, values)
end

function LearnBase.convertlabelview(dst::LearnBase.VectorLabelEncoding{T,K},
                                    values::AbstractVector{V},
                                    src::LearnBase.VectorLabelEncoding{V,K}) where {T,K,V}
    f    = x -> convertlabel(dst, x, src)
    finv = x -> convertlabel(src, x, dst)
    MappedArray{T,1,typeof(values),typeof(f),typeof(finv)}(f, finv, values)
end

## General Vector based

function LearnBase.convertlabel(dst::LearnBase.VectorLabelEncoding{T,K}, x, 
                      src::LearnBase.VectorLabelEncoding{S,K}) where {T,K,S}
    ind2label(label2ind(x, src), dst)::T
end

function LearnBase.convertlabel(dst::LearnBase.VectorLabelEncoding{T,K},
                                values::AbstractVector,
                                src::LearnBase.VectorLabelEncoding{S,K}) where {T,K,S}
    convertlabel.(dst, values, src)::_array_type(T,Val{1})
end

## Generic types to objects

function LearnBase.convertlabel(::Type{L}, x, src::LearnBase.LabelEncoding{T,K}) where {L<:LearnBase.LabelEncoding,T,K}
    convertlabel(_lm(L,Val{K}), x, src)
end

function LearnBase.convertlabel(::Type{L}, values::AbstractArray{Bool}, src::LearnBase.LabelEncoding{T,K};
                                kwargs...) where {L<:LearnBase.LabelEncoding,T,K}
    convertlabel(_lm(L,Val{K}), values, src; kwargs...)
end

function LearnBase.convertlabel(::Type{L}, values::AbstractArray{T}, src::LearnBase.LabelEncoding{S,K};
                                kwargs...) where {L<:LearnBase.LabelEncoding,T,S,K}
    convertlabel(_lm(L,T,Val{K}), values, src; kwargs...)
end

function LearnBase.convertlabel(dst::LearnBase.LabelEncoding{T,K,M}, values::AbstractMatrix) where {T,K,M}
    convertlabel(dst, values, labelenc(values))::_array_type(T,Val{M})
end

function LearnBase.convertlabel(dst::LearnBase.LabelEncoding{T,K,M}, values::AbstractVector) where {T,K,M} # avoid method clash
    convertlabel(dst, values, labelenc(values))::_array_type(T,Val{M})
end

LearnBase.convertlabel(dst, values::AbstractArray) = convertlabel(dst, values, labelenc(values))

## OneVsRest

function LearnBase.convertlabel(dst::LabelEnc.OneVsRest{T}, values::AbstractVector{T}) where {T}
    convertlabel(dst, values, dst)
end

## NativeLabels

function LearnBase.convertlabel(dst, values, src_lbl::AbstractVector; kwargs...)
    convertlabel(dst, values, LabelEnc.NativeLabels(src_lbl); kwargs...)
end

# NativeLabels binary inference helper

function LearnBase.convertlabel(dst_lbl::AbstractVector{T}, values, src::LearnBase.LabelEncoding{S,K};
                                kwargs...) where {T,S,K}
    convertlabel(LabelEnc.NativeLabels{T,K}(dst_lbl), values, src; kwargs...)
end

function LearnBase.convertlabel(dst::LearnBase.LabelEncoding{S,K}, values, src_lbl::AbstractVector{T};
                                kwargs...) where {S,K,T}
    convertlabel(dst, values, LabelEnc.NativeLabels{T,K}(src_lbl); kwargs...)
end

function LearnBase.convertlabel(::Type{L}, values, src_lbl::AbstractVector{T};
                                kwargs...) where {L<:LearnBase.BinaryLabelEncoding,T}
    convertlabel(L, values, LabelEnc.NativeLabels{T,2}(src_lbl); kwargs...)
end

## OneOfK obsdim kw specified

LearnBase.convertlabel(dst, values; obsdim = LearnBase.default_obsdim(values)) =
    convertlabel(dst, values, labelenc(values); obsdim = obsdim)

LearnBase.convertlabel(dst, values::AbstractMatrix; obsdim = LearnBase.default_obsdim(values)) =
    convertlabel(dst, values, labelenc(values; obsdim = obsdim); obsdim = obsdim)

## To OneOfK

LearnBase.convertlabel(dst::LabelEnc.OneOfK{TD,K},
                       values::AbstractMatrix,
                       src::LabelEnc.OneOfK{TS,K};
                       kwargs...) where {TD,TS,K} =
    _array_type(TD, Val{2})(values)

function LearnBase.convertlabel(dst::LabelEnc.OneOfK{T,K},
                                values::AbstractVector,
                                src::LearnBase.VectorLabelEncoding{S,KS};
                                obsdim = LearnBase.default_obsdim(_array_type(T, Val{2}))) where {T,K,S,KS}
    @assert KS <= K
    n = length(values)
    buffer = (obsdim == 1) ? _array_type(T, Val{2})(zeros(T, n, K)) : _array_type(T, Val{2})(zeros(T, K, n))
    @inbounds for i in 1:n
        if obsdim == 1
            buffer[i, label2ind(values[i], src)] = one(T)
        else
            buffer[label2ind(values[i], src), i] = one(T)
        end
    end

    return buffer
end

## From OneOfK

function LearnBase.convertlabel(dst::LearnBase.VectorLabelEncoding{TD,K},
                                values::AbstractMatrix{T},
                                src::LabelEnc.OneOfK{TS,K};
                                obsdim = LearnBase.default_obsdim(values)) where {TD,T,TS,K}
    not_obsdim = mod1(obsdim + 1, 2)
    @assert size(values, not_obsdim) == K
    n = size(values, obsdim)
    buffer = _array_type(TD, Val{1})(undef, n)
    inds = argmax(values; dims = not_obsdim)
    @inbounds for i in 1:n
        buffer[i] = ind2label(Tuple(inds[i])[not_obsdim], dst)
    end

    return buffer
end
