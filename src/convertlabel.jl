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

convertlabelview(dst::LabelEnc.FuzzyBinary, values, src::VectorLabelEncoding) = throw(MethodError(convertlabelview, (dst,values,src)))
convertlabelview(dst::VectorLabelEncoding,  values, src::LabelEnc.FuzzyBinary) = throw(MethodError(convertlabelview, (dst,values,src)))

function convertlabelview(dst::VectorLabelEncoding, values::AbstractVector)
    convertlabelview(dst, values, labelenc(values))
end

function convertlabelview(dst::VectorLabelEncoding{T,2}, values::AbstractVector{V}, src::LabelEnc.OneVsRest{V}) where {T,V}
    f = x -> convertlabel(dst, x, src)
    ReadonlyMappedArray{T,1,typeof(values),typeof(f)}(f, values)
end

function convertlabelview(dst::VectorLabelEncoding{T,K}, values::AbstractVector{V}, src::VectorLabelEncoding{V,K}) where {T,K,V}
    f    = x -> convertlabel(dst, x, src)
    finv = x -> convertlabel(src, x, dst)
    MappedArray{T,1,typeof(values),typeof(f),typeof(finv)}(f, finv, values)
end

## General Vector based

function convertlabel(dst::VectorLabelEncoding{T,K}, x, src::VectorLabelEncoding{S,K}) where {T,K,S}
    ind2label(label2ind(x, src), dst)::T
end

function convertlabel(dst::VectorLabelEncoding{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,K}) where {T,K,S}
    convertlabel.(dst, values, src)::_array_type(T,Val{1})
end

## Generic types to objects

function convertlabel(::Type{L}, x, src::LabelEncoding{T,K}) where {L<:LabelEncoding,T,K}
    convertlabel(_lm(L,Val{K}), x, src)
end

function convertlabel(::Type{L}, values::AbstractArray{Bool}, src::LabelEncoding{T,K}, args...) where {L<:LabelEncoding,T,K}
    convertlabel(_lm(L,Val{K}), values, src, args...)
end

function convertlabel(::Type{L}, values::AbstractArray{T}, src::LabelEncoding{S,K}, args...) where {L<:LabelEncoding,T,S,K}
    convertlabel(_lm(L,T,Val{K}), values, src, args...)
end

function convertlabel(dst::LabelEncoding{T,K,M}, values::AbstractMatrix) where {T,K,M}
    convertlabel(dst, values, labelenc(values))::_array_type(T,Val{M})
end

function convertlabel(dst::LabelEncoding{T,K,M}, values::AbstractVector) where {T,K,M} # avoid method clash
    convertlabel(dst, values, labelenc(values))::_array_type(T,Val{M})
end

convertlabel(dst, values::AbstractArray) = convertlabel(dst, values, labelenc(values))

## OneVsRest

function convertlabel(dst::LabelEnc.OneVsRest{T}, values::AbstractVector{T}) where {T}
    convertlabel(dst, values, dst)
end

## NativeLabels

function convertlabel(dst, values, src_lbl::AbstractVector, args...)
    convertlabel(dst, values, LabelEnc.NativeLabels(src_lbl), args...)
end

# NativeLabels binary inference helper

function convertlabel(dst_lbl::AbstractVector{T}, values, src::LabelEncoding{S,K}, args...) where {T,S,K}
    convertlabel(LabelEnc.NativeLabels{T,K}(dst_lbl), values, src, args...)
end

function convertlabel(dst::LabelEncoding{S,K}, values, src_lbl::AbstractVector{T}, args...) where {S,K,T}
    convertlabel(dst, values, LabelEnc.NativeLabels{T,K}(src_lbl), args...)
end

function convertlabel(::Type{L}, values, src_lbl::AbstractVector{T}, args...) where {L<:BinaryLabelEncoding,T}
    convertlabel(L, values, LabelEnc.NativeLabels{T,2}(src_lbl), args...)
end

## OneOfK obsdim kw specified

convertlabel(dst, values, src; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, src, convert(LearnBase.ObsDimension,obsdim))

convertlabel(dst, values; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, labelenc(values), convert(LearnBase.ObsDimension,obsdim))

function convertlabel(dst, values::AbstractMatrix; obsdim = LearnBase.default_obsdim(values))
    nobsdim = convert(LearnBase.ObsDimension,obsdim)
    convertlabel(dst, values, labelenc(values, nobsdim), nobsdim)
end

# OneOfK default obsdim inserted

function convertlabel(dst::LabelEnc.OneOfK, values::AbstractMatrix, src::LabelEnc.OneOfK)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel(dst::LabelEncoding, values::AbstractMatrix, src::LabelEnc.OneOfK)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel(dst::LabelEnc.OneOfK, values::AbstractArray, src::LabelEncoding)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

## To OneOfK

function convertlabel(dst::LabelEnc.OneOfK{TD,K}, values::AbstractMatrix, src::LabelEnc.OneOfK{TS,K}, ::LearnBase.ObsDimension) where {TD,TS,K}
    _array_type(TD,Val{2})(values)
end

function convertlabel(dst::LabelEnc.OneOfK{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,KS}, ::Union{ObsDim.Last,ObsDim.Constant{2}}) where {T,K,S,KS}
    @assert KS <= K
    n = length(values)
    buffer = _array_type(T,Val{2})(zeros(T, K, n))
    @inbounds for i in 1:n
        buffer[label2ind(values[i], src), i] = one(T)
    end
    buffer
end

function convertlabel(dst::LabelEnc.OneOfK{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,KS}, ::ObsDim.First) where {T,K,S,KS}
    @assert KS <= K
    n = length(values)
    buffer = _array_type(T,Val{2})(zeros(T, n, K))
    @inbounds for i in 1:n
        buffer[i, label2ind(values[i], src)] = one(T)
    end
    buffer
end

## From OneOfK

function convertlabel(dst::VectorLabelEncoding{TD,K}, values::AbstractMatrix{T}, src::LabelEnc.OneOfK{TS,K}, ::Union{ObsDim.Last,ObsDim.Constant{2}}) where {TD,T,TS,K}
    @assert size(values, 1) == K
    n = size(values, 2)
    buffer = _array_type(TD,Val{1})(undef, n)
    @inbounds for i in 1:n
        tind = 1
        tmax = typemin(T)
        for j in 1:K
            tval = values[j,i]
            if tval > tmax
                tind = j
                tmax = tval
            end
        end
        buffer[i] = ind2label(tind, dst)
    end
    buffer
end

function convertlabel(dst::VectorLabelEncoding{TD,K}, values::AbstractMatrix{T}, src::LabelEnc.OneOfK{TS,K}, ::ObsDim.First) where {TD,T,TS,K}
    @assert size(values, 2) == K
    n = size(values, 1)
    buffer = _array_type(TD,Val{1})(n)
    @inbounds for i in 1:n
        tind = 1
        tmax = typemin(T)
        for j in 1:K
            tval = values[i,j]
            if tval > tmax
                tind = j
                tmax = tval
            end
        end
        buffer[i] = ind2label(tind, dst)
    end
    buffer
end
