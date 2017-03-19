_lm(::Type{LabelEnc.TrueFalse},   ::Type{Val{2}}) = LabelEnc.TrueFalse()
_lm(::Type{LabelEnc.ZeroOne},     ::Type{Val{2}}) = LabelEnc.ZeroOne()
_lm(::Type{LabelEnc.MarginBased}, ::Type{Val{2}}) = LabelEnc.MarginBased()
_lm{K}(::Type{LabelEnc.OneOfK},   ::Type{Val{K}}) = LabelEnc.OneOfK(Val{K})
_lm{K}(::Type{LabelEnc.Indices},  ::Type{Val{K}}) = LabelEnc.Indices(Val{K})
_lm{D,K}(::Type{LabelEnc.OneOfK{D}},  ::Type{Val{K}}) = LabelEnc.OneOfK(D,Val{K})
_lm{D,K}(::Type{LabelEnc.Indices{D}}, ::Type{Val{K}}) = LabelEnc.Indices(D,Val{K})

_lm{T}(::Type{LabelEnc.TrueFalse},   ::Type{T}, ::Type{Val{2}}) = LabelEnc.TrueFalse()
_lm{D<:LabelEnc.ZeroOne}(::Type{D},     ::Type, ::Type{Val{2}}) = D()
_lm{D<:LabelEnc.MarginBased}(::Type{D}, ::Type, ::Type{Val{2}}) = D()
_lm{T<:Number}(::Type{LabelEnc.ZeroOne},     ::Type{T}, ::Type{Val{2}}) = LabelEnc.ZeroOne(T)
_lm{T<:Number}(::Type{LabelEnc.MarginBased}, ::Type{T}, ::Type{Val{2}}) = LabelEnc.MarginBased(T)
_lm{T<:Number,K}(::Type{LabelEnc.OneOfK},  ::Type{T}, ::Type{Val{K}}) = LabelEnc.OneOfK(T,Val{K})
_lm{T<:Number,K}(::Type{LabelEnc.Indices}, ::Type{T}, ::Type{Val{K}}) = LabelEnc.Indices(T,Val{K})
_lm{K}(::Type{LabelEnc.OneOfK},  ::Type, ::Type{Val{K}}) = LabelEnc.OneOfK(Val{K})
_lm{K}(::Type{LabelEnc.Indices}, ::Type, ::Type{Val{K}}) = LabelEnc.Indices(Val{K})
_lm{D,K}(::Type{LabelEnc.OneOfK{D}},  ::Type, ::Type{Val{K}}) = LabelEnc.OneOfK(D,Val{K})
_lm{D,K}(::Type{LabelEnc.Indices{D}}, ::Type, ::Type{Val{K}}) = LabelEnc.Indices(D,Val{K})

## Convert views for Vector based using MappedArrays.

convertlabelview(dst::LabelEnc.FuzzyBinary, values, src::VectorLabelEncoding) = throw(MethodError(convertlabelview, (dst,values,src)))
convertlabelview(dst::VectorLabelEncoding,  values, src::LabelEnc.FuzzyBinary) = throw(MethodError(convertlabelview, (dst,values,src)))

function convertlabelview(dst::VectorLabelEncoding, values::AbstractVector)
    convertlabelview(dst, values, labelenc(values))
end

function convertlabelview{T,V}(dst::VectorLabelEncoding{T,2}, values::AbstractVector{V}, src::LabelEnc.OneVsRest{V})
    f    = x -> convertlabel(dst, x, src)
    ReadonlyMappedArray{T,1,typeof(values),typeof(f)}(f, values)
end

function convertlabelview{T,K,V}(dst::VectorLabelEncoding{T,K}, values::AbstractVector{V}, src::VectorLabelEncoding{V,K})
    f    = x -> convertlabel(dst, x, src)
    finv = x -> convertlabel(src, x, dst)
    MappedArray{T,1,typeof(values),typeof(f),typeof(finv)}(f, finv, values)
end

## General Vector based

function convertlabel{T,K,S}(dst::VectorLabelEncoding{T,K}, x, src::VectorLabelEncoding{S,K})::T
    ind2label(label2ind(x, src), dst)::T
end

function convertlabel{T,K,S}(dst::VectorLabelEncoding{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,K})
    convertlabel.(dst, values, src)::Vector{T}
end

function convertlabel{K,S}(dst::VectorLabelEncoding{Bool,K}, values::AbstractVector, src::VectorLabelEncoding{S,K})
    convertlabel.(dst, values, src)::BitVector
end

## Generic types to objects

function convertlabel{L<:LabelEncoding,T,K}(::Type{L}, values::AbstractArray{Bool}, src::LabelEncoding{T,K}, args...)
    convertlabel(_lm(L,Val{K}), values, src, args...)
end

function convertlabel{L<:LabelEncoding,T,S,K}(::Type{L}, values::AbstractArray{T}, src::LabelEncoding{S,K}, args...)
    convertlabel(_lm(L,T,Val{K}), values, src, args...)
end

function convertlabel{T,K,M}(dst::LabelEncoding{T,K,M}, values::AbstractMatrix)
    convertlabel(dst, values, labelenc(values))::Array{T,M}
end

function convertlabel{K}(dst::LabelEncoding{Bool,K,1}, values::AbstractMatrix)
    convertlabel(dst, values, labelenc(values))::BitVector
end

function convertlabel{T,K,M}(dst::LabelEncoding{T,K,M}, values::AbstractVector) # avoid method clash
    convertlabel(dst, values, labelenc(values))::Array{T,M}
end

function convertlabel{K}(dst::LabelEncoding{Bool,K,1}, values::AbstractVector) # avoid method clash
    convertlabel(dst, values, labelenc(values))::BitVector
end

convertlabel(dst, values::AbstractArray) = convertlabel(dst, values, labelenc(values))

## OneVsRest

function convertlabel{T}(dst::LabelEnc.OneVsRest{T}, values::AbstractVector{T})
    convertlabel(dst, values, dst)
end

## NativeLabels

function convertlabel(dst, values, src_lbl::AbstractVector, args...)
    convertlabel(dst, values, LabelEnc.NativeLabels(src_lbl), args...)
end

# NativeLabels binary inference helper

function convertlabel{T,S,K}(dst_lbl::AbstractVector{T}, values, src::LabelEncoding{S,K}, args...)
    convertlabel(LabelEnc.NativeLabels{T,K}(dst_lbl), values, src, args...)
end

function convertlabel{S,K,T}(dst::LabelEncoding{S,K}, values, src_lbl::AbstractVector{T}, args...)
    convertlabel(dst, values, LabelEnc.NativeLabels{T,K}(src_lbl), args...)
end

function convertlabel{L<:BinaryLabelEncoding,T}(::Type{L}, values, src_lbl::AbstractVector{T}, args...)
    convertlabel(L, values, LabelEnc.NativeLabels{T,2}(src_lbl), args...)
end

## OneOfK obsdim kw specified

convertlabel(dst, values, src; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, src, LearnBase.obs_dim(obsdim))

convertlabel(dst, values; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, labelenc(values), LearnBase.obs_dim(obsdim))

function convertlabel(dst, values::AbstractMatrix; obsdim = LearnBase.default_obsdim(values))
    nobsdim = LearnBase.obs_dim(obsdim)
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

function convertlabel{TD,TS,K}(dst::LabelEnc.OneOfK{TD,K}, values::AbstractMatrix, src::LabelEnc.OneOfK{TS,K}, ::LearnBase.ObsDimension)
    Matrix{TD}(values)
end

function convertlabel{T,K,S,KS}(dst::LabelEnc.OneOfK{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,KS}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    @assert KS <= K
    n = length(values)
    buffer = zeros(T, K, n)
    @inbounds for i in 1:n
        buffer[label2ind(values[i], src), i] = one(T)
    end
    buffer
end

function convertlabel{T,K,S,KS}(dst::LabelEnc.OneOfK{T,K}, values::AbstractVector, src::VectorLabelEncoding{S,KS}, ::ObsDim.First)
    @assert KS <= K
    n = length(values)
    buffer = zeros(T, n, K)
    @inbounds for i in 1:n
        buffer[i, label2ind(values[i], src)] = one(T)
    end
    buffer
end

## From OneOfK

@inline _new_vec{T}(::Type{T}, n) = Array{T}(n)
@inline _new_vec(::Type{Bool}, n) = BitVector(n)

function convertlabel{TD,T,TS,K}(dst::VectorLabelEncoding{TD,K}, values::AbstractMatrix{T}, src::LabelEnc.OneOfK{TS,K}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    @assert size(values, 1) == K
    n = size(values, 2)
    buffer = _new_vec(TD, n)
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

function convertlabel{TD,T,TS,K}(dst::VectorLabelEncoding{TD,K}, values::AbstractMatrix{T}, src::LabelEnc.OneOfK{TS,K}, ::ObsDim.First)
    @assert size(values, 2) == K
    n = size(values, 1)
    buffer = _new_vec(TD, n)
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
