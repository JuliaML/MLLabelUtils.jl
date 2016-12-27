_lm(::Type{LabelModes.TrueFalse},   ::Type{Val{2}}) = LabelModes.TrueFalse()
_lm(::Type{LabelModes.ZeroOne},     ::Type{Val{2}}) = LabelModes.ZeroOne()
_lm(::Type{LabelModes.MarginBased}, ::Type{Val{2}}) = LabelModes.MarginBased()
_lm{K}(::Type{LabelModes.OneOfK},   ::Type{Val{K}}) = LabelModes.OneOfK(Val{K})
_lm{K}(::Type{LabelModes.Indices},  ::Type{Val{K}}) = LabelModes.Indices(Val{K})
_lm{D,K}(::Type{LabelModes.OneOfK{D}},  ::Type{Val{K}}) = LabelModes.OneOfK(D,Val{K})
_lm{D,K}(::Type{LabelModes.Indices{D}}, ::Type{Val{K}}) = LabelModes.Indices(D,Val{K})

_lm{T}(::Type{LabelModes.TrueFalse},   ::Type{T}, ::Type{Val{2}}) = LabelModes.TrueFalse()
_lm{D<:LabelModes.ZeroOne}(::Type{D},     ::Type, ::Type{Val{2}}) = D()
_lm{D<:LabelModes.MarginBased}(::Type{D}, ::Type, ::Type{Val{2}}) = D()
_lm{T<:Number}(::Type{LabelModes.ZeroOne},     ::Type{T}, ::Type{Val{2}}) = LabelModes.ZeroOne(T)
_lm{T<:Number}(::Type{LabelModes.MarginBased}, ::Type{T}, ::Type{Val{2}}) = LabelModes.MarginBased(T)
_lm{T<:Number,K}(::Type{LabelModes.OneOfK},  ::Type{T}, ::Type{Val{K}}) = LabelModes.OneOfK(T,Val{K})
_lm{T<:Number,K}(::Type{LabelModes.Indices}, ::Type{T}, ::Type{Val{K}}) = LabelModes.Indices(T,Val{K})
_lm{K}(::Type{LabelModes.OneOfK},  ::Type, ::Type{Val{K}}) = LabelModes.OneOfK(Val{K})
_lm{K}(::Type{LabelModes.Indices}, ::Type, ::Type{Val{K}}) = LabelModes.Indices(Val{K})
_lm{D,K}(::Type{LabelModes.OneOfK{D}},  ::Type, ::Type{Val{K}}) = LabelModes.OneOfK(D,Val{K})
_lm{D,K}(::Type{LabelModes.Indices{D}}, ::Type, ::Type{Val{K}}) = LabelModes.Indices(D,Val{K})

## General Vector based

function convertlabel{T,K,S}(dst::MLLabelUtils.VectorLabelMode{T,K}, x, src::MLLabelUtils.VectorLabelMode{S,K})::T
    ind2label(label2ind(x, src), dst)::T
end

function convertlabel{T,K,S}(dst::MLLabelUtils.VectorLabelMode{T,K}, values::AbstractVector, src::MLLabelUtils.VectorLabelMode{S,K})
    convertlabel.(dst, values, src)::Vector{T}
end

## Generic types to objects

function convertlabel{L<:MLLabelUtils.LabelMode,T,K}(::Type{L}, values::AbstractArray{Bool}, src::MLLabelUtils.LabelMode{T,K}, args...)
    convertlabel(_lm(L,Val{K}), values, src, args...)
end

function convertlabel{L<:MLLabelUtils.LabelMode,T,S,K}(::Type{L}, values::AbstractArray{T}, src::MLLabelUtils.LabelMode{S,K}, args...)
    convertlabel(_lm(L,T,Val{K}), values, src, args...)
end

function convertlabel{T,K,M}(dst::MLLabelUtils.LabelMode{T,K,M}, values::AbstractMatrix)
    convertlabel(dst, values, labelmode(values))::Array{T,M}
end

function convertlabel{T,K,M}(dst::MLLabelUtils.LabelMode{T,K,M}, values::AbstractVector) # avoid method clash
    convertlabel(dst, values, labelmode(values))::Array{T,M}
end

convertlabel(dst, values::AbstractArray) = convertlabel(dst, values, labelmode(values))

## OneVsRest

function convertlabel{T}(dst::LabelModes.OneVsRest{T}, values::AbstractVector{T})
    convertlabel(dst, values, dst)
end

## NativeLabels

function convertlabel(dst_lbl::AbstractVector, values, src::MLLabelUtils.LabelMode, args...)
    convertlabel(LabelModes.NativeLabels(dst_lbl), values, src, args...)
end

function convertlabel(dst, values, src_lbl::AbstractVector, args...)
    convertlabel(dst, values, LabelModes.NativeLabels(src_lbl), args...)
end

# NativeLabels binary inference helper

function convertlabel{T}(dst_lbl::AbstractVector{T}, values, src::MLLabelUtils.BinaryLabelMode, args...)
    convertlabel(LabelModes.NativeLabels{T,2}(dst_lbl), values, src, args...)
end

function convertlabel{L<:MLLabelUtils.BinaryLabelMode,T}(::Type{L}, values, src_lbl::AbstractVector{T}, args...)
    convertlabel(L, values, LabelModes.NativeLabels{T,2}(src_lbl), args...)
end

function convertlabel{T}(dst::MLLabelUtils.BinaryLabelMode, values, src_lbl::AbstractVector{T}, args...)
    convertlabel(dst, values, LabelModes.NativeLabels{T,2}(src_lbl), args...)
end

## OneOfK obsdim kw specified

convertlabel(dst, values, src; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, src, LearnBase.obs_dim(obsdim))

convertlabel(dst, values; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, labelmode(values), LearnBase.obs_dim(obsdim))

function convertlabel(dst, values::AbstractMatrix; obsdim = LearnBase.default_obsdim(values))
    nobsdim = LearnBase.obs_dim(obsdim)
    convertlabel(dst, values, labelmode(values, nobsdim), nobsdim)
end

# OneOfK default obsdim inserted

function convertlabel(dst::LabelModes.OneOfK, values::AbstractMatrix, src::LabelModes.OneOfK)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel(dst::MLLabelUtils.LabelMode, values::AbstractMatrix, src::LabelModes.OneOfK)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel(dst::LabelModes.OneOfK, values::AbstractArray, src::MLLabelUtils.LabelMode)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

## To OneOfK

function convertlabel{TD,TS,K}(dst::LabelModes.OneOfK{TD,K}, values::AbstractMatrix, src::LabelModes.OneOfK{TS,K}, ::LearnBase.ObsDimension)
    Matrix{TD}(values)
end

function convertlabel{T,K,S,KS}(dst::LabelModes.OneOfK{T,K}, values::AbstractVector, src::MLLabelUtils.VectorLabelMode{S,KS}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    @assert KS <= K
    n = length(values)
    buffer = zeros(T, K, n)
    @inbounds for i in 1:n
        buffer[label2ind(values[i], src), i] = one(T)
    end
    buffer
end

function convertlabel{T,K,S,KS}(dst::LabelModes.OneOfK{T,K}, values::AbstractVector, src::MLLabelUtils.VectorLabelMode{S,KS}, ::ObsDim.First)
    @assert KS <= K
    n = length(values)
    buffer = zeros(T, n, K)
    @inbounds for i in 1:n
        buffer[i, label2ind(values[i], src)] = one(T)
    end
    buffer
end

## From OneOfK

function convertlabel{TD,T,TS,K}(dst::MLLabelUtils.VectorLabelMode{TD,K}, values::AbstractMatrix{T}, src::LabelModes.OneOfK{TS,K}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    @assert size(values, 1) == K
    n = size(values, 2)
    buffer = Array{TD}(n)
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

function convertlabel{TD,T,TS,K}(dst::MLLabelUtils.VectorLabelMode{TD,K}, values::AbstractMatrix{T}, src::LabelModes.OneOfK{TS,K}, ::ObsDim.First)
    @assert size(values, 2) == K
    n = size(values, 1)
    buffer = Array{TD}(n)
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

