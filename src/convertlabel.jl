_lm(::Type{LabelModes.FuzzyBinary}, ::Type{Val{2}}) = LabelModes.FuzzyBinary()
_lm(::Type{LabelModes.TrueFalse},   ::Type{Val{2}}) = LabelModes.TrueFalse()
_lm(::Type{LabelModes.ZeroOne},     ::Type{Val{2}}) = LabelModes.ZeroOne()
_lm(::Type{LabelModes.MarginBased}, ::Type{Val{2}}) = LabelModes.MarginBased()
_lm{K}(::Type{LabelModes.OneOfK},   ::Type{Val{K}}) = LabelModes.OneOfK(Val{K})
_lm{K}(::Type{LabelModes.Indices},  ::Type{Val{K}}) = LabelModes.Indices(Val{K})
_lm{D,K}(::Type{LabelModes.OneOfK{D}},  ::Type{Val{K}}) = LabelModes.OneOfK(D,Val{K})
_lm{D,K}(::Type{LabelModes.Indices{D}}, ::Type{Val{K}}) = LabelModes.Indices(D,Val{K})

_lm{T}(::Type{LabelModes.FuzzyBinary}, ::Type{T}, ::Type{Val{2}}) = LabelModes.FuzzyBinary()
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

## General Binary

function convertlabel{T<:MLLabelUtils.BinaryLabelMode}(dst::T, x, src::MLLabelUtils.BinaryLabelMode)::labeltype(T)
    isposlabel(x, src) ? poslabel(dst) : neglabel(dst)
end

function convertlabel{T<:MLLabelUtils.BinaryLabelMode}(dst::T, values::AbstractVector, src::MLLabelUtils.BinaryLabelMode)
    convertlabel.(dst, values, src)::Vector{labeltype(T)}
end

function convertlabel{L<:MLLabelUtils.LabelMode,K}(::Type{L}, values::AbstractArray{Bool}, src::MLLabelUtils.LabelMode{K})
    convertlabel(_lm(L,Val{K}), values, src)
end

function convertlabel{L<:MLLabelUtils.LabelMode,T,K}(::Type{L}, values::AbstractArray{T}, src::MLLabelUtils.LabelMode{K})
    convertlabel(_lm(L,T,Val{K}), values, src)
end

convertlabel(dst, values::AbstractArray) = convertlabel(dst, values, labelmode(values))

## OneVsRest

function convertlabel{T}(dst::LabelModes.OneVsRest{T}, values::AbstractVector{T})
    convertlabel(dst, values, dst)
end

## NativeLabels

convertlabel{T}(dst_lbl::AbstractVector{T}, values, src::MLLabelUtils.BinaryLabelMode) = convertlabel(LabelModes.NativeLabels{T,2}(dst_lbl), values, src)

convertlabel(dst_lbl::AbstractVector, values, src::MLLabelUtils.LabelMode) = convertlabel(LabelModes.NativeLabels(dst_lbl), values, src)

function convertlabel{L<:MLLabelUtils.BinaryLabelMode,T}(::Type{L}, values, src_lbl::AbstractVector{T})
    convertlabel(L, values, LabelModes.NativeLabels{T,2}(src_lbl))
end

function convertlabel{T}(dst::MLLabelUtils.BinaryLabelMode, values, src_lbl::AbstractVector{T})
    convertlabel(dst, values, LabelModes.NativeLabels{T,2}(src_lbl))
end

convertlabel(dst, values, src_lbl::AbstractVector) = convertlabel(dst, values, LabelModes.NativeLabels(src_lbl))

## OneOfK

function convertlabel{T}(dst::MLLabelUtils.BinaryLabelMode, values::AbstractMatrix, src::LabelModes.OneOfK{T,2})
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel{T<:LabelModes.OneOfK}(dst::T, values::AbstractVector, src::MLLabelUtils.BinaryLabelMode)
    convertlabel(dst, values, src, LearnBase.default_obsdim(values))
end

function convertlabel{L<:LabelModes.OneOfK,K}(::Type{L}, values::AbstractArray{Bool}, src::MLLabelUtils.LabelMode{K}, obsdim)
    convertlabel(_lm(L,Val{K}), values, src, obsdim)
end

function convertlabel{L<:LabelModes.OneOfK,T,K}(::Type{L}, values::AbstractArray{T}, src::MLLabelUtils.LabelMode{K}, obsdim)
    convertlabel(_lm(L,T,Val{K}), values, src, obsdim)
end

convertlabel(dst, values, src; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, src, LearnBase.obs_dim(obsdim))

convertlabel(dst, values; obsdim = LearnBase.default_obsdim(values)) = convertlabel(dst, values, labelmode(values), LearnBase.obs_dim(obsdim))

function convertlabel{T,K}(dst::LabelModes.OneOfK{T,K}, values::AbstractVector, src::MLLabelUtils.BinaryLabelMode, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    n = length(values)
    buffer = zeros(T, K, n)
    @inbounds for i in 1:n
        if isposlabel(values[i], src)
            buffer[1,i] = one(T)
        else
            buffer[2,i] = one(T)
        end
    end
    buffer
end

function convertlabel{T,K}(dst::LabelModes.OneOfK{T,K}, values::AbstractVector, src::MLLabelUtils.BinaryLabelMode, ::ObsDim.First)
    n = length(values)
    buffer = zeros(T, n, K)
    @inbounds for i in 1:n
        if isposlabel(values[i], src)
            buffer[i,1] = one(T)
        else
            buffer[i,2] = one(T)
        end
    end
    buffer
end

function convertlabel{TD,TS}(dst::LabelModes.OneOfK{TD,2}, values::AbstractMatrix, src::LabelModes.OneOfK{TS,2}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    Matrix{TD}(values)
end

function convertlabel{L<:MLLabelUtils.BinaryLabelMode,T}(dst::L, values::AbstractMatrix, src::LabelModes.OneOfK{T,2}, ::Union{ObsDim.Last,ObsDim.Constant{2}})
    @assert size(values,1) >= 2
    n = size(values, 2)
    buffer = Array{labeltype(L)}(n)
    @inbounds for i in 1:n
        buffer[i] = values[1,i] >= values[2,i] ? poslabel(dst) : neglabel(dst)
    end
    buffer
end

