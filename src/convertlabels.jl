_lm(::Type{LabelModes.FuzzyBinary}, ::Type{Val{2}}) = LabelModes.FuzzyBinary()
_lm(::Type{LabelModes.TrueFalse},   ::Type{Val{2}}) = LabelModes.TrueFalse()
_lm(::Type{LabelModes.ZeroOne},     ::Type{Val{2}}) = LabelModes.ZeroOne()
_lm(::Type{LabelModes.MarginBased}, ::Type{Val{2}}) = LabelModes.MarginBased()
_lm{K}(::Type{LabelModes.OneOfK},   ::Type{Val{K}}) = LabelModes.OneOfK(Val{K})
_lm{K}(::Type{LabelModes.Indices},  ::Type{Val{K}}) = LabelModes.Indices(Val{K})

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

function convertlabels{T<:MLLabelUtils.BinaryLabelMode}(dst::T, x, src::MLLabelUtils.BinaryLabelMode)::eltype(T)
    isposlabel(x, src) ? poslabel(dst) : neglabel(dst)
end

function convertlabels{T<:MLLabelUtils.BinaryLabelMode}(dst::T, values::AbstractVector, src::MLLabelUtils.BinaryLabelMode)
    convertlabels.(dst, values, src)::Vector{eltype(T)}
end

function convertlabels{L<:MLLabelUtils.LabelMode}(::Type{L}, values::AbstractVector{Bool}, src::LabelModes.TrueFalse)
    convertlabels(_lm(L,Val{2}), values, src)
end

function convertlabels{L<:MLLabelUtils.LabelMode,T,K}(::Type{L}, values::AbstractVector{T}, src::MLLabelUtils.LabelMode{K})
    convertlabels(_lm(L,T,Val{K}), values, src)
end

convertlabels(lbl::AbstractVector, values, src::MLLabelUtils.LabelMode) = convertlabels(LabelModes.NativeLabels(lbl), values, src)

convertlabels(dst, values) = convertlabels(dst, values, labelmode(values))

convertlabels{T}(dst::LabelModes.OneVsRest{T}, values::AbstractVector{T}) = convertlabels(dst, values, dst)

