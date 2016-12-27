# everything that should move to learnbase

abstract LabelMode{T,K,M} # Eltype, Labelcount, Arraydimensions
typealias BinaryLabelMode{T,M} LabelMode{T,2,M}
typealias VectorLabelMode{T,K} LabelMode{T,K,1}
typealias MatrixLabelMode{T,K} LabelMode{T,K,2}

Base.size(::LabelMode) = ()
Base.getindex(lm::LabelMode, idx) = lm

Base.size{T<:LabelMode}(::Type{T}) = ()
Base.getindex{T<:LabelMode}(t::Type{T}, idx) = t

nlabel{T<:BinaryLabelMode}(::Type{T}) = 2
nlabel{T,K,M}(::Type{LabelMode{T,K,M}}) = Int(K)
nlabel(::Type{Any}) = error("nlabel not defined for the given type")
nlabel{T}(::Type{T}) = nlabel(supertype(T))
nlabel{T,K}(::LabelMode{T,K}) = Int(K)

label(itr) = unique(itr)

labeltype{T}(::Type{MatrixLabelMode{T}}) = T
labeltype{T}(::Type{VectorLabelMode{T}}) = T
labeltype{T,K,M}(::Type{LabelMode{T,K,M}}) = T
labeltype(::Type{Any}) = Any
labeltype{T}(::Type{T}) = labeltype(supertype(T))
labeltype{T}(lm::LabelMode{T}) = T

function ind2label end
function label2ind end
function poslabel end
function neglabel end
function labelmode end
function isposlabel end
function isneglabel end
function labeltype end

function classify end
function convertlabel end

