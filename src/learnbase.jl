# everything that should move to learnbase

abstract LabelEncoding{T,K,M} # Eltype, Labelcount, Arraydimensions
typealias BinaryLabelEncoding{T,M} LabelEncoding{T,2,M}
typealias VectorLabelEncoding{T,K} LabelEncoding{T,K,1}
typealias MatrixLabelEncoding{T,K} LabelEncoding{T,K,2}

Base.size(::LabelEncoding) = ()
Base.getindex(lm::LabelEncoding, idx) = lm

Base.size{T<:LabelEncoding}(::Type{T}) = ()
Base.getindex{T<:LabelEncoding}(t::Type{T}, idx) = t

nlabel{T<:BinaryLabelEncoding}(::Type{T}) = 2
nlabel{T,K,M}(::Type{LabelEncoding{T,K,M}}) = Int(K)
nlabel(::Type{Any}) = error("nlabel not defined for the given type")
nlabel{T}(::Type{T}) = nlabel(supertype(T))
nlabel{T,K}(::LabelEncoding{T,K}) = Int(K)

label(itr) = unique(itr)

labeltype{T}(::Type{MatrixLabelEncoding{T}}) = T
labeltype{T}(::Type{VectorLabelEncoding{T}}) = T
labeltype{T,K,M}(::Type{LabelEncoding{T,K,M}}) = T
labeltype(::Type{Any}) = Any
labeltype{T}(::Type{T}) = labeltype(supertype(T))
labeltype{T}(lm::LabelEncoding{T}) = T

function ind2label end
function label2ind end
function poslabel end
function neglabel end
function labelenc end
function isposlabel end
function isneglabel end
function labeltype end

function classify end
function convertlabel end

