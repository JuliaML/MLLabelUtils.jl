# everything that should move to learnbase

abstract LabelMode{K}
typealias BinaryLabelMode LabelMode{2}

Base.size(::LabelMode) = ()
Base.getindex(lm::LabelMode, idx) = lm

Base.size{T<:LabelMode}(::Type{T}) = ()
Base.getindex{T<:LabelMode}(t::Type{T}, idx) = t

nlabels{K}(::LabelMode{K}) = Int(K)
labels(itr) = unique(itr)
labels{T,N}(A::AbstractArray{T,N}) = unique(A, N) # TODO: Use ObsDim

function poslabel end
function neglabel end
function labelmode end
function isposlabel end
function isneglabel end
function labeltype end

function classify end
function convertlabels end

