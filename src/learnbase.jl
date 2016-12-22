# everything that should move to learnbase

abstract LabelMode{K}
typealias BinaryLabelMode LabelMode{2}

nlabels{K}(::LabelMode{K}) = Int(K)
labels(itr) = unique(itr)
labels{T,N}(A::AbstractArray{T,N}) = unique(A, N) # TODO: Use ObsDim

function labelmode end
function isposlabel end
function isneglabel end

