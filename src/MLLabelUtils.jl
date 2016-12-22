module MLLabelUtils

using LearnBase

export

    labels,
    nlabels,

    isposlabel,
    isneglabel,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")

end # module

