module MLLabelUtils

using LearnBase

export

    labels,
    nlabels,

    poslabel,
    neglabel,
    isposlabel,
    isneglabel,

    ObsDim,
    classify,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")
include("classify.jl")

end # module

