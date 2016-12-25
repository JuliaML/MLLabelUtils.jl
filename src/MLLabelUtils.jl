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

    convertlabels,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")
include("classify.jl")
include("convertlabels.jl")

end # module

