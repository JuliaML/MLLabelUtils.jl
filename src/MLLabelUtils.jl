module MLLabelUtils

using LearnBase

export

    labels,
    nlabels,

    poslabel,
    neglabel,
    poslabel,
    neglabel,
    isposlabel,
    isneglabel,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")

end # module

