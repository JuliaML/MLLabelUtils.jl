module MLLabelUtils

using LearnBase

export

    ind2label,
    label2ind,

    labeltype,

    label,
    nlabel,

    poslabel,
    neglabel,
    isposlabel,
    isneglabel,

    ObsDim,
    classify,

    convertlabel,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")
include("classify.jl")
include("convertlabel.jl")

end # module

