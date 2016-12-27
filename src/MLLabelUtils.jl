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

    LabelEnc,
    labelenc

include("learnbase.jl")
include("labelencoding.jl")
include("classify.jl")
include("convertlabel.jl")

end # module

