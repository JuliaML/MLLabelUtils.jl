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
    classify!,

    convertlabel,
#    convertlabel!,

    LabelEnc,
    labelenc,
    islabelenc

include("learnbase.jl")
include("labelencoding.jl")
include("classify.jl")
include("convertlabel.jl")

end # module

