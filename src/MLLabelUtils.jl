module MLLabelUtils

using StatsBase
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

    labelmap,
    labelmap!,
    labelfreq,
    labelfreq!,

    LabelEnc,
    labelenc,
    islabelenc

include("learnbase.jl")
include("labelencoding.jl")
include("classify.jl")
include("convertlabel.jl")
include("labelmap.jl")

end # module

