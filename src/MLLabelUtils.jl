module MLLabelUtils

using LearnBase

export

    isposlabel,
    isneglabel,

    LabelModes,
    labelmode

include("learnbase.jl")
include("labelmode.jl")

end # module

