using Test

# check for ambiguities
refambs = detect_ambiguities(Base, Core)
using MLLabelUtils
ambs = detect_ambiguities(MLLabelUtils, Base, Core)
#@test length(setdiff(ambs, refambs)) <= 24 # these are fine for now

tests = [
    "tst_labelencoding.jl"
    "tst_classify.jl"
    "tst_convertlabel.jl"
    "tst_labelmap.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end
