using MLLabelUtils
using Base.Test

tests = [
    "tst_labelmode.jl"
    "tst_classify.jl"
    "tst_convertlabel.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end

