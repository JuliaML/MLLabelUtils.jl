using MLLabelUtils
using Base.Test

tests = [
    "tst_labelencoding.jl"
    "tst_classify.jl"
    "tst_convertlabel.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end

