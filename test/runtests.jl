using MLLabelUtils
using Base.Test

tests = [
    "tst_labelmode.jl"
]

for t in tests
    @testset "$t" begin
        include(t)
    end
end

