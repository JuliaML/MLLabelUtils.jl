@testset "constructor" begin
    @testset "FuzzyBinary" begin
        @test LabelModes.FuzzyBinary <: MLLabelUtils.LabelMode{2}
        @test_throws MethodError labels(LabelModes.FuzzyBinary())
        @test @inferred(nlabels(LabelModes.FuzzyBinary())) === 2
    end

    @testset "TrueFalse" begin
        @test LabelModes.TrueFalse <: MLLabelUtils.LabelMode{2}
        @test typeof(@inferred(labelmode([true, false, true]))) <: LabelModes.TrueFalse
        @test typeof(@inferred(labelmode([true, true, true]))) <: LabelModes.TrueFalse
        @test @inferred(nlabels(LabelModes.TrueFalse())) === 2
        @test @inferred(labels(LabelModes.TrueFalse())) == [true, false]
        @test typeof(@inferred(LabelModes.TrueFalse())) <: LabelModes.TrueFalse
    end

    @testset "ZeroOne" begin
        @test LabelModes.ZeroOne <: MLLabelUtils.LabelMode{2}
        targets = [0, 0, 0]
        @test typeof(labelmode(targets)) <: LabelModes.ZeroOne{Int,Float64}
        @test labelmode(targets).cutoff === 0.5
        @test labels(labelmode(targets)) == [1, 0]
        @test @inferred(nlabels(labelmode(targets))) === 2
        targets = [0, 1, 1]
        @test typeof(labelmode(targets)) <: LabelModes.ZeroOne{Int,Float64}
        @test labels(labelmode(targets)) == [1, 0]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Int
        targets = [0., 1., 1.]
        @test typeof(labelmode(targets)) <: LabelModes.ZeroOne{Float64,Float64}
        @test labels(labelmode(targets)) == [1., 0.]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Float64
        targets = [0.f0, 1f0, 1f0]
        @test typeof(labelmode(targets)) <: LabelModes.ZeroOne{Float32,Float64}
        @test labels(labelmode(targets)) == [1f0, 0f0]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Float32

        @test typeof(@inferred(LabelModes.ZeroOne())) <: LabelModes.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(0))) <: LabelModes.ZeroOne{Int,Int}
        @test typeof(@inferred(LabelModes.ZeroOne(0.0))) <: LabelModes.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(0.2))) <: LabelModes.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(1.0))) <: LabelModes.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(0.2f0))) <: LabelModes.ZeroOne{Float32,Float32}
        @test_throws AssertionError LabelModes.ZeroOne(1.1)
        @test_throws AssertionError LabelModes.ZeroOne(-0.1)
        @test_throws MethodError LabelModes.ZeroOne(String)
        @test typeof(@inferred(LabelModes.ZeroOne(Float64))) <: LabelModes.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(UInt8))) <: LabelModes.ZeroOne{UInt8,Float64}
        @test typeof(@inferred(LabelModes.ZeroOne(UInt8,0.1f0))) <: LabelModes.ZeroOne{UInt8,Float32}
        @test LabelModes.ZeroOne().cutoff === 0.5
        @test LabelModes.ZeroOne(UInt8).cutoff === 0.5
        @test LabelModes.ZeroOne(0.1).cutoff === 0.1
        @test LabelModes.ZeroOne(0.8f0).cutoff === 0.8f0
    end

    @testset "MarginBased" begin
        @test LabelModes.MarginBased <: MLLabelUtils.LabelMode{2}
        targets = [-1, -1, -1]
        @test typeof(labelmode(targets)) <: LabelModes.MarginBased{Int}
        @test labels(labelmode(targets)) == [1, -1]
        @test @inferred(nlabels(labelmode(targets))) === 2
        targets = [-1, 1, 1]
        @test typeof(labelmode(targets)) <: LabelModes.MarginBased{Int}
        @test labels(labelmode(targets)) == [1, -1]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Int
        targets = [-1., 1., 1.]
        @test typeof(labelmode(targets)) <: LabelModes.MarginBased{Float64}
        @test labels(labelmode(targets)) == [1., -1.]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Float64
        targets = [-1.f0, 1.f0, 1.f0]
        @test typeof(labelmode(targets)) <: LabelModes.MarginBased{Float32}
        @test labels(labelmode(targets)) == [1.f0, -1.f0]
        @test @inferred(nlabels(labelmode(targets))) === 2
        @test eltype(@inferred(labels(labelmode(targets)))) <: Float32

        @test_throws MethodError LabelModes.MarginBased(String)
        @test typeof(@inferred(LabelModes.MarginBased())) <: LabelModes.MarginBased{Float64}
        @test typeof(@inferred(LabelModes.MarginBased(Int))) <: LabelModes.MarginBased{Int}
        @test typeof(@inferred(LabelModes.MarginBased(Float64))) <: LabelModes.MarginBased{Float64}
    end

end

