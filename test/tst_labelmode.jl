@testset "constructor" begin
    @testset "ambiguous" begin
        @test_throws ArgumentError labelmode([1,1,1])
        @test_throws ArgumentError labelmode((1,1,1))
    end

    @testset "FuzzyBinary" begin
        @test LabelModes.FuzzyBinary <: MLLabelUtils.LabelMode{2}
        @test_throws MethodError labels(LabelModes.FuzzyBinary())
        @test @inferred(nlabels(LabelModes.FuzzyBinary())) === 2
    end

    @testset "TrueFalse" begin
        @test LabelModes.TrueFalse <: MLLabelUtils.LabelMode{2}
        @test typeof(@inferred(LabelModes.TrueFalse())) <: LabelModes.TrueFalse
    end

    @testset "ZeroOne" begin
        @test LabelModes.ZeroOne <: MLLabelUtils.LabelMode{2}
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
        @test typeof(@inferred(LabelModes.MarginBased())) <: LabelModes.MarginBased{Float64}
        @test typeof(@inferred(LabelModes.MarginBased(Int))) <: LabelModes.MarginBased{Int}
        @test typeof(@inferred(LabelModes.MarginBased(Float64))) <: LabelModes.MarginBased{Float64}
        @test_throws MethodError LabelModes.MarginBased(String)
    end

    @testset "OneVsRest" begin
        @test LabelModes.OneVsRest <: MLLabelUtils.LabelMode{2}
        @test typeof(@inferred(LabelModes.OneVsRest(:yes)))  <: LabelModes.OneVsRest{Symbol}
        @test typeof(@inferred(LabelModes.OneVsRest(true)))  <: LabelModes.OneVsRest{Bool}
        @test typeof(@inferred(LabelModes.OneVsRest("yes"))) <: LabelModes.OneVsRest{String}
        @test typeof(@inferred(LabelModes.OneVsRest(1)))     <: LabelModes.OneVsRest{Int}
    end

    @testset "Indices" begin
        @test LabelModes.Indices <: MLLabelUtils.LabelMode
        @test_throws TypeError LabelModes.Indices(Val{3.})
        @test typeof(@inferred(LabelModes.Indices(Val{3}))) <: LabelModes.Indices{Int,3}
        @test typeof(@inferred(LabelModes.Indices(Val{2}))) <: LabelModes.Indices{Int,2}
        @test typeof(@inferred(LabelModes.Indices(Val{2}))) <: MLLabelUtils.BinaryLabelMode
        @test typeof(@inferred(LabelModes.Indices(Float32,Val{5}))) <: LabelModes.Indices{Float32,5}
        @test typeof(LabelModes.Indices(3)) <: LabelModes.Indices{Int,3}
        @test typeof(LabelModes.Indices(3.)) <: LabelModes.Indices{Int,3}
        @test typeof(LabelModes.Indices(UInt8,3.)) <: LabelModes.Indices{UInt8,3}
        @test typeof(LabelModes.Indices(Float64,8)) <: LabelModes.Indices{Float64,8}
    end
end

@testset "interface" begin
    @testset "FuzzyBinary" begin
        lm = LabelModes.FuzzyBinary()
        @test_throws ArgumentError isposlabel(:yes, lm)
        @test_throws ArgumentError isposlabel("yes", lm)
        @test_throws ArgumentError isneglabel(:no, lm)
        @test_throws ArgumentError isneglabel("no", lm)
        @test @inferred(isposlabel(true, lm)) === true
        @test @inferred(isposlabel(false, lm)) === false
        @test @inferred(isposlabel(1, lm)) === true
        @test @inferred(isposlabel(0.1, lm)) === true
        @test @inferred(isposlabel(0.5, lm)) === true
        @test @inferred(isposlabel(0, lm)) === false
        @test @inferred(isposlabel(-1, lm)) === false
        @test @inferred(isneglabel(true, lm)) === false
        @test @inferred(isneglabel(false, lm)) === true
        @test @inferred(isneglabel(1, lm)) === false
        @test @inferred(isneglabel(0.1, lm)) === false
        @test @inferred(isneglabel(0.5, lm)) === false
        @test @inferred(isneglabel(0, lm)) === true
        @test @inferred(isneglabel(-1, lm)) === true
        @test @inferred(nlabels(lm)) === 2
        @test_throws MethodError poslabel(lm)
        @test_throws MethodError neglabel(lm)
        @test_throws MethodError labels(lm)
    end

    @testset "TrueFalse" begin
        @test typeof(@inferred(labelmode([true,  false, true])))  <: LabelModes.TrueFalse
        @test typeof(@inferred(labelmode([true,  true,  true])))  <: LabelModes.TrueFalse
        @test typeof(@inferred(labelmode([false, false, false]))) <: LabelModes.TrueFalse
        lm = LabelModes.TrueFalse()
        @test_throws MethodError isposlabel(1, lm)
        @test_throws MethodError isposlabel(1., lm)
        @test_throws MethodError isposlabel(:yes, lm)
        @test_throws MethodError isneglabel(0, lm)
        @test_throws MethodError isneglabel(0., lm)
        @test_throws MethodError isneglabel(:no, lm)
        @test @inferred(isposlabel(true, lm))  === true
        @test @inferred(isposlabel(false, lm)) === false
        @test @inferred(isneglabel(true, lm))  === false
        @test @inferred(isneglabel(false, lm)) === true
        @test @inferred(nlabels(lm)) === 2
        @test @inferred(poslabel(lm)) === true
        @test @inferred(neglabel(lm)) === false
        @test @inferred(labels(lm)) == [true, false]
        @test eltype(@inferred(labels(lm))) <: Bool
    end

    @testset "ZeroOne" begin
        for T in (Int, UInt8, Float32, Float64)
            for targets in (T[0,0,0], T[1,0,1], T[0,1,0])
                @testset "$targets" begin
                    lm = labelmode(targets)
                    @test typeof(lm) <: LabelModes.ZeroOne{T,Float64}
                    @test lm.cutoff === 0.5
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test @inferred(isposlabel(true, lm))  === true
                    @test @inferred(isposlabel(false, lm)) === false
                    @test @inferred(isneglabel(true, lm))  === false
                    @test @inferred(isneglabel(false, lm)) === true
                    @test @inferred(isposlabel(one(T), lm))  === true
                    @test @inferred(isposlabel(zero(T), lm)) === false
                    @test @inferred(isneglabel(one(T), lm))  === false
                    @test @inferred(isneglabel(zero(T), lm)) === true
                    @test @inferred(nlabels(lm)) === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(0)
                    @test @inferred(labels(lm)) == T[1, 0]
                    @test eltype(@inferred(labels(lm))) <: T
                end
            end
        end
    end

    @testset "MarginBased" begin
        for T in (Int, Int8, Float32, Float64)
            for targets in (T[-1,-1,-1], T[1,-1,1], T[-1,1,-1])
                @testset "$targets" begin
                    lm = labelmode(targets)
                    @test typeof(lm) <: LabelModes.MarginBased{T}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test @inferred(isposlabel(one(T), lm))  === true
                    @test @inferred(isposlabel(-one(T), lm)) === false
                    @test @inferred(isneglabel(one(T), lm))  === false
                    @test @inferred(isneglabel(-one(T), lm)) === true
                    @test @inferred(nlabels(lm)) === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(-1)
                    @test @inferred(labels(lm)) == T[1, -1]
                    @test eltype(@inferred(labels(lm))) <: T
                end
            end
        end
    end

    @testset "OneVsRest" begin
        @test_throws MethodError labels(LabelModes.OneVsRest(rand(2,2)))
        @test_throws MethodError labels(LabelModes.OneVsRest(rand(2)))
        for (pos, neg) in ((:pos, :not_pos),
                           ("pos", "not_pos"),
                           (true, false),
                           (false, true),
                           (0f0, 1f0),
                           (0x0, 0x01),
                           (2., 0.),
                           (2, 0),
                           (0, 1))
            @testset "pos=$pos, neg=$neg" begin
                lm = @inferred LabelModes.OneVsRest(pos)
                @test @inferred(labels(lm)) == [pos, neg]
                @test @inferred(poslabel(lm)) === pos
                if typeof(neg) <: String
                    @test @inferred(neglabel(lm)) == neg
                else
                    @test @inferred(neglabel(lm)) === neg
                end
                @test @inferred(isposlabel(pos, lm)) === true
                @test @inferred(isposlabel(neg, lm)) === false
                @test @inferred(isneglabel(pos, lm)) === false
                @test @inferred(isneglabel(neg, lm)) === true
            end
        end
    end

    @testset "Indices" begin
        for T in (Int, UInt8, Int8, Float32, Float64)
            for targets in (T[1,2,1], T[2,1,2])
                @testset "binary $targets" begin
                    lm = labelmode(targets)
                    @test typeof(lm) <: LabelModes.Indices{T,2}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test @inferred(isposlabel(one(T), lm))  === true
                    @test @inferred(isposlabel(2one(T), lm)) === false
                    @test @inferred(isneglabel(one(T), lm))  === false
                    @test @inferred(isneglabel(2one(T), lm)) === true
                    @test @inferred(nlabels(lm)) === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(2)
                    @test @inferred(labels(lm)) == T[1, 2]
                    @test eltype(@inferred(labels(lm))) <: T
                end
            end
            for targets in (T[3,1,2,1], T[2,1,4,2])
                @testset "multiclass $targets" begin
                    lm = labelmode(targets)
                    @test typeof(lm) <: LabelModes.Indices{T,Int(maximum(targets))}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test_throws MethodError isposlabel(T(1),lm)
                    @test_throws MethodError isneglabel(T(2),lm)
                    @test_throws MethodError poslabel(lm)
                    @test_throws MethodError neglabel(lm)
                    @test @inferred(nlabels(lm)) === Int(maximum(targets))
                    @test @inferred(labels(lm)) == Vector{T}(collect(1:maximum(targets)))
                    @test eltype(@inferred(labels(lm))) <: T
                end
            end
        end
    end
end

