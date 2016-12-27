@testset "constructor" begin
    @testset "ambiguous" begin
        @test_throws ArgumentError labelmode([1,1,1])
        @test_throws ArgumentError labelmode((1,1,1))
    end

    @testset "FuzzyBinary" begin
        @test LabelModes.FuzzyBinary <: MLLabelUtils.BinaryLabelMode
        @test @inferred(nlabel(LabelModes.FuzzyBinary())) === 2
        @test @inferred(labeltype(LabelModes.FuzzyBinary)) <: Any
        @test_throws MethodError label(LabelModes.FuzzyBinary())
    end

    @testset "TrueFalse" begin
        @test LabelModes.TrueFalse <: MLLabelUtils.BinaryLabelMode
        @test @inferred(nlabel(LabelModes.TrueFalse)) === 2
        @test @inferred(labeltype(LabelModes.TrueFalse)) <: Bool
        @test typeof(@inferred(LabelModes.TrueFalse())) <: LabelModes.TrueFalse
    end

    @testset "ZeroOne" begin
        @test LabelModes.ZeroOne <: MLLabelUtils.BinaryLabelMode
        @test @inferred(nlabel(LabelModes.ZeroOne)) === 2
        @test @inferred(nlabel(LabelModes.ZeroOne{Float32})) === 2
        @test @inferred(labeltype(LabelModes.ZeroOne)) == Any
        @test @inferred(labeltype(LabelModes.ZeroOne{Float32})) <: Float32
        @test typeof(@inferred(LabelModes.ZeroOne{Int,Float32}())) <: LabelModes.ZeroOne{Int,Float32}
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
        @test LabelModes.MarginBased <: MLLabelUtils.BinaryLabelMode
        @test @inferred(nlabel(LabelModes.MarginBased)) === 2
        @test @inferred(nlabel(LabelModes.MarginBased{Float32})) === 2
        @test @inferred(labeltype(LabelModes.MarginBased)) == Any
        @test @inferred(labeltype(LabelModes.MarginBased{Float32})) <: Float32
        @test typeof(@inferred(LabelModes.MarginBased())) <: LabelModes.MarginBased{Float64}
        @test typeof(@inferred(LabelModes.MarginBased(Int))) <: LabelModes.MarginBased{Int}
        @test typeof(@inferred(LabelModes.MarginBased(Float64))) <: LabelModes.MarginBased{Float64}
        @test_throws MethodError LabelModes.MarginBased(String)
    end

    @testset "OneVsRest" begin
        @test LabelModes.OneVsRest <: MLLabelUtils.BinaryLabelMode
        @test @inferred(nlabel(LabelModes.OneVsRest)) === 2
        @test @inferred(nlabel(LabelModes.OneVsRest{Float32})) === 2
        @test @inferred(labeltype(LabelModes.OneVsRest)) == Any
        @test @inferred(labeltype(LabelModes.OneVsRest{Symbol})) <: Symbol
        @test_throws MethodError LabelModes.OneVsRest([1,2])
        @test_throws MethodError LabelModes.OneVsRest(:yes, nothing)
        @test_throws MethodError LabelModes.OneVsRest(:yes, "no")
        @test typeof(@inferred(LabelModes.OneVsRest([1,2],[2,1])))  <: LabelModes.OneVsRest{Vector{Int}}
        @test typeof(@inferred(LabelModes.OneVsRest(:yes)))  <: LabelModes.OneVsRest{Symbol}
        @test typeof(@inferred(LabelModes.OneVsRest(:yes, :no)))  <: LabelModes.OneVsRest{Symbol}
        @test typeof(@inferred(LabelModes.OneVsRest(true)))  <: LabelModes.OneVsRest{Bool}
        @test typeof(@inferred(LabelModes.OneVsRest(true, false)))  <: LabelModes.OneVsRest{Bool}
        @test typeof(@inferred(LabelModes.OneVsRest("yes"))) <: LabelModes.OneVsRest{String}
        @test typeof(@inferred(LabelModes.OneVsRest("yes", "no"))) <: LabelModes.OneVsRest{String}
        @test typeof(@inferred(LabelModes.OneVsRest(1)))   <: LabelModes.OneVsRest{Int}
        @test typeof(@inferred(LabelModes.OneVsRest(1,4))) <: LabelModes.OneVsRest{Int}
    end

    @testset "Indices" begin
        @test LabelModes.Indices <: MLLabelUtils.LabelMode
        @test_throws ErrorException nlabel(LabelModes.Indices)
        @test @inferred(nlabel(LabelModes.Indices{Float32,4})) === 4
        @test @inferred(labeltype(LabelModes.Indices)) == Any
        @test @inferred(labeltype(LabelModes.Indices{UInt8})) <: UInt8
        @test @inferred(labeltype(LabelModes.Indices{UInt8,3})) <: UInt8
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

    @testset "OneOfK" begin
        @test LabelModes.OneOfK <: MLLabelUtils.LabelMode
        @test_throws ErrorException nlabel(LabelModes.OneOfK)
        @test @inferred(nlabel(LabelModes.OneOfK{Float32,4})) === 4
        @test @inferred(labeltype(LabelModes.OneOfK)) == Any
        @test @inferred(labeltype(LabelModes.OneOfK{UInt8})) <: UInt8
        @test @inferred(labeltype(LabelModes.OneOfK{UInt8,3})) <: UInt8
        @test_throws TypeError LabelModes.OneOfK(Val{3.})
        @test typeof(@inferred(LabelModes.OneOfK(Val{3}))) <: LabelModes.OneOfK{Int,3}
        @test typeof(@inferred(LabelModes.OneOfK(Val{2}))) <: LabelModes.OneOfK{Int,2}
        @test typeof(@inferred(LabelModes.OneOfK(Val{2}))) <: MLLabelUtils.BinaryLabelMode
        @test typeof(@inferred(LabelModes.OneOfK(Float32,Val{5}))) <: LabelModes.OneOfK{Float32,5}
        @test typeof(LabelModes.OneOfK(3)) <: LabelModes.OneOfK{Int,3}
        @test typeof(LabelModes.OneOfK(3.)) <: LabelModes.OneOfK{Int,3}
        @test typeof(LabelModes.OneOfK(UInt8,3.)) <: LabelModes.OneOfK{UInt8,3}
        @test typeof(LabelModes.OneOfK(Float64,8)) <: LabelModes.OneOfK{Float64,8}
    end

    @testset "NativeLabels" begin
        @test LabelModes.NativeLabels <: MLLabelUtils.LabelMode
        @test_throws ErrorException nlabel(LabelModes.NativeLabels)
        @test @inferred(nlabel(LabelModes.NativeLabels{Float32,4})) === 4
        @test @inferred(labeltype(LabelModes.NativeLabels)) == Any
        @test @inferred(labeltype(LabelModes.NativeLabels{String})) <: String
        @test @inferred(labeltype(LabelModes.NativeLabels{UInt8,3})) <: UInt8
        @test_throws TypeError LabelModes.NativeLabels{Int,2.}([1,2])
        @test_throws AssertionError LabelModes.NativeLabels{Int,3}([1,2])
        @test_throws AssertionError LabelModes.NativeLabels([1,2],Val{3})
        @test typeof(LabelModes.NativeLabels([5,2,3])) <: LabelModes.NativeLabels{Int,3}
        @test typeof(LabelModes.NativeLabels([:a,:b,:c,:d])) <: LabelModes.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelModes.NativeLabels{Int,3}([5,2,3]))) <: LabelModes.NativeLabels{Int,3}
        @test typeof(@inferred(LabelModes.NativeLabels([5,2,3],Val{3}))) <: LabelModes.NativeLabels{Int,3}
        @test typeof(@inferred(LabelModes.NativeLabels{Symbol,4}([:a,:b,:c,:d]))) <: LabelModes.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelModes.NativeLabels([:a,:b,:c,:d],Val{4}))) <: LabelModes.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelModes.NativeLabels([1,2],Val{2}))) <: MLLabelUtils.BinaryLabelMode
    end
end

@testset "interface" begin
    @testset "FuzzyBinary" begin
        lm = LabelModes.FuzzyBinary()
        @test MLLabelUtils.labeltype(lm) <: Any
        @test_throws ArgumentError isposlabel(:yes, lm)
        @test_throws ArgumentError isposlabel("yes", lm)
        @test_throws ArgumentError isneglabel(:no, lm)
        @test_throws ArgumentError isneglabel("no", lm)
        @test @inferred(isposlabel(true, lm))  === true
        @test @inferred(isposlabel(false, lm)) === false
        @test @inferred(isposlabel(1, lm)) === true
        @test @inferred(isposlabel(0, lm)) === false
        @test @inferred(isposlabel(0.1, lm)) === true
        @test @inferred(isposlabel(0.5, lm)) === true
        @test @inferred(isposlabel(-1, lm))  === false
        @test @inferred(isneglabel(true, lm))  === false
        @test @inferred(isneglabel(false, lm)) === true
        @test @inferred(isneglabel(1, lm)) === false
        @test @inferred(isneglabel(0, lm)) === true
        @test @inferred(isneglabel(0.1, lm)) === false
        @test @inferred(isneglabel(0.5, lm)) === false
        @test @inferred(isneglabel(-1, lm))  === true
        @test @inferred(nlabel(lm)) === 2
        @test_throws MethodError poslabel(lm)
        @test_throws MethodError neglabel(lm)
        @test_throws MethodError label(lm)
    end

    @testset "TrueFalse" begin
        @test typeof(@inferred(labelmode([true,  false, true])))  <: LabelModes.TrueFalse
        @test typeof(@inferred(labelmode([true,  true,  true])))  <: LabelModes.TrueFalse
        @test typeof(@inferred(labelmode([false, false, false]))) <: LabelModes.TrueFalse
        lm = LabelModes.TrueFalse()
        @test MLLabelUtils.labeltype(lm) <: Bool
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
        @test @inferred(nlabel(lm))  === 2
        @test @inferred(poslabel(lm)) === true
        @test @inferred(neglabel(lm)) === false
        @test @inferred(label(lm)) == [true, false]
        @test eltype(@inferred(label(lm))) <: Bool
    end

    @testset "ZeroOne" begin
        for T in (Int, UInt8, Float32, Float64)
            for targets in (T[0,0,0], T[1,0,1], T[0,1,0])
                @testset "$targets" begin
                    lm = labelmode(targets)
                    @test MLLabelUtils.labeltype(lm) <: T
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
                    @test @inferred(nlabel(lm))  === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(0)
                    @test @inferred(label(lm)) == T[1, 0]
                    @test eltype(@inferred(label(lm))) <: T
                end
            end
        end
    end

    @testset "MarginBased" begin
        for T in (Int, Int8, Float32, Float64)
            for targets in (T[-1,-1,-1], T[1,-1,1], T[-1,1,-1])
                @testset "$targets" begin
                    lm = labelmode(targets)
                    @test MLLabelUtils.labeltype(lm) <: T
                    @test typeof(lm) <: LabelModes.MarginBased{T}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test @inferred(isposlabel(zero(T), lm)) === true
                    @test @inferred(isposlabel(one(T), lm))  === true
                    @test @inferred(isposlabel(-one(T), lm)) === false
                    @test @inferred(isneglabel(one(T), lm))  === false
                    @test @inferred(isneglabel(-one(T), lm)) === true
                    @test @inferred(nlabel(lm))  === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(-1)
                    @test @inferred(label(lm)) == T[1, -1]
                    @test eltype(@inferred(label(lm))) <: T
                end
            end
        end
    end

    @testset "OneVsRest" begin
        @test_throws MethodError label(LabelModes.OneVsRest(rand(2,2)))
        @test_throws MethodError label(LabelModes.OneVsRest(rand(2)))
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
                @test MLLabelUtils.labeltype(lm) <: typeof(pos)
                @test @inferred(label(lm)) == [pos, neg]
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
                    @test MLLabelUtils.labeltype(lm) <: T
                    @test typeof(lm) <: LabelModes.Indices{T,2}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test @inferred(isposlabel(one(T), lm))  === true
                    @test @inferred(isposlabel(2one(T), lm)) === false
                    @test @inferred(isneglabel(one(T), lm))  === false
                    @test @inferred(isneglabel(2one(T), lm)) === true
                    @test @inferred(nlabel(lm))  === 2
                    @test @inferred(poslabel(lm)) === T(1)
                    @test @inferred(neglabel(lm)) === T(2)
                    @test @inferred(label(lm)) == T[1, 2]
                    @test eltype(@inferred(label(lm))) <: T
                end
            end
            for targets in (T[3,1,2,1], T[2,1,4,2])
                @testset "multiclass $targets" begin
                    lm = labelmode(targets)
                    @test MLLabelUtils.labeltype(lm) <: T
                    @test typeof(lm) <: LabelModes.Indices{T,Int(maximum(targets))}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test_throws MethodError isposlabel(T(1),lm)
                    @test_throws MethodError isneglabel(T(2),lm)
                    @test_throws MethodError poslabel(lm)
                    @test_throws MethodError neglabel(lm)
                    @test @inferred(nlabel(lm)) === Int(maximum(targets))
                    @test @inferred(label(lm)) == Vector{T}(collect(1:maximum(targets)))
                    @test eltype(@inferred(label(lm))) <: T
                end
            end
        end
    end

    @testset "OneOfK" begin
        for T in (Int, UInt8, Int8, Float32, Float64)
            @testset "binary T = $T" begin
                lm = LabelModes.OneOfK(T,2)
                @test MLLabelUtils.labeltype(lm) <: T
                @test typeof(lm) <: LabelModes.OneOfK{T,2}
                @test_throws MethodError isposlabel(:yes, lm)
                @test_throws MethodError isneglabel(:neg, lm)
                @test_throws MethodError isposlabel(true, lm)
                @test_throws MethodError isneglabel(false, lm)
                @test @inferred(isposlabel(1, lm)) === true
                @test @inferred(isposlabel(2, lm)) === false
                @test @inferred(isneglabel(1, lm)) === false
                @test @inferred(isneglabel(2, lm)) === true
                @test @inferred(nlabel(lm))  === 2
                @test @inferred(poslabel(lm)) == [1, 0]
                @test @inferred(neglabel(lm)) == [0, 1]
                @test @inferred(label(lm)) == [1, 2]
                @test eltype(@inferred(label(lm))) <: Int
                @test @inferred(isposlabel([.6,.4], lm)) === true
                @test @inferred(isposlabel([.3,.7], lm)) === false
                @test @inferred(isneglabel([.9,.1], lm)) === false
                @test @inferred(isneglabel([2,4], lm)) === true
                for R in (Bool, Int, Float64)
                    @test @inferred(isposlabel(R[1,0], lm)) === true
                    @test @inferred(isposlabel(R[0,1], lm)) === false
                    @test @inferred(isneglabel(R[1,0], lm)) === false
                    @test @inferred(isneglabel(R[0,1], lm)) === true
                end
            end
            @testset "multiclass T = $T" begin
                @test @inferred(ind2label(4, LabelModes.OneOfK(5))) == [0, 0, 0, 1, 0]
                for K in (3,5)
                    lm = LabelModes.OneOfK(T,K)
                    @test typeof(lm) <: LabelModes.OneOfK{T,K}
                    @test_throws MethodError isposlabel(:yes, lm)
                    @test_throws MethodError isneglabel(:neg, lm)
                    @test_throws MethodError isposlabel(true, lm)
                    @test_throws MethodError isneglabel(false, lm)
                    @test_throws MethodError isposlabel(T(1),lm)
                    @test_throws MethodError isneglabel(T(2),lm)
                    @test_throws MethodError poslabel(lm)
                    @test_throws MethodError neglabel(lm)
                    @test @inferred(nlabel(lm)) === K
                    @test @inferred(label(lm)) == collect(1:K)
                    @test eltype(@inferred(label(lm))) <: Int
                end
            end
        end
    end

    @testset "NativeLabels" begin
        @testset "binary" begin
            for T in (Float64,Int,Float32)
                lm = labelmode(T[-2,3])
                @test MLLabelUtils.labeltype(lm) <: T
                @test poslabel(lm) === T(3)
                @test neglabel(lm) === T(-2)
            end
            for targets in ([:yes,:no,:yes], ["yes","yes","no"], [3,-2], [1.4,1.3])
                lm = labelmode(targets)
                @test MLLabelUtils.labeltype(lm) <: eltype(targets)
                @test typeof(lm) <: MLLabelUtils.BinaryLabelMode
                @test typeof(lm) <: LabelModes.NativeLabels{eltype(targets),length(unique(targets))}
                @test @inferred(isposlabel(label(lm)[1], lm)) === true
                @test @inferred(isposlabel(label(lm)[2], lm)) === false
                @test @inferred(isposlabel(:nix, lm)) === false
                @test @inferred(isposlabel(1, lm)) === false
                @test @inferred(isposlabel(2, lm)) === false
                @test @inferred(isneglabel(label(lm)[1], lm)) === false
                @test @inferred(isneglabel(label(lm)[2], lm)) === true
                @test @inferred(isneglabel(:nix, lm)) === false
                @test @inferred(isneglabel(1, lm)) === false
                @test @inferred(isneglabel(2, lm)) === false
                @test lm.label == unique(targets)
                @test @inferred(label(lm)) == unique(targets)
                @test eltype(@inferred(label(lm))) <: eltype(targets)
                @test @inferred(nlabel(lm)) === length(unique(targets))
            end
        end
        @testset "multiclass"  begin
            for targets in ([:yes,:maybe,:no,:yes], ["yes","noidea","yes","no"], [1.2,3,2], rand(50))
                lm = labelmode(targets)
                @test typeof(lm) <: LabelModes.NativeLabels{eltype(targets),length(unique(targets))}
                @test_throws MethodError isposlabel(:yes, lm)
                @test_throws MethodError isneglabel(:neg, lm)
                @test_throws MethodError isposlabel(true, lm)
                @test_throws MethodError isneglabel(false, lm)
                @test_throws MethodError isposlabel(1,lm)
                @test_throws MethodError isneglabel(2,lm)
                @test_throws MethodError poslabel(lm)
                @test_throws MethodError neglabel(lm)
                @test lm.label == unique(targets)
                @test @inferred(label(lm)) == unique(targets)
                @test eltype(@inferred(label(lm))) <: eltype(targets)
                @test @inferred(nlabel(lm)) === length(unique(targets))
            end
        end
    end
end

