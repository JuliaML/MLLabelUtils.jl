@testset "constructor" begin
    @testset "ambiguous" begin
        @test_throws ArgumentError labelenc([1,1,1])
        @test_throws ArgumentError labelenc((1,1,1))
    end

    @testset "FuzzyBinary" begin
        @test LabelEnc.FuzzyBinary <: MLLabelUtils.BinaryLabelEncoding
        @test @inferred(nlabel(LabelEnc.FuzzyBinary())) === 2
        @test @inferred(labeltype(LabelEnc.FuzzyBinary)) <: Any
        @test_throws MethodError label(LabelEnc.FuzzyBinary())
    end

    @testset "TrueFalse" begin
        @test LabelEnc.TrueFalse <: MLLabelUtils.BinaryLabelEncoding
        @test @inferred(nlabel(LabelEnc.TrueFalse)) === 2
        @test @inferred(labeltype(LabelEnc.TrueFalse)) <: Bool
        @test typeof(@inferred(LabelEnc.TrueFalse())) <: LabelEnc.TrueFalse
    end

    @testset "ZeroOne" begin
        @test LabelEnc.ZeroOne <: MLLabelUtils.BinaryLabelEncoding
        @test @inferred(nlabel(LabelEnc.ZeroOne)) === 2
        @test @inferred(nlabel(LabelEnc.ZeroOne{Float32})) === 2
        @test @inferred(labeltype(LabelEnc.ZeroOne)) == Any
        @test @inferred(labeltype(LabelEnc.ZeroOne{Float32})) <: Float32
        @test typeof(@inferred(LabelEnc.ZeroOne{Int,Float32}())) <: LabelEnc.ZeroOne{Int,Float32}
        @test typeof(@inferred(LabelEnc.ZeroOne())) <: LabelEnc.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(0))) <: LabelEnc.ZeroOne{Int,Int}
        @test typeof(@inferred(LabelEnc.ZeroOne(0.0))) <: LabelEnc.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(0.2))) <: LabelEnc.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(1.0))) <: LabelEnc.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(0.2f0))) <: LabelEnc.ZeroOne{Float32,Float32}
        @test_throws AssertionError LabelEnc.ZeroOne(1.1)
        @test_throws AssertionError LabelEnc.ZeroOne(-0.1)
        @test_throws MethodError LabelEnc.ZeroOne(String)
        @test typeof(@inferred(LabelEnc.ZeroOne(Float64))) <: LabelEnc.ZeroOne{Float64,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(UInt8))) <: LabelEnc.ZeroOne{UInt8,Float64}
        @test typeof(@inferred(LabelEnc.ZeroOne(UInt8,0.1f0))) <: LabelEnc.ZeroOne{UInt8,Float32}
        @test LabelEnc.ZeroOne().cutoff === 0.5
        @test LabelEnc.ZeroOne(UInt8).cutoff === 0.5
        @test LabelEnc.ZeroOne(0.1).cutoff === 0.1
        @test LabelEnc.ZeroOne(0.8f0).cutoff === 0.8f0
    end

    @testset "MarginBased" begin
        @test LabelEnc.MarginBased <: MLLabelUtils.BinaryLabelEncoding
        @test @inferred(nlabel(LabelEnc.MarginBased)) === 2
        @test @inferred(nlabel(LabelEnc.MarginBased{Float32})) === 2
        @test @inferred(labeltype(LabelEnc.MarginBased)) == Any
        @test @inferred(labeltype(LabelEnc.MarginBased{Float32})) <: Float32
        @test typeof(@inferred(LabelEnc.MarginBased())) <: LabelEnc.MarginBased{Float64}
        @test typeof(@inferred(LabelEnc.MarginBased(Int))) <: LabelEnc.MarginBased{Int}
        @test typeof(@inferred(LabelEnc.MarginBased(Float64))) <: LabelEnc.MarginBased{Float64}
        @test_throws MethodError LabelEnc.MarginBased(String)
    end

    @testset "OneVsRest" begin
        @test LabelEnc.OneVsRest <: MLLabelUtils.BinaryLabelEncoding
        @test @inferred(nlabel(LabelEnc.OneVsRest)) === 2
        @test @inferred(nlabel(LabelEnc.OneVsRest{Float32})) === 2
        @test @inferred(labeltype(LabelEnc.OneVsRest)) == Any
        @test @inferred(labeltype(LabelEnc.OneVsRest{Symbol})) <: Symbol
        @test_throws MethodError LabelEnc.OneVsRest([1,2])
        @test_throws MethodError LabelEnc.OneVsRest(:yes, nothing)
        @test_throws MethodError LabelEnc.OneVsRest(:yes, "no")
        @test typeof(@inferred(LabelEnc.OneVsRest([1,2],[2,1])))  <: LabelEnc.OneVsRest{Vector{Int}}
        @test typeof(@inferred(LabelEnc.OneVsRest(:yes)))  <: LabelEnc.OneVsRest{Symbol}
        @test typeof(@inferred(LabelEnc.OneVsRest(:yes, :no)))  <: LabelEnc.OneVsRest{Symbol}
        @test typeof(@inferred(LabelEnc.OneVsRest(true)))  <: LabelEnc.OneVsRest{Bool}
        @test typeof(@inferred(LabelEnc.OneVsRest(true, false)))  <: LabelEnc.OneVsRest{Bool}
        @test typeof(@inferred(LabelEnc.OneVsRest("yes"))) <: LabelEnc.OneVsRest{String}
        @test typeof(@inferred(LabelEnc.OneVsRest("yes", "no"))) <: LabelEnc.OneVsRest{String}
        @test typeof(@inferred(LabelEnc.OneVsRest(1)))   <: LabelEnc.OneVsRest{Int}
        @test typeof(@inferred(LabelEnc.OneVsRest(1,4))) <: LabelEnc.OneVsRest{Int}
    end

    @testset "Indices" begin
        @test LabelEnc.Indices <: MLLabelUtils.LabelEncoding
        @test_throws ErrorException nlabel(LabelEnc.Indices)
        @test @inferred(nlabel(LabelEnc.Indices{Float32,4})) === 4
        @test @inferred(labeltype(LabelEnc.Indices)) == Any
        @test @inferred(labeltype(LabelEnc.Indices{UInt8})) <: UInt8
        @test @inferred(labeltype(LabelEnc.Indices{UInt8,3})) <: UInt8
        @test_throws TypeError LabelEnc.Indices(Val{3.})
        @test typeof(@inferred(LabelEnc.Indices(Val{3}))) <: LabelEnc.Indices{Int,3}
        @test typeof(@inferred(LabelEnc.Indices(Val{2}))) <: LabelEnc.Indices{Int,2}
        @test typeof(@inferred(LabelEnc.Indices(Val{2}))) <: MLLabelUtils.BinaryLabelEncoding
        @test typeof(@inferred(LabelEnc.Indices(Float32,Val{5}))) <: LabelEnc.Indices{Float32,5}
        @test typeof(LabelEnc.Indices(3)) <: LabelEnc.Indices{Int,3}
        @test typeof(LabelEnc.Indices(3.)) <: LabelEnc.Indices{Int,3}
        @test typeof(LabelEnc.Indices(UInt8,3.)) <: LabelEnc.Indices{UInt8,3}
        @test typeof(LabelEnc.Indices(Float64,8)) <: LabelEnc.Indices{Float64,8}
    end

    @testset "OneOfK" begin
        @test LabelEnc.OneOfK <: MLLabelUtils.LabelEncoding
        @test_throws ErrorException nlabel(LabelEnc.OneOfK)
        @test @inferred(nlabel(LabelEnc.OneOfK{Float32,4})) === 4
        @test @inferred(labeltype(LabelEnc.OneOfK)) == Any
        @test @inferred(labeltype(LabelEnc.OneOfK{UInt8})) <: UInt8
        @test @inferred(labeltype(LabelEnc.OneOfK{UInt8,3})) <: UInt8
        @test_throws TypeError LabelEnc.OneOfK(Val{3.})
        @test typeof(@inferred(LabelEnc.OneOfK(Val{3}))) <: LabelEnc.OneOfK{Int,3}
        @test typeof(@inferred(LabelEnc.OneOfK(Val{2}))) <: LabelEnc.OneOfK{Int,2}
        @test typeof(@inferred(LabelEnc.OneOfK(Val{2}))) <: MLLabelUtils.BinaryLabelEncoding
        @test typeof(@inferred(LabelEnc.OneOfK(Float32,Val{5}))) <: LabelEnc.OneOfK{Float32,5}
        @test typeof(LabelEnc.OneOfK(3)) <: LabelEnc.OneOfK{Int,3}
        @test typeof(LabelEnc.OneOfK(3.)) <: LabelEnc.OneOfK{Int,3}
        @test typeof(LabelEnc.OneOfK(UInt8,3.)) <: LabelEnc.OneOfK{UInt8,3}
        @test typeof(LabelEnc.OneOfK(Float64,8)) <: LabelEnc.OneOfK{Float64,8}
    end

    @testset "NativeLabels" begin
        @test LabelEnc.NativeLabels <: MLLabelUtils.LabelEncoding
        @test_throws ErrorException nlabel(LabelEnc.NativeLabels)
        @test @inferred(nlabel(LabelEnc.NativeLabels{Float32,4})) === 4
        @test @inferred(labeltype(LabelEnc.NativeLabels)) == Any
        @test @inferred(labeltype(LabelEnc.NativeLabels{String})) <: String
        @test @inferred(labeltype(LabelEnc.NativeLabels{UInt8,3})) <: UInt8
        @test_throws TypeError LabelEnc.NativeLabels{Int,2.}([1,2])
        @test_throws AssertionError LabelEnc.NativeLabels{Int,3}([1,2])
        @test_throws AssertionError LabelEnc.NativeLabels([1,2],Val{3})
        @test typeof(LabelEnc.NativeLabels([5,2,3])) <: LabelEnc.NativeLabels{Int,3}
        @test typeof(LabelEnc.NativeLabels([:a,:b,:c,:d])) <: LabelEnc.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelEnc.NativeLabels{Int,3}([5,2,3]))) <: LabelEnc.NativeLabels{Int,3}
        @test typeof(@inferred(LabelEnc.NativeLabels([5,2,3],Val{3}))) <: LabelEnc.NativeLabels{Int,3}
        @test typeof(@inferred(LabelEnc.NativeLabels{Symbol,4}([:a,:b,:c,:d]))) <: LabelEnc.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelEnc.NativeLabels([:a,:b,:c,:d],Val{4}))) <: LabelEnc.NativeLabels{Symbol,4}
        @test typeof(@inferred(LabelEnc.NativeLabels([1,2],Val{2}))) <: MLLabelUtils.BinaryLabelEncoding
    end
end

@testset "interface" begin
    @testset "FuzzyBinary" begin
        lm = LabelEnc.FuzzyBinary()
        @test labeltype(lm) <: Any
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
        @test typeof(@inferred(labelenc([true,  false, true])))  <: LabelEnc.TrueFalse
        @test typeof(@inferred(labelenc([true,  true,  true])))  <: LabelEnc.TrueFalse
        @test typeof(@inferred(labelenc([false, false, false]))) <: LabelEnc.TrueFalse
        lm = LabelEnc.TrueFalse()
        @test labeltype(lm) <: Bool
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
                    lm = labelenc(targets)
                    @test labeltype(lm) <: T
                    @test typeof(lm) <: LabelEnc.ZeroOne{T,Float64}
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
                    lm = labelenc(targets)
                    @test labeltype(lm) <: T
                    @test typeof(lm) <: LabelEnc.MarginBased{T}
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
        @test_throws MethodError label(LabelEnc.OneVsRest(rand(2,2)))
        @test_throws MethodError label(LabelEnc.OneVsRest(rand(2)))
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
                lm = @inferred LabelEnc.OneVsRest(pos)
                @test labeltype(lm) <: typeof(pos)
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
                    lm = labelenc(targets)
                    @test labeltype(lm) <: T
                    @test typeof(lm) <: LabelEnc.Indices{T,2}
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
                    lm = labelenc(targets)
                    @test labeltype(lm) <: T
                    @test typeof(lm) <: LabelEnc.Indices{T,Int(maximum(targets))}
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
                lm = LabelEnc.OneOfK(T,2)
                @test labeltype(lm) <: T
                @test typeof(lm) <: LabelEnc.OneOfK{T,2}
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
                @test ind2label.([4,1], LabelEnc.OneOfK(5)) == [[0, 0, 0, 1, 0],[1, 0, 0, 0 ,0]]
                @test @inferred(ind2label(4, LabelEnc.OneOfK(5))) == [0, 0, 0, 1, 0]
                @test @inferred(label2ind([0,0,0,1,0], LabelEnc.OneOfK(5))) === 4
                @test @inferred(label2ind(4., LabelEnc.OneOfK(Float64,5))) === 4
                for K in (3,5)
                    lm = LabelEnc.OneOfK(T,K)
                    @test typeof(lm) <: LabelEnc.OneOfK{T,K}
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
                lm = labelenc(T[-2,3])
                @test labeltype(lm) <: T
                @test poslabel(lm) === T(3)
                @test neglabel(lm) === T(-2)
            end
            for targets in ([:yes,:no,:yes], ["yes","yes","no"], [3,-2], [1.4,1.3])
                lm = labelenc(targets)
                @test labeltype(lm) <: eltype(targets)
                @test typeof(lm) <: MLLabelUtils.BinaryLabelEncoding
                @test typeof(lm) <: LabelEnc.NativeLabels{eltype(targets),length(unique(targets))}
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
                lm = labelenc(targets)
                @test typeof(lm) <: LabelEnc.NativeLabels{eltype(targets),length(unique(targets))}
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

