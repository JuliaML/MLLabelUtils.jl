@testset "ZeroOne" begin
    @test @inferred(classify(0.2f0, LabelModes.ZeroOne(0.3))) === 0.
    @test @inferred(classify(0.4f0, LabelModes.ZeroOne(0.3))) === 1.
    @test classify.([0.2,0.4,0.6], LabelModes.ZeroOne(0.3)) == [0,1,1]
    @test eltype(classify.([0.2,0.4,0.6], LabelModes.ZeroOne(0.3))) <: Float64
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T(0.4), 0.5)) === zero(T)
        @test @inferred(classify(T(0.5), 0.5)) === one(T)
        @test @inferred(classify(T(0.4), 0.3)) === one(T)
        @test @inferred(classify(T(0.4), LabelModes.ZeroOne)) === zero(T)
        @test @inferred(classify(T(0.6), LabelModes.ZeroOne)) === one(T)
        # broadcast
        @test classify.(T[0.4,0.6], 0.5) == [0,1]
        @test classify.(T[0.4,0.6], LabelModes.ZeroOne) == [0,1]
        @test eltype(classify.(T[0.4,0.6], 0.5)) <: T
        @test eltype(classify.(T[0.4,0.6], LabelModes.ZeroOne)) <: T
    end
    for T in (Float16, Float32, Float64, UInt8, Int32, Int64)
        @test @inferred(classify(0.4, LabelModes.ZeroOne{T})) === zero(T)
        @test @inferred(classify(0.6, LabelModes.ZeroOne{T})) === one(T)
        @test @inferred(classify(0.2f0, LabelModes.ZeroOne(T,0.3))) === zero(T)
        @test @inferred(classify(0.4f0, LabelModes.ZeroOne(T,0.3))) === one(T)
        @test @inferred(classify(0.2f0, LabelModes.ZeroOne(T))) === zero(T)
        @test @inferred(classify(0.4f0, LabelModes.ZeroOne(T))) === zero(T)
        @test @inferred(classify(0.6f0, LabelModes.ZeroOne(T))) === one(T)
        # broadcast
        @test classify.([0.4,0.6], LabelModes.ZeroOne{T}) == [0,1]
        @test classify.([0.4,0.6], LabelModes.ZeroOne(T)) == [0,1]
        @test classify.([0.2,0.4,0.6], LabelModes.ZeroOne(T,0.3)) == [0,1,1]
        @test eltype(classify.([0.4,0.6], LabelModes.ZeroOne{T})) <: T
        @test eltype(classify.([0.4,0.6], LabelModes.ZeroOne(T))) <: T
        @test eltype(classify.([0.2,0.4,0.6], LabelModes.ZeroOne(T,0.3))) <: T
    end
end

@testset "MarginBased" begin
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T(0.2),  LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(0.6),  LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(0.0),  LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(-0.1), LabelModes.MarginBased)) === -one(T)
        @test @inferred(classify(T(-5.1), LabelModes.MarginBased)) === -one(T)
        @test classify.(T[0,-.1,0.2,3,-4], LabelModes.MarginBased) == [1,-1,1,1,-1]
        @test eltype(classify.(T[0,-.1,0.2,3,-4], LabelModes.MarginBased)) <: T
    end
    for T in (Int16, Int32, Int64)
        @test @inferred(classify(T(12), LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(2),  LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(0),  LabelModes.MarginBased)) === one(T)
        @test @inferred(classify(T(-1), LabelModes.MarginBased)) === -one(T)
        @test @inferred(classify(T(-5), LabelModes.MarginBased)) === -one(T)
        @test classify.(T[0,-1,2,3,-4], LabelModes.MarginBased) == [1,-1,1,1,-1]
        @test eltype(classify.(T[0,-1,2,3,-4], LabelModes.MarginBased)) <: T
    end
    for T in (Float16, Float32, Float64, Int16, Int32, Int64)
        @test @inferred(classify(12,  LabelModes.MarginBased{T})) === one(T)
        @test @inferred(classify(2f0, LabelModes.MarginBased{T})) === one(T)
        @test @inferred(classify(0,   LabelModes.MarginBased{T})) === one(T)
        @test @inferred(classify(-1., LabelModes.MarginBased{T})) === -one(T)
        @test @inferred(classify(-5,  LabelModes.MarginBased{T})) === -one(T)
        @test @inferred(classify(12,  LabelModes.MarginBased(T))) === one(T)
        @test @inferred(classify(2f0, LabelModes.MarginBased(T))) === one(T)
        @test @inferred(classify(0,   LabelModes.MarginBased(T))) === one(T)
        @test @inferred(classify(-1., LabelModes.MarginBased(T))) === -one(T)
        @test @inferred(classify(-5,  LabelModes.MarginBased(T))) === -one(T)
        @test classify.([0,-.1,0.2,3,-4], LabelModes.MarginBased{T}) == [1,-1,1,1,-1]
        @test classify.([0,-.1,0.2,3,-4], LabelModes.MarginBased(T)) == [1,-1,1,1,-1]
        @test eltype(classify.([0,-.1,0.2,3,-4], LabelModes.MarginBased{T})) <: T
        @test eltype(classify.([0,-.1,0.2,3,-4], LabelModes.MarginBased(T))) <: T
    end
end

@testset "OneOfK" begin
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelModes.OneOfK)) === 3
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelModes.OneOfK{T})) === 3
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelModes.OneOfK(Val{4}))) === 3
    end
    for T in (Int16, Int32, Int64)
        @test @inferred(classify(T[4,2,9,10,2], LabelModes.OneOfK)) === 4
        @test @inferred(classify(T[4,2,9,10,2], LabelModes.OneOfK{T})) === 4
        @test @inferred(classify(T[4,2,9,10,2], LabelModes.OneOfK(Val{4}))) === 4
    end
    for T in (Float16, Float32, Float64, Int16, Int32, Int64)
        A = T[2 1 0 2 1 0 1;
              1 0 2 1 2 1 2;
              0 2 1 0 0 2 0]
        @test @inferred(classify(A, LabelModes.OneOfK)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelModes.OneOfK, ObsDim.Last())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelModes.OneOfK, obsdim=2)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelModes.OneOfK(Val{3}))) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelModes.OneOfK(Val{3}), ObsDim.Last())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelModes.OneOfK(Val{3}), obsdim=2)) == [1,3,2,1,2,3,2]
        @test eltype(classify(A, LabelModes.OneOfK)) <: Int
        @test eltype(classify(A, LabelModes.OneOfK, ObsDim.Last())) <: Int
        @test eltype(classify(A, LabelModes.OneOfK, obsdim=2)) <: Int
        @test eltype(classify(A, LabelModes.OneOfK(Val{3}))) <: Int
        @test eltype(classify(A, LabelModes.OneOfK(Val{3}), ObsDim.Last())) <: Int
        @test eltype(classify(A, LabelModes.OneOfK(Val{3}), obsdim=2)) <: Int
        At = A'
        @test @inferred(classify(At, LabelModes.OneOfK, ObsDim.First())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelModes.OneOfK, obsdim=1)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelModes.OneOfK(Val{3}), ObsDim.First())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelModes.OneOfK(Val{3}), obsdim=1)) == [1,3,2,1,2,3,2]
        @test eltype(classify(At, LabelModes.OneOfK, ObsDim.First())) <: Int
        @test eltype(classify(At, LabelModes.OneOfK, obsdim=1)) <: Int
        @test eltype(classify(At, LabelModes.OneOfK(Val{3}))) <: Int
        @test eltype(classify(At, LabelModes.OneOfK(Val{3}), ObsDim.First())) <: Int
        @test eltype(classify(At, LabelModes.OneOfK(Val{3}), obsdim=1)) <: Int
    end
end

