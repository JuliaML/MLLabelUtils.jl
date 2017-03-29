@testset "ZeroOne" begin
    @test @inferred(classify(0.2f0, LabelEnc.ZeroOne(0.3))) === 0.
    @test @inferred(classify(0.4f0, LabelEnc.ZeroOne(0.3))) === 1.
    @test classify.([0.2,0.4,0.6], LabelEnc.ZeroOne(0.3)) == [0,1,1]
    @test eltype(classify.([0.2,0.4,0.6], LabelEnc.ZeroOne(0.3))) <: Float64
    @test @inferred(classify([0.2,0.4,0.6], LabelEnc.ZeroOne(0.3))) == [0,1,1]
    @test eltype(classify([0.2,0.4,0.6], LabelEnc.ZeroOne(0.3))) <: Float64
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T(0.4), 0.5)) === zero(T)
        @test @inferred(classify(T(0.5), 0.5)) === one(T)
        @test @inferred(classify(T(0.4), 0.3)) === one(T)
        @test @inferred(classify(T(0.4), LabelEnc.ZeroOne)) === zero(T)
        @test @inferred(classify(T(0.6), LabelEnc.ZeroOne)) === one(T)
        # broadcast
        @test classify.(T[0.4,0.6], 0.5) == [0,1]
        @test classify.(T[0.4,0.6], LabelEnc.ZeroOne) == [0,1]
        @test @inferred(classify(T[0.4,0.6], 0.5)) == [0,1]
        @test @inferred(classify(T[0.4,0.6], LabelEnc.ZeroOne)) == [0,1]
        @test eltype(classify.(T[0.4,0.6], 0.5)) <: T
        @test eltype(classify.(T[0.4,0.6], LabelEnc.ZeroOne)) <: T
        @test eltype(classify(T[0.4,0.6], 0.5)) <: T
        @test eltype(classify(T[0.4,0.6], LabelEnc.ZeroOne)) <: T
        buffer = zeros(2)
        @test @inferred(classify!(buffer, [0.4,0.6], 0.5)) == [0,1]
        @test buffer == [0,1]
        buffer = zeros(2)
        @test @inferred(classify!(buffer, [0.4,0.6], LabelEnc.ZeroOne)) == [0,1]
        @test buffer == [0,1]
    end
    for T in (Float16, Float32, Float64, UInt8, Int32, Int64)
        @test @inferred(classify(0.2f0, LabelEnc.ZeroOne(T,0.3))) === zero(T)
        @test @inferred(classify(0.4f0, LabelEnc.ZeroOne(T,0.3))) === one(T)
        @test @inferred(classify(0.2f0, LabelEnc.ZeroOne(T))) === zero(T)
        @test @inferred(classify(0.4f0, LabelEnc.ZeroOne(T))) === zero(T)
        @test @inferred(classify(0.6f0, LabelEnc.ZeroOne(T))) === one(T)
        # broadcast
        @test @inferred(classify([0.4,0.6], LabelEnc.ZeroOne(T))) == [0,1]
        @test classify.([0.4,0.6], LabelEnc.ZeroOne(T)) == [0,1]
        @test classify.([0.2,0.4,0.6], LabelEnc.ZeroOne(T,0.3)) == [0,1,1]
        @test eltype(classify([0.4,0.6], LabelEnc.ZeroOne(T))) <: T
        @test eltype(classify.([0.4,0.6], LabelEnc.ZeroOne(T))) <: T
        @test eltype(classify.([0.2,0.4,0.6], LabelEnc.ZeroOne(T,0.3))) <: T
        buffer = zeros(2)
        @test @inferred(classify!(buffer, [0.4,0.6], LabelEnc.ZeroOne(T))) == [0,1]
        @test buffer == [0,1]
    end
end

@testset "MarginBased" begin
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T(0.2),  LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(0.6),  LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(0.0),  LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(-0.1), LabelEnc.MarginBased)) === -one(T)
        @test @inferred(classify(T(-5.1), LabelEnc.MarginBased)) === -one(T)
        @test classify.(T[0,-.1,0.2,3,-4], LabelEnc.MarginBased) == [1,-1,1,1,-1]
        @test @inferred(classify(T[0,-.1,0.2,3,-4], LabelEnc.MarginBased)) == [1,-1,1,1,-1]
        @test eltype(classify.(T[0,-.1,0.2,3,-4], LabelEnc.MarginBased)) <: T
        @test eltype(classify(T[0,-.1,0.2,3,-4], LabelEnc.MarginBased)) <: T
        buffer = zeros(5)
        @test @inferred(classify!(buffer, T[0,-.1,0.2,3,-4], LabelEnc.MarginBased)) == [1,-1,1,1,-1]
        @test buffer == [1,-1,1,1,-1]
    end
    for T in (Int16, Int32, Int64)
        @test @inferred(classify(T(12), LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(2),  LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(0),  LabelEnc.MarginBased)) === one(T)
        @test @inferred(classify(T(-1), LabelEnc.MarginBased)) === -one(T)
        @test @inferred(classify(T(-5), LabelEnc.MarginBased)) === -one(T)
        @test classify.(T[0,-1,2,3,-4], LabelEnc.MarginBased) == [1,-1,1,1,-1]
        @test @inferred(classify(T[0,-1,2,3,-4], LabelEnc.MarginBased)) == [1,-1,1,1,-1]
        @test eltype(classify.(T[0,-1,2,3,-4], LabelEnc.MarginBased)) <: T
        @test eltype(classify(T[0,-1,2,3,-4], LabelEnc.MarginBased)) <: T
    end
    for T in (Float16, Float32, Float64, Int16, Int32, Int64)
        @test @inferred(classify(12,  LabelEnc.MarginBased(T))) === one(T)
        @test @inferred(classify(2f0, LabelEnc.MarginBased(T))) === one(T)
        @test @inferred(classify(0,   LabelEnc.MarginBased(T))) === one(T)
        @test @inferred(classify(-1., LabelEnc.MarginBased(T))) === -one(T)
        @test @inferred(classify(-5,  LabelEnc.MarginBased(T))) === -one(T)
        @test classify.([0,-.1,0.2,3,-4], LabelEnc.MarginBased(T)) == [1,-1,1,1,-1]
        @test @inferred(classify([0,-.1,0.2,3,-4], LabelEnc.MarginBased(T))) == [1,-1,1,1,-1]
        @test eltype(classify.([0,-.1,0.2,3,-4], LabelEnc.MarginBased(T))) <: T
        @test eltype(classify([0,-.1,0.2,3,-4], LabelEnc.MarginBased(T))) <: T
    end
end

@testset "OneOfK" begin
    for T in (Float16, Float32, Float64)
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelEnc.OneOfK)) === 3
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelEnc.OneOfK{T})) === 3
        @test @inferred(classify(T[0.4,0.2,0.9,.1], LabelEnc.OneOfK(Val{4}))) === 3
    end
    for T in (Int16, Int32, Int64)
        @test @inferred(classify(T[4,2,9,10,2], LabelEnc.OneOfK)) === 4
        @test @inferred(classify(T[4,2,9,10,2], LabelEnc.OneOfK{T})) === 4
        @test @inferred(classify(T[4,2,9,10,2], LabelEnc.OneOfK(Val{4}))) === 4
    end
    for T in (Float16, Float32, Float64, Int16, Int32, Int64)
        A = T[2 1 0 2 1 0 1;
              1 0 2 1 2 1 2;
              0 2 1 0 0 2 0]
        @test @inferred(classify(A, LabelEnc.OneOfK)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelEnc.OneOfK, ObsDim.Last())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelEnc.OneOfK, obsdim=2)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelEnc.OneOfK(Val{3}))) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelEnc.OneOfK(Val{3}), ObsDim.Last())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(A, LabelEnc.OneOfK(Val{3}), obsdim=2)) == [1,3,2,1,2,3,2]
        @test eltype(classify(A, LabelEnc.OneOfK)) <: Int
        @test eltype(classify(A, LabelEnc.OneOfK, ObsDim.Last())) <: Int
        @test eltype(classify(A, LabelEnc.OneOfK, obsdim=2)) <: Int
        @test eltype(classify(A, LabelEnc.OneOfK(Val{3}))) <: Int
        @test eltype(classify(A, LabelEnc.OneOfK(Val{3}), ObsDim.Last())) <: Int
        @test eltype(classify(A, LabelEnc.OneOfK(Val{3}), obsdim=2)) <: Int
        buffer = zeros(7)
        @test @inferred(classify!(buffer, A, LabelEnc.OneOfK(3), ObsDim.Last())) == [1,3,2,1,2,3,2]
        @test buffer == [1,3,2,1,2,3,2]
        buffer = zeros(7)
        @test @inferred(classify!(buffer, A, LabelEnc.OneOfK(3); obsdim=:last)) == [1,3,2,1,2,3,2]
        @test buffer == [1,3,2,1,2,3,2]
        At = A'
        @test @inferred(classify(At, LabelEnc.OneOfK, ObsDim.First())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelEnc.OneOfK, obsdim=1)) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelEnc.OneOfK(Val{3}), ObsDim.First())) == [1,3,2,1,2,3,2]
        @test @inferred(classify(At, LabelEnc.OneOfK(Val{3}), obsdim=1)) == [1,3,2,1,2,3,2]
        @test eltype(classify(At, LabelEnc.OneOfK, ObsDim.First())) <: Int
        @test eltype(classify(At, LabelEnc.OneOfK, obsdim=1)) <: Int
        @test eltype(classify(At, LabelEnc.OneOfK(Val{3}))) <: Int
        @test eltype(classify(At, LabelEnc.OneOfK(Val{3}), ObsDim.First())) <: Int
        @test eltype(classify(At, LabelEnc.OneOfK(Val{3}), obsdim=1)) <: Int
        buffer = zeros(7)
        @test @inferred(classify!(buffer, At, LabelEnc.OneOfK(3), ObsDim.First())) == [1,3,2,1,2,3,2]
        @test buffer == [1,3,2,1,2,3,2]
        buffer = zeros(7)
        @test @inferred(classify!(buffer, At, LabelEnc.OneOfK, obsdim=1)) == [1,3,2,1,2,3,2]
        @test buffer == [1,3,2,1,2,3,2]
    end
end
