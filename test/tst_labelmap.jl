@testset "labelmap" begin
    @test_throws ArgumentError labelmap(rand(2,2))
    @testset "Symbol" begin
        lm = @inferred labelmap([:yes,:no,:no,:yes,:yes])
        @test typeof(labelenc(lm)) <: LabelEnc.NativeLabels{Symbol,2}
        @test typeof(lm) <: Dict{Symbol,Vector{Int}}
        @test @inferred(label(lm)) == [:yes, :no]
        @test typeof(label(lm)) <: Vector{Symbol}
        @test nlabel(lm) === 2
        @test lm[:yes] == [1,4,5]
        @test lm[:no] == [2,3]
        @test @inferred(labelmap!(lm, 6:7, [:yes,:maybe])) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == [1,4,5,6]
        @test lm[:no] == [2,3]
        @test lm[:maybe] == [7]
        @test @inferred(labelmap!(lm, 8, :no)) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == [1,4,5,6]
        @test lm[:no] == [2,3,8]
        @test lm[:maybe] == [7]
        @test @inferred(labelmap!(lm, 9, "no")) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == [1,4,5,6]
        @test lm[:no] == [2,3,8,9]
        @test lm[:maybe] == [7]
    end
    @testset "Float64" begin
        lm = @inferred labelmap([1.,-1,-1,1,1])
        @test typeof(labelenc(lm)) <: LabelEnc.MarginBased{Float64}
        @test typeof(lm) <: Dict{Float64,Vector{Int}}
        @test @inferred(label(lm)) == [1,-1]
        @test typeof(label(lm)) <: Vector{Float64}
        @test nlabel(lm) === 2
        @test lm[1] == [1,4,5]
        @test lm[-1] == [2,3]
        @test @inferred(labelmap!(lm, 6:7, [1,2])) === lm
        @test nlabel(lm) === 3
        @test lm[1] == [1,4,5,6]
        @test lm[-1] == [2,3]
        @test lm[2] == [7]
        @test @inferred(labelmap!(lm, 8, UInt8(2))) === lm
        @test lm[1] == [1,4,5,6]
        @test lm[-1] == [2,3]
        @test lm[2] == [7,8]
        @test_throws MethodError labelmap!(lm, 8, :no)
        @test nlabel(lm) === 3
        @test lm[1] == [1,4,5,6]
        @test lm[-1] == [2,3]
        @test lm[2] == [7,8]
        @test_throws MethodError labelmap!(lm, 9, "no")
        @test nlabel(lm) === 3
        @test lm[1] == [1,4,5,6]
        @test lm[-1] == [2,3]
        @test lm[2] == [7,8]
        @test_throws MethodError labelmap!(lm, 9:10, ["no","yes"])
        @test nlabel(lm) === 3
        @test lm[1] == [1,4,5,6]
        @test lm[-1] == [2,3]
        @test lm[2] == [7,8]
    end
end

@testset "labelfreq" begin
    @test_throws ArgumentError labelfreq(rand(2,2))
    @testset "from labelmap" begin
        lm = @inferred labelfreq(labelmap([:yes,:no,:no,:yes,:yes]))
        @test typeof(labelenc(lm)) <: LabelEnc.NativeLabels{Symbol,2}
        @test typeof(lm) <: Dict{Symbol,Int}
        @test @inferred(label(lm)) == [:yes, :no]
        @test typeof(label(lm)) <: Vector{Symbol}
        @test nlabel(lm) === 2
        @test lm[:yes] == 3
        @test lm[:no] == 2
    end
    @testset "Symbol" begin
        lm = @inferred labelfreq([:yes,:no,:no,:yes,:yes])
        @test typeof(labelenc(lm)) <: LabelEnc.NativeLabels{Symbol,2}
        @test typeof(lm) <: Dict{Symbol,Int}
        @test @inferred(label(lm)) == [:yes, :no]
        @test typeof(label(lm)) <: Vector{Symbol}
        @test nlabel(lm) === 2
        @test lm[:yes] == 3
        @test lm[:no] == 2
        @test @inferred(labelfreq!(lm, [:yes,:maybe])) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == 4
        @test lm[:no] == 2
        @test lm[:maybe] == 1
        @test @inferred(labelfreq!(lm, :no)) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == 4
        @test lm[:no] == 3
        @test lm[:maybe] == 1
        @test @inferred(labelfreq!(lm, "no")) === lm
        @test nlabel(lm) === 3
        @test lm[:yes] == 4
        @test lm[:no] == 4
        @test lm[:maybe] == 1
    end
    @testset "Float64" begin
        lm = @inferred labelfreq([1.,-1,-1,1,1])
        @test typeof(labelenc(lm)) <: LabelEnc.MarginBased{Float64}
        @test typeof(lm) <: Dict{Float64,Int}
        @test @inferred(label(lm)) == [1,-1]
        @test typeof(label(lm)) <: Vector{Float64}
        @test nlabel(lm) === 2
        @test lm[1] == 3
        @test lm[-1] == 2
        @test @inferred(labelfreq!(lm, [1,2])) === lm
        @test nlabel(lm) === 3
        @test lm[1] == 4
        @test lm[-1] == 2
        @test lm[2] == 1
        @test @inferred(labelfreq!(lm, UInt8(2))) === lm
        @test lm[1] == 4
        @test lm[-1] == 2
        @test lm[2] == 2
        @test_throws MethodError labelfreq!(lm, :no)
        @test nlabel(lm) === 3
        @test lm[1] == 4
        @test lm[-1] == 2
        @test lm[2] == 2
        @test_throws MethodError labelfreq!(lm, "no")
        @test nlabel(lm) === 3
        @test lm[1] == 4
        @test lm[-1] == 2
        @test lm[2] == 2
        @test_throws MethodError labelfreq!(lm, ["no","yes"])
        @test nlabel(lm) === 3
        @test lm[1] == 4
        @test lm[-1] == 2
        @test lm[2] == 2
    end
end

