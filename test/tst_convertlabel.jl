_dst_eltype(any, default) = default
_dst_eltype(::Type{T}, default) where {T<:Number} = T
_dst_eltype(::Type{Bool}, default) = default

@testset "convertlabelview binary" begin
    @test_throws MethodError convertlabelview(LabelEnc.MarginBased, [1.,0,1], LabelEnc.ZeroOne())
    @test_throws MethodError convertlabelview(LabelEnc.OneOfK(2), [1.,0,1], LabelEnc.ZeroOne())
    @test_throws MethodError convertlabelview(LabelEnc.Indices(3), [1.,0,1], LabelEnc.ZeroOne())
    @test_throws MethodError convertlabelview(LabelEnc.ZeroOne(), [1,2,3], LabelEnc.Indices(Int,3))
    @test_throws MethodError convertlabelview(LabelEnc.FuzzyBinary(), [1.,0,1], LabelEnc.ZeroOne())
    @test_throws MethodError convertlabelview(LabelEnc.MarginBased(), Any[1.,0,1], LabelEnc.FuzzyBinary())
    @test_throws MethodError convertlabelview(LabelEnc.FuzzyBinary(), [1.,0,1], LabelEnc.ZeroOne())
    @test_throws MethodError convertlabelview(LabelEnc.FuzzyBinary(), [1.,0,1])
    for (src_lm, src_x) in (
            (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
            (LabelEnc.ZeroOne(Int32),Int32[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(Int64),Int64[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(Float32),Float32[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),Float64[1,0,1,0,0,1]),
            (LabelEnc.MarginBased(Int32),Int32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(Int64),Int64[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),Float64[1,-1,1,-1,-1,1]),
            (LabelEnc.Indices(Float64,2),Float64[1,2,1,2,2,1]),
            (LabelEnc.OneVsRest(:yes),[:yes,:no,:yes,:maybe,:no,:yes]),
            (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
                (LabelEnc.ZeroOne(Int32),Int32[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(Int64),Int64[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(Float32),Float32[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(),Float64[1,0,1,0,0,1]),
                (LabelEnc.MarginBased(Int32),Int32[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(Int64),Int64[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(),Float64[1,-1,1,-1,-1,1]),
                (LabelEnc.Indices(Float64,2),Float64[1,2,1,2,2,1]),
                (LabelEnc.OneVsRest(:yes),[:yes,:not_yes,:yes,:not_yes,:not_yes,:yes]),
                (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
            )
            c_src_x = copy(src_x)
            res = @inferred(convertlabelview(dst_lm, c_src_x, src_lm))
            @test res == dst_x
            if typeof(src_lm) <: LabelEnc.OneVsRest
                @test typeof(res) <: MappedArrays.ReadonlyMappedArray
            else
                res2 = convertlabelview(dst_lm, c_src_x)
                @test res == res2
                @test typeof(res) <: MappedArrays.MappedArray
                @test c_src_x[1] == poslabel(src_lm)
                res[1] = neglabel(dst_lm)
                @test c_src_x[1] == neglabel(src_lm)
                @test c_src_x[2] == neglabel(src_lm)
                res[2] = poslabel(dst_lm)
                @test c_src_x[2] == poslabel(src_lm)
            end
        end
    end
end

println("<HEARTBEAT>")

@testset "convertlabelview multiclass" begin
    for (src_lm, src_x) in (
            (LabelEnc.Indices(Int32,3),Int32[1,2,3,2,2,1]),
            (LabelEnc.Indices(Float64,3),Float64[1,2,3,2,2,1]),
            (LabelEnc.NativeLabels([:a,:b,:c]),[:a,:b,:c,:b,:b,:a]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.Indices(UInt8,3),UInt8[1,2,3,2,2,1]),
                (LabelEnc.Indices(Float64,3),Float64[1,2,3,2,2,1]),
                (LabelEnc.NativeLabels(["a","b","c"]),["a","b","c","b","b","a"]),
            )
            c_src_x = copy(src_x)
            res = @inferred(convertlabelview(dst_lm, c_src_x, src_lm))
            @test res == dst_x
            res2 = convertlabelview(dst_lm, c_src_x)
            @test res == res2
            @test typeof(res) <: MappedArrays.MappedArray
            @test c_src_x[1] == ind2label(1, src_lm)
            res[1] = ind2label(2, dst_lm)
            @test c_src_x[1] == ind2label(2, src_lm)
            @test c_src_x[3] == ind2label(3, src_lm)
            res[3] = ind2label(1, dst_lm)
            @test c_src_x[3] == ind2label(1, src_lm)
        end
    end
end

println("<HEARTBEAT>")

@testset "convert binary" begin
    @test convertlabel(LabelEnc.MarginBased, UInt8[1,0,1], LabelEnc.ZeroOne()) == [0x1,0xff,0x1]
    for (src_lm, src_x) in (
            (LabelEnc.FuzzyBinary(),Any[true,0,1,-1,false,3]),
            (LabelEnc.FuzzyBinary(),Int32[1,0,1,0,0,1]),
            (LabelEnc.FuzzyBinary(),Float64[1,-1,1,-1,-1,1]),
            (LabelEnc.TrueFalse(),BitArray([true,false,true,false,false,true])),
            (LabelEnc.ZeroOne(),Int32[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),Int64[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),Float32[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),Float64[1,0,1,0,0,1]),
            (LabelEnc.MarginBased(),Int32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),Int64[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),Float32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),Float64[1,-1,1,-1,-1,1]),
            (LabelEnc.Indices(2),Float64[1,2,1,2,2,1]),
            (LabelEnc.OneVsRest(:yes),[:yes,:no,:yes,:maybe,:no,:yes]),
            (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
            ([:a,:b],[:a,:b,:a,:b,:b,:a]),
            (LabelEnc.OneOfK(Bool,2),BitArray(Bool[1 0 1 0 0 1; 0 1 0 1 1 0])),
            (LabelEnc.OneOfK(Int8,2),Int8[1 0 1 0 0 1; 0 1 0 1 1 0]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.TrueFalse,BitArray([true,false,true,false,false,true])),
                (LabelEnc.TrueFalse(),BitArray([true,false,true,false,false,true])),
                (LabelEnc.ZeroOne,(_dst_eltype(eltype(src_x),Float64))[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(UInt8),UInt8[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(Int),Int[1,0,1,0,0,1]),
                (LabelEnc.ZeroOne(),[1.,0,1,0,0,1]),
                (LabelEnc.MarginBased,(_dst_eltype(eltype(src_x),Float64))[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(Int),Int[1,-1,1,-1,-1,1]),
                (LabelEnc.MarginBased(),[1,-1.,1,-1,-1,1]),
                (LabelEnc.Indices,(_dst_eltype(eltype(src_x),Int))[1,2,1,2,2,1]),
                (LabelEnc.Indices{UInt8},UInt8[1,2,1,2,2,1]),
                (LabelEnc.Indices(2),Int[1,2,1,2,2,1]),
                (LabelEnc.Indices(Float32,2),Float32[1,2,1,2,2,1]),
                (LabelEnc.OneVsRest(:yes),[:yes,:not_yes,:yes,:not_yes,:not_yes,:yes]),
                (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
                ([:a,:b],[:a,:b,:a,:b,:b,:a]),
                (LabelEnc.OneOfK,(_dst_eltype(eltype(src_x),Int))[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelEnc.OneOfK{UInt8},UInt8[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelEnc.OneOfK{Float32},Float32[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelEnc.OneOfK(Bool,2),BitArray(Bool[1 0 1 0 0 1; 0 1 0 1 1 0])),
            )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                if !(typeof(dst_lm) <: Vector)
                    @test @inferred(islabelenc(dst_x, dst_lm)) == true
                end
                if !(typeof(src_lm)<:LabelEnc.FuzzyBinary) && !xor((typeof(src_lm) <: LabelEnc.OneVsRest),(typeof(dst_lm) <: LabelEnc.OneVsRest))
                    res = if typeof(dst_lm) <: Type || typeof(dst_lm) <: Array
                        convertlabel(dst_lm, src_x)
                    else
                        @inferred convertlabel(dst_lm, src_x)
                    end
                    @test typeof(res) <: typeof(dst_x)
                    @test res == dst_x
                end
                res = if typeof(src_lm) <: Vector && typeof(dst_lm) <: Type && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
                    # in this situation K can not be inferred
                    # this is because we neither specify in src or dst the number of labels at compile time
                    convertlabel(dst_lm, src_x, src_lm)
                elseif typeof(src_lm) <: Vector && typeof(dst_lm) <: Vector
                    # same here
                    convertlabel(dst_lm, src_x, src_lm)
                else
                    @inferred convertlabel(dst_lm, src_x, src_lm)
                end
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
            end
        end
        println("<HEARTBEAT>")
    end
end

@testset "convert multiclass" begin
    for (src_lm, src_x) in (
            (LabelEnc.Indices(3),[1,2,3,2,3,1]),
            (LabelEnc.Indices(Float64,3),Float64[1,2,3,2,3,1]),
            (LabelEnc.NativeLabels([:a,:b,:c]),[:a,:b,:c,:b,:c,:a]),
            ([:a,:b,:c],[:a,:b,:c,:b,:c,:a]),
            (LabelEnc.OneOfK(Bool,3),BitArray(Bool[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0])),
            (LabelEnc.OneOfK(Int8,3),Int8[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.Indices,(_dst_eltype(eltype(src_x),Int))[1,2,3,2,3,1]),
                (LabelEnc.Indices{UInt8},UInt8[1,2,3,2,3,1]),
                (LabelEnc.Indices(3),Int[1,2,3,2,3,1]),
                (LabelEnc.Indices(Float32,3),Float32[1,2,3,2,3,1]),
                (LabelEnc.NativeLabels([:x,:y,:z]),[:x,:y,:z,:y,:z,:x]),
                ([:x,:y,:z],[:x,:y,:z,:y,:z,:x]),
                (LabelEnc.OneOfK,(_dst_eltype(eltype(src_x),Int))[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0]),
                (LabelEnc.OneOfK{UInt8},UInt8[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0]),
                (LabelEnc.OneOfK{Float32},Float32[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0]),
                (LabelEnc.OneOfK(Bool,3),BitArray(Bool[1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0])),
            )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                if !(typeof(dst_lm) <: Vector)
                    @test @inferred(islabelenc(dst_x, dst_lm)) == true
                end
                res = if typeof(dst_lm) <: Type || typeof(dst_lm) <: Array
                    convertlabel(dst_lm, src_x)
                else
                    @inferred convertlabel(dst_lm, src_x)
                end
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = if typeof(src_lm) <: Vector && typeof(dst_lm) <: Type && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
                    # in this situation K can not be inferred
                    # this is because we neither specify in src or dst the number of labels at compile time
                    convertlabel(dst_lm, src_x, src_lm)
                elseif typeof(src_lm) <: Vector && typeof(dst_lm) <: Vector
                    # same here
                    convertlabel(dst_lm, src_x, src_lm)
                else
                    @inferred convertlabel(dst_lm, src_x, src_lm)
                end
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
            end
        end
    end
end

println("<HEARTBEAT>")

@testset "binary OneOfK with and without ObsDim" begin
    x = [1 0 1 0 0 1; 0 1 0 1 1 0]
    xt = x'
    # From OneOfK
    for (dst_lm, dst_x) in (
            (LabelEnc.TrueFalse,BitArray([true,false,true,false,false,true])),
            (LabelEnc.TrueFalse(),BitArray([true,false,true,false,false,true])),
            (LabelEnc.ZeroOne,(_dst_eltype(eltype(x),Float64))[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(UInt8),UInt8[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(Int),Int[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),[1.,0,1,0,0,1]),
            (LabelEnc.MarginBased,(_dst_eltype(eltype(x),Float64))[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(Int),Int[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),[1,-1.,1,-1,-1,1]),
            (LabelEnc.Indices,(_dst_eltype(eltype(x),Int))[1,2,1,2,2,1]),
            (LabelEnc.Indices{UInt8},UInt8[1,2,1,2,2,1]),
            (LabelEnc.Indices(2),Int[1,2,1,2,2,1]),
            (LabelEnc.Indices(Float32,2),Float32[1,2,1,2,2,1]),
            (LabelEnc.OneVsRest(:yes),[:yes,:not_yes,:yes,:not_yes,:not_yes,:yes]),
            (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
            ([:a,:b],[:a,:b,:a,:b,:b,:a]),
        )
        @testset "$x -> ($dst_lm) $dst_x" begin
            res = if typeof(dst_lm) <: Type && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
                convertlabel(dst_lm, x)
            else
                @inferred convertlabel(dst_lm, x)
            end
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            res = @inferred convertlabel(dst_lm, x, LabelEnc.OneOfK(2))
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            # kw obsdim
            res = convertlabel(dst_lm, x; obsdim=2)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt; obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt, LabelEnc.OneOfK(2); obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
        end
    end
    println("<HEARTBEAT>")
    # To OneOfK
    for (src_lm, src_x) in (
            (LabelEnc.TrueFalse(),BitArray([true,false,true,false,false,true])),
            (LabelEnc.ZeroOne(UInt8),UInt8[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(Int),Int[1,0,1,0,0,1]),
            (LabelEnc.ZeroOne(),[1.,0,1,0,0,1]),
            (LabelEnc.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(Int),Int[1,-1,1,-1,-1,1]),
            (LabelEnc.MarginBased(),[1,-1.,1,-1,-1,1]),
            (LabelEnc.Indices(2),Int[1,2,1,2,2,1]),
            (LabelEnc.Indices(Float32,2),Float32[1,2,1,2,2,1]),
            (LabelEnc.OneVsRest(:yes),[:yes,:not_yes,:yes,:not_yes,:not_yes,:yes]),
            (LabelEnc.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.OneOfK,Array{_dst_eltype(eltype(src_x),Int)}(x)),
                (LabelEnc.OneOfK{Float32},Array{Float32}(x)),
                (LabelEnc.OneOfK(Bool,2),BitArray(x)),
             )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                res = @inferred convertlabel(dst_lm, src_x, src_lm)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x

                # TODO in julia 0.7 we take transposition seriously:
                #   typeof(res) == BitArray{2}
                #   typeof(dst_x') == LinearAlgebra.Adjoint{Bool,BitArray{2}}
                
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x

                # kw obsdim
                res = convertlabel(dst_lm, src_x; obsdim=2)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x, src_lm; obsdim=2)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x; obsdim=1)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x'
                res = convertlabel(dst_lm, src_x, src_lm; obsdim=1)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x'
            end
        end
    end
end

println("<HEARTBEAT>")

@testset "multiclass OneOfK with and without ObsDim" begin
    x = [1 0 0 0 0 1; 0 1 0 1 0 0; 0 0 1 0 1 0]
    xt = x'
    # From OneOfK
    for (dst_lm, dst_x) in (
            (LabelEnc.Indices(3),[1,2,3,2,3,1]),
            (LabelEnc.Indices(Float64,3),Float64[1,2,3,2,3,1]),
            (LabelEnc.NativeLabels([:a,:b,:c]),[:a,:b,:c,:b,:c,:a]),
            ([:a,:b,:c],[:a,:b,:c,:b,:c,:a]),
        )
        @testset "$x -> ($dst_lm) $dst_x" begin
            res = if typeof(dst_lm) <: Type && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
                convertlabel(dst_lm, x)
            else
                @inferred convertlabel(dst_lm, x)
            end
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            res = @inferred convertlabel(dst_lm, x, LabelEnc.OneOfK(3))
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            # kw obsdim
            res = convertlabel(dst_lm, x; obsdim=2)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, x, LabelEnc.OneOfK(3); obsdim=2)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt; obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt, LabelEnc.OneOfK(3); obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
        end
    end
    # To OneOfK
    @testset "implicit NativeLabels" begin
        res = @inferred convertlabel(LabelEnc.OneOfK(Int,3), [:a,:b,:c,:b,:c,:a], [:a,:b,:c]; obsdim=1)
        @test typeof(res) <: typeof(x)
        @test res == x'
        res = convertlabel(LabelEnc.OneOfK, [:a,:b,:c,:b,:c,:a], [:a,:b,:c]; obsdim=1)
        @test typeof(res) <: typeof(x)
        @test res == x'
        res = convertlabel(LabelEnc.OneOfK{Float64}, [:a,:b,:c,:b,:c,:a], [:a,:b,:c]; obsdim=1)
        @test typeof(res) <: Matrix{Float64}
        @test res == x'
    end
    for (src_lm, src_x) in (
            (LabelEnc.Indices(3),[1,2,3,2,3,1]),
            (LabelEnc.Indices(Float64,3),Float64[1,2,3,2,3,1]),
            (LabelEnc.NativeLabels([:a,:b,:c]),[:a,:b,:c,:b,:c,:a]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.OneOfK,Array{(_dst_eltype(eltype(src_x),Int))}(x)),
                (LabelEnc.OneOfK{Float32},Array{Float32}(x)),
                (LabelEnc.OneOfK(Bool,3),BitArray(x)),
             )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                res = @inferred convertlabel(dst_lm, src_x, src_lm)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x

                # kw obsdim
                res = convertlabel(dst_lm, src_x; obsdim=2)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x, src_lm; obsdim=2)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x; obsdim=1)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x'
                res = convertlabel(dst_lm, src_x, src_lm; obsdim=1)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x'
            end
        end
    end
end

@testset "special NativeLabels-Indices examples" begin
    enc = LabelEnc.NativeLabels(["a","b","c","d"])
    @test @inferred(convertlabel(LabelEnc.Indices, "c", enc)) == 3

    csub = SubString("c",1,1)
    @test @inferred(convertlabel(LabelEnc.Indices, csub, enc)) == 3
end
