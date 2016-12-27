_dst_eltype(any, default) = default
_dst_eltype{T<:Number}(::Type{T}, default) = T
_dst_eltype(::Type{Bool}, default) = default

@testset "convert binary" begin
    @test convertlabel(LabelEnc.MarginBased, UInt8[1,0,1], LabelEnc.ZeroOne()) == [0x1,0xff,0x1]
    for (src_lm, src_x) in (
            (LabelEnc.FuzzyBinary(),Any[true,0,1,-1,false,3]),
            (LabelEnc.FuzzyBinary(),Int32[1,0,1,0,0,1]),
            (LabelEnc.FuzzyBinary(),Float64[1,-1,1,-1,-1,1]),
            (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
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
            (LabelEnc.OneOfK(Bool,2),Bool[1 0 1 0 0 1; 0 1 0 1 1 0]),
            (LabelEnc.OneOfK(Int8,2),Int8[1 0 1 0 0 1; 0 1 0 1 1 0]),
        )
        for (dst_lm, dst_x) in (
                (LabelEnc.TrueFalse,[true,false,true,false,false,true]),
                (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
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
                (LabelEnc.OneOfK(Bool,2),Bool[1 0 1 0 0 1; 0 1 0 1 1 0]),
            )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                if !(typeof(src_lm)<:LabelEnc.FuzzyBinary) && !((typeof(src_lm) <: LabelEnc.OneVsRest)$(typeof(dst_lm) <: LabelEnc.OneVsRest))
                    res = if typeof(dst_lm) <: DataType || typeof(dst_lm) <: Array
                        convertlabel(dst_lm, src_x)
                    else
                        @inferred convertlabel(dst_lm, src_x)
                    end
                    @test typeof(res) <: typeof(dst_x)
                    @test res == dst_x
                end
                res = if typeof(src_lm) <: Vector && typeof(dst_lm) <: DataType && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
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

@testset "OneOfK with(out) ObsDim" begin
    x = [1 0 1 0 0 1; 0 1 0 1 1 0]
    xt = x'
    for (dst_lm, dst_x) in (
            (LabelEnc.TrueFalse,[true,false,true,false,false,true]),
            (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
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
            res = if typeof(dst_lm) <: DataType && (dst_lm <: LabelEnc.Indices || dst_lm <: LabelEnc.OneOfK)
                convertlabel(dst_lm, x)
            else
                @inferred convertlabel(dst_lm, x)
            end
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            res = @inferred convertlabel(dst_lm, x, LabelEnc.OneOfK(2))
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            # positional obsdim
            res = @inferred convertlabel(dst_lm, x, LabelEnc.OneOfK(2), ObsDim.Constant(2))
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = @inferred convertlabel(dst_lm, x, LabelEnc.OneOfK(2), ObsDim.Last())
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = @inferred convertlabel(dst_lm, xt, LabelEnc.OneOfK(2), ObsDim.First())
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x

            # kw obsdim
            res = convertlabel(dst_lm, x, obsdim=2)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, x, LabelEnc.OneOfK(2), obsdim=:last)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt, obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
            res = convertlabel(dst_lm, xt, LabelEnc.OneOfK(2), obsdim=1)
            @test typeof(res) <: typeof(dst_x)
            @test res == dst_x
        end
    end
    for (src_lm, src_x) in (
            (LabelEnc.TrueFalse(),[true,false,true,false,false,true]),
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
                (LabelEnc.OneOfK,Array{(_dst_eltype(eltype(src_x),Int))}(x)),
                (LabelEnc.OneOfK{Float32},Array{Float32}(x)),
                (LabelEnc.OneOfK(Bool,2),Array{Bool}(x)),
             )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                res = @inferred convertlabel(dst_lm, src_x, src_lm)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x

                # positional obsdim
                res = @inferred convertlabel(dst_lm, src_x, src_lm, ObsDim.Constant(2))
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = @inferred convertlabel(dst_lm, src_x, src_lm, ObsDim.Last())
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = @inferred convertlabel(dst_lm, src_x, src_lm, ObsDim.First())
                @test typeof(res) <: typeof(dst_x')
                @test res == dst_x'

                # kw obsdim
                res = convertlabel(dst_lm, src_x, obsdim=:last)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x, src_lm, obsdim=:last)
                @test typeof(res) <: typeof(dst_x)
                @test res == dst_x
                res = convertlabel(dst_lm, src_x, obsdim=1)
                @test typeof(res) <: typeof(dst_x')
                @test res == dst_x'
                res = convertlabel(dst_lm, src_x, src_lm, obsdim=1)
                @test typeof(res) <: typeof(dst_x')
                @test res == dst_x'
            end
        end
    end
end

