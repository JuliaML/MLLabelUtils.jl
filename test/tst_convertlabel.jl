_dst_eltype(any, default) = default
_dst_eltype{T<:Number}(::Type{T}, default) = T
_dst_eltype(::Type{Bool}, default) = default

@testset "convert binary" begin
    @test convertlabel(LabelModes.MarginBased, UInt8[1,0,1], LabelModes.ZeroOne()) == [0x1,0xff,0x1]
    for (src_lm, src_x) in (
            (LabelModes.FuzzyBinary(),Any[true,0,1,-1,false,3]),
            (LabelModes.FuzzyBinary(),Int32[1,0,1,0,0,1]),
            (LabelModes.FuzzyBinary(),Float64[1,-1,1,-1,-1,1]),
            (LabelModes.TrueFalse(),[true,false,true,false,false,true]),
            (LabelModes.ZeroOne(),Int32[1,0,1,0,0,1]),
            (LabelModes.ZeroOne(),Int64[1,0,1,0,0,1]),
            (LabelModes.ZeroOne(),Float32[1,0,1,0,0,1]),
            (LabelModes.ZeroOne(),Float64[1,0,1,0,0,1]),
            (LabelModes.MarginBased(),Int32[1,-1,1,-1,-1,1]),
            (LabelModes.MarginBased(),Int64[1,-1,1,-1,-1,1]),
            (LabelModes.MarginBased(),Float32[1,-1,1,-1,-1,1]),
            (LabelModes.MarginBased(),Float64[1,-1,1,-1,-1,1]),
            (LabelModes.Indices(2),Float64[1,2,1,2,2,1]),
            (LabelModes.OneVsRest(:yes),[:yes,:no,:yes,:maybe,:no,:yes]),
            (LabelModes.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
            ([:a,:b],[:a,:b,:a,:b,:b,:a]),
            (LabelModes.OneOfK(Bool,2),Bool[1 0 1 0 0 1; 0 1 0 1 1 0]),
            (LabelModes.OneOfK(Int8,2),Int8[1 0 1 0 0 1; 0 1 0 1 1 0]),
        )
        for (dst_lm, dst_x) in (
                (LabelModes.TrueFalse,[true,false,true,false,false,true]),
                (LabelModes.TrueFalse(),[true,false,true,false,false,true]),
                (LabelModes.ZeroOne,(_dst_eltype(eltype(src_x),Float64))[1,0,1,0,0,1]),
                (LabelModes.ZeroOne(UInt8),UInt8[1,0,1,0,0,1]),
                (LabelModes.ZeroOne(Int),Int[1,0,1,0,0,1]),
                (LabelModes.ZeroOne(),[1.,0,1,0,0,1]),
                (LabelModes.MarginBased,(_dst_eltype(eltype(src_x),Float64))[1,-1,1,-1,-1,1]),
                (LabelModes.MarginBased(Float32),Float32[1,-1,1,-1,-1,1]),
                (LabelModes.MarginBased(Int),Int[1,-1,1,-1,-1,1]),
                (LabelModes.MarginBased(),[1,-1.,1,-1,-1,1]),
                (LabelModes.Indices,(_dst_eltype(eltype(src_x),Int))[1,2,1,2,2,1]),
                (LabelModes.Indices{UInt8},UInt8[1,2,1,2,2,1]),
                (LabelModes.Indices(2),Int[1,2,1,2,2,1]),
                (LabelModes.Indices(Float32,2),Float32[1,2,1,2,2,1]),
                (LabelModes.OneVsRest(:yes),[:yes,:not_yes,:yes,:not_yes,:not_yes,:yes]),
                (LabelModes.NativeLabels([:a,:b]),[:a,:b,:a,:b,:b,:a]),
                ([:a,:b],[:a,:b,:a,:b,:b,:a]),
                (LabelModes.OneOfK,(_dst_eltype(eltype(src_x),Int))[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelModes.OneOfK{UInt8},UInt8[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelModes.OneOfK{Float32},Float32[1 0 1 0 0 1; 0 1 0 1 1 0]),
                (LabelModes.OneOfK(Bool,2),Bool[1 0 1 0 0 1; 0 1 0 1 1 0]),
            )
            @testset "($src_lm) $src_x -> ($dst_lm) $dst_x" begin
                if !(typeof(src_lm)<:LabelModes.FuzzyBinary) && !((typeof(src_lm) <: LabelModes.OneVsRest)$(typeof(dst_lm) <: LabelModes.OneVsRest))
                    res = convertlabel(dst_lm, src_x)
                    @test typeof(res) <: typeof(dst_x)
                    @test res == dst_x
                end
                res = if typeof(src_lm) <: Vector && typeof(dst_lm) <: DataType && (dst_lm <: LabelModes.Indices || dst_lm <: LabelModes.OneOfK)
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
