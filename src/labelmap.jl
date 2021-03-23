## labelmap

function LearnBase.labelmap!(dict::Dict{T,Vector{Int}}, idx::Int, elem) where {T}
    labelmap!(dict, idx, T(elem))
end

function LearnBase.labelmap!(dict::Dict{T,Vector{Int}}, idx::Int, elem::T) where {T}
    if !haskey(dict, elem)
        push!(dict, elem => [idx])
    else
        push!(dict[elem], idx)
    end
    dict
end

function LearnBase.labelmap!(dict::Dict{T,Vector{Int}}, idx::AbstractVector{Int}, itr::S) where {T,S}
    for (i, elem) in zip(idx, itr)
        labelmap!(dict, i, elem)
    end
    dict
end

function LearnBase.labelmap(itr::T) where {T}
    dict = Dict{eltype(T),Vector{Int}}()
    for (idx, elem) in enumerate(itr)
        labelmap!(dict, idx, elem)
    end
    dict
end

## labelfreq

function LearnBase.labelfreq!(dict::Dict{T,Int}, elem) where {T}
    labelfreq!(dict, T(elem))
end

function LearnBase.labelfreq!(dict::Dict{T,Int}, elem::T) where {T}
    cnt = get(dict, elem, 0)
    dict[elem] = cnt + 1
    dict
end

function LearnBase.labelfreq!(dict::Dict{T,Int}, iter::AbstractVector) where {T}
    for elem in iter
        labelfreq!(dict, T(elem))
    end
    dict
end

LearnBase.labelfreq(itr::T) where {T} = countmap(itr, alg = :dict)::Dict{eltype(T),Int}
LearnBase.labelfreq(dict::Dict{T,Vector{Int}})  where {T} = Dict(k => length(v) for (k,v) in dict)::Dict{T,Int}

## Errors

LearnBase.labelmap(::AbstractMatrix)  = throw(ArgumentError("labelmap not supported for matrices"))
LearnBase.labelfreq(::AbstractMatrix) = throw(ArgumentError("labelfreq not supported for matrices"))

## General Dict

LearnBase.labelenc(x::Dict) = labelenc(label(x))
LearnBase.nlabel(x::Dict) = length(keys(x))
LearnBase.label(x::Dict) = _arrange_label(collect(keys(x)))

## Convert label map to label vector

function LearnBase.labelmap2vec(lm::Dict{T, Vector{Int}}) where T
    isempty(lm) && return Vector{T}(undef, 0)
    labelvec = Vector{T}(undef, mapreduce(length, +, values(lm)))
    @inbounds for (k, v) in lm
        labelvec[v] .= k
    end
    return labelvec
end
