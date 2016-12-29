## labelmap

function labelmap!{T}(dict::Dict{T,Vector{Int}}, idx::Int, elem)
    labelmap!(dict, idx, T(elem))
end

function labelmap!{T}(dict::Dict{T,Vector{Int}}, idx::Int, elem::T)
    if !haskey(dict, elem)
        push!(dict, elem => [idx])
    else
        push!(dict[elem], idx)
    end
    dict
end

function labelmap!{T,S}(dict::Dict{T,Vector{Int}}, idx::AbstractVector{Int}, itr::S)
    for (i, elem) in zip(idx, itr)
        labelmap!(dict, i, elem)
    end
    dict
end

function labelmap{T}(itr::T)
    dict = Dict{eltype(T),Vector{Int}}()
    for (idx, elem) in enumerate(itr)
        labelmap!(dict, idx, elem)
    end
    dict
end

## labelfreq

function labelfreq!{T}(dict::Dict{T,Int}, elem)
    labelfreq!(dict, T(elem))
end

function labelfreq!{T}(dict::Dict{T,Int}, elem::T)
    cnt = get(dict, elem, 0)
    dict[elem] = cnt + 1
    dict
end

function labelfreq!{T}(dict::Dict{T,Int}, iter::AbstractVector)
    for elem in iter
        labelfreq!(dict, T(elem))
    end
    dict
end

labelfreq{T}(itr::T) = countmap(itr)::Dict{eltype(T),Int}
labelfreq{T}(dict::Dict{T,Vector{Int}}) = Dict(k => length(v) for (k,v) in dict)::Dict{T,Int}

## Errors

labelmap(A::AbstractMatrix)  = throw(ArgumentError("labelmap not supported for matrices"))
labelfreq(A::AbstractMatrix) = throw(ArgumentError("labelfreq not supported for matrices"))

## General Dict

labelenc(x::Dict) = labelenc(label(x))
nlabel(x::Dict) = length(keys(x))
label(x::Dict) = _arrange_label(collect(keys(x)))

