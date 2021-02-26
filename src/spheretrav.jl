
###
# SphereTrav
###


"""
    SphereTrav(A::AbstractMatrix)

is an anlogue of `DiagTrav` but for coefficients stored according to 
FastTransforms.jl spherical harmonics layout
"""
struct SphereTrav{T, AA<:AbstractMatrix{T}} <: AbstractBlockVector{T}
    matrix::AA
    function SphereTrav{T, AA}(matrix::AA) where {T,AA<:AbstractMatrix{T}}
        n,m = size(matrix)
        m == 2n-1 || throw(ArgumentError("size must match"))
        new{T,AA}(matrix)
    end
end

SphereTrav{T}(matrix::AbstractMatrix{T}) where T = SphereTrav{T,typeof(matrix)}(matrix)
SphereTrav(matrix::AbstractMatrix{T}) where T = SphereTrav{T}(matrix)

axes(A::SphereTrav) = (blockedrange(range(1; step=2, length=size(A.matrix,1))),)

function getindex(A::SphereTrav, K::Block{1})
    k = Int(K)
    m = size(A.matrix,1)
    st = stride(A.matrix,2)
    # nonnegative terms
    p = A.matrix[range(k; step=2*st-1, length=k)]
    k == 1 && return p
    # negative terms
    n = A.matrix[range(k+st-1; step=2*st-1, length=k-1)]
    [reverse!(n); p] 
end

getindex(A::SphereTrav, k::Int) = A[findblockindex(axes(A,1), k)]

"""
    RealSphereTrav(A::AbstractMatrix)

    takes coefficients as provided by the spherical harmonics layout of FastTransforms.jl and
    makes them accessible sorted such that in each block the m=0 entries are always in first place, 
    followed by alternating sin and cos terms of increasing |m|.
"""
struct RealSphereTrav{T, AA<:AbstractMatrix{T}} <: AbstractBlockVector{T}
    matrix::AA
    function RealSphereTrav{T, AA}(matrix::AA) where {T,AA<:AbstractMatrix{T}}
        n,m = size(matrix)
        m == 2n-1 || throw(ArgumentError("size must match"))
        new{T,AA}(matrix)
    end
end

RealSphereTrav{T}(matrix::AbstractMatrix{T}) where T = RealSphereTrav{T,typeof(matrix)}(matrix)
RealSphereTrav(matrix::AbstractMatrix{T}) where T = RealSphereTrav{T}(matrix)

axes(A::RealSphereTrav) = (blockedrange(range(1; step=2, length=size(A.matrix,1))),)

function getindex(A::RealSphereTrav, K::Block{1})
    k = Int(K)
    m = size(A.matrix,1)
    st = stride(A.matrix,2)
    # nonnegative terms
    p = A.matrix[range(k; step=2*st-1, length=k)]
    k == 1 && return p
    # negative terms
    n = A.matrix[range(k+st-1; step=2*st-1, length=k-1)]
    interlace(p,n)
end

getindex(A::RealSphereTrav, k::Int) = A[findblockindex(axes(A,1), k)]