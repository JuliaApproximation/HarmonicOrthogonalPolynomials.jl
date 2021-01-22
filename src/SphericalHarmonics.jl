module SphericalHarmonics
using FastTransforms, LinearAlgebra, OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, 
        BlockArrays, BlockBandedMatrices, InfiniteArrays, StaticArrays, QuasiArrays, Base, SpecialFunctions
import Base: OneTo, axes, getindex, convert, to_indices, _maybetail, tail, eltype, *, ==, ^, copy
import BlockArrays: block, blockindex, unblock, BlockSlice
import DomainSets: indomain
import LinearAlgebra: norm, factorize
import QuasiArrays: to_quasi_index, SubQuasiArray
import ContinuumArrays: TransformFactorization
import OrthogonalPolynomialsQuasi: checkpoints
import BlockBandedMatrices: BlockRange1
import FastTransforms: Plan
import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle

export SphericalHarmonic, UnitSphere, SphericalCoordinate, Block, associatedlegendre, RealSphericalHarmonic


include("multivariateops.jl")


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

###
# SphericalCoordinate
###

abstract type AbstractSphericalCoordinate{T} <: StaticVector{3,T} end
norm(::AbstractSphericalCoordinate{T}) where T = real(one(T))
Base.in(::AbstractSphericalCoordinate, ::UnitSphere{T}) where T = true
"""
   SphericalCoordinate(θ, φ)

represents a point in the unit sphere as a `StaticVector{3}` in
spherical coordinates where the pole is `SphericalCoordinate(0,φ) == SVector(0,0,1)`
and `SphericalCoordinate(π/2,0) == SVector(1,0,0)`. 
"""
struct SphericalCoordinate{T} <: AbstractSphericalCoordinate{T}
    θ::T
    φ::T
end

SphericalCoordinate(θ, φ) = SphericalCoordinate(promote(θ, φ)...)

"""
   ZSphericalCoordinate(φ, z)

represents a point in the unit sphere as a `StaticVector{3}` in
where `z` is specified while the angle coordinate is given by spherical coordinates where the pole is `SVector(0,0,1)`.
"""
struct ZSphericalCoordinate{T} <: AbstractSphericalCoordinate{T}
    φ::T
    z::T
    function ZSphericalCoordinate{T}(φ::T, z::T) where T 
        -1 ≤ z ≤ 1 || throw(ArgumentError("z must be between -1 and 1"))
        new{T}(φ, z)
    end
end
ZSphericalCoordinate(φ::T, z::V) where {T,V} = ZSphericalCoordinate{promote_type(T,V)}(φ,z)
ZSphericalCoordinate(S::SphericalCoordinate) = ZSphericalCoordinate(S.φ, cos(S.θ))
ZSphericalCoordinate{T}(S::SphericalCoordinate) where T = ZSphericalCoordinate{T}(S.φ, cos(S.θ))

SphericalCoordinate(S::ZSphericalCoordinate) = SphericalCoordinate(acos(S.z), S.φ)
SphericalCoordinate{T}(S::ZSphericalCoordinate) where T = SphericalCoordinate{T}(acos(S.z), S.φ)


function getindex(S::SphericalCoordinate, k::Int)
    k == 1 && return sin(S.θ) * cos(S.φ)
    k == 2 && return sin(S.θ) * sin(S.φ)
    k == 3 && return cos(S.θ)
    throw(BoundsError(S, k))
end
function getindex(S::ZSphericalCoordinate, k::Int) 
    k == 1 && return sqrt(1-S.z^2) * cos(S.φ)
    k == 2 && return sqrt(1-S.z^2) * sin(S.φ)
    k == 3 && return S.z
    throw(BoundsError(S, k))
end

convert(::Type{SVector{3,T}}, S::SphericalCoordinate) where T = SVector{3,T}(sin(S.θ)*cos(S.φ), sin(S.θ)*sin(S.φ), cos(S.θ))
convert(::Type{SVector{3,T}}, S::ZSphericalCoordinate) where T = SVector{3,T}(sqrt(1-S.z^2)*cos(S.φ), sqrt(1-S.z^2)*sin(S.φ), S.z)
convert(::Type{SVector{3}}, S::SphericalCoordinate) = SVector(sin(S.θ)*cos(S.φ), sin(S.θ)*sin(S.φ), cos(S.θ))
convert(::Type{SVector{3}}, S::ZSphericalCoordinate) = SVector(sqrt(1-S.z^2)*cos(S.φ), sqrt(1-S.z^2)*sin(S.φ), S.z)

convert(::Type{SphericalCoordinate}, S::ZSphericalCoordinate) = SphericalCoordinate(S)
convert(::Type{SphericalCoordinate{T}}, S::ZSphericalCoordinate) where T = SphericalCoordinate{T}(S)
convert(::Type{ZSphericalCoordinate}, S::SphericalCoordinate) = ZSphericalCoordinate(S)
convert(::Type{ZSphericalCoordinate{T}}, S::SphericalCoordinate) where T = ZSphericalCoordinate{T}(S)


checkpoints(::UnitSphere{T}) where T = [SphericalCoordinate{T}(0.1,0.2), SphericalCoordinate{T}(0.3,0.4)]

abstract type AbstractSphericalHarmonic{T} <: MultivariateOrthogonalPolynomial{3,T} end
struct RealSphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
struct SphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
SphericalHarmonic() = SphericalHarmonic{ComplexF64}()
RealSphericalHarmonic() = RealSphericalHarmonic{Float64}()

axes(S::AbstractSphericalHarmonic{T}) where T = (Inclusion{SphericalCoordinate{real(T)}}(UnitSphere{real(T)}()), blockedrange(1:2:∞))

associatedlegendre(m) = ((-1)^m*prod(1:2:(2m-1)))*(UltrasphericalWeight((m+1)/2).*Ultraspherical(m+1/2))
lgamma(n) = logabsgamma(n)[1]


function getindex(S::SphericalHarmonic{T}, x::ZSphericalCoordinate, K::BlockIndex{1}) where T
    ℓ = Int(block(K))
    k = blockindex(K)
    m = k-ℓ
    m̃ = abs(m)
    θ = acos(x.z)
    exp((lgamma(ℓ+m̃)+lgamma(ℓ-m̃)-2lgamma(ℓ))/2)*sqrt((2ℓ-1)/(4π)) * exp(im*m*x.φ) * sin(θ/2)^m̃ * cos(θ/2)^m̃ * Jacobi{real(T)}(m̃,m̃)[x.z, ℓ-m̃]
end

function getindex(S::RealSphericalHarmonic{T}, x::ZSphericalCoordinate, K::BlockIndex{1}) where T
    ℓ = Int(block(K))
    k = blockindex(K)
    m = k-ℓ
    m̃ = abs(m)
    θ = acos(x.z)
    indepm = (-1)^m̃*exp((lgamma(ℓ-m̃)-lgamma(ℓ+m̃))/2)*sqrt((2ℓ-1)/(2π))*associatedlegendre(m̃)[x.z,ℓ-m̃]
    m>0 && return cos(m*x.φ)*indepm
    m==0 && return cos(m*x.φ)/sqrt(2)*indepm
    m<0 && return sin(m̃*x.φ)*indepm
end

getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, K::BlockIndex{1}) = 
    S[ZSphericalCoordinate(x), K]

getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, k::Int) = S[x, findblockindex(axes(S,2), k)]

# @simplify *(Ac::QuasiAdjoint{<:Any,<:SphericalHarmonic}, B::SphericalHarmonic) = 


##
# Expansion
##

const FiniteSphericalHarmonic{T} = SubQuasiArray{T,2,SphericalHarmonic{T},<:Tuple{<:Inclusion,<:BlockSlice{BlockRange1{OneTo{Int}}}}}

function grid(S::FiniteSphericalHarmonic)
    T = real(eltype(S))
    N = blocksize(S,2)
    # The colatitudinal grid (mod $\pi$):
    θ = ((1:N) .- one(T)/2)/N
    # The longitudinal grid (mod $\pi$):
    M = 2*N-1
    φ = (0:M-1)*2/convert(T, M)
    SphericalCoordinate.(π*θ, π*φ')
end



struct SphericalHarmonicTransform{T} <: Plan{T}
    sph2fourier::FastTransforms.FTPlan{T,2,FastTransforms.SPINSPHERE}
    analysis::FastTransforms.FTPlan{T,2,FastTransforms.SPINSPHEREANALYSIS}
end

SphericalHarmonicTransform{T}(N::Int) where T<:Complex = SphericalHarmonicTransform{T}(plan_spinsph2fourier(T, N, 0), plan_spinsph_analysis(T, N, 2N-1, 0))

SphericalHarmonicTransform{T}(N::Int) where T<:Real = SphericalHarmonicTransform{T}(plan_sph2fourier(T, N), plan_sph_analysis(T, N, 2N-1))

*(P::SphericalHarmonicTransform{T}, f::Matrix{T}) where T = SphereTrav(P.sph2fourier \ (P.analysis * f))

factorize(S::FiniteSphericalHarmonic{T}) where T =
    TransformFactorization(grid(S), SphericalHarmonicTransform{T}(blocksize(S,2)))

end # module