module SphericalHarmonics
using FastTransforms, LinearAlgebra, OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, 
        BlockArrays, BlockBandedMatrices, InfiniteArrays, StaticArrays, QuasiArrays, Base, SpecialFunctions
import Base: OneTo, axes, getindex, convert, to_indices, _maybetail, tail, eltype
import BlockArrays: block, blockindex, unblock, BlockSlice
import DomainSets: indomain
import LinearAlgebra: norm, factorize
import QuasiArrays: to_quasi_index, SubQuasiArray
import ContinuumArrays: TransformFactorization
import BlockBandedMatrices: BlockRange1
import FastTransforms: Plan

export SphericalHarmonic, UnitSphere, SphericalCoordinate, Block, associatedlegendre



###
# SphericalCoordinate
###

abstract type AbstractSphericalCoordinate{T} <: StaticVector{3,T} end
norm(::AbstractSphericalCoordinate{T}) where T = real(one(T))

struct SphericalCoordinate{T} <: StaticVector{3,T}
    θ::T
    φ::T
end

struct ZSphericalCoordinate{T} <: StaticVector{3,T}
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


abstract type AbstractSphericalHarmonic{T} <: Basis{T} end
struct RealSphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
struct SphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
SphericalHarmonic() = SphericalHarmonic{ComplexF64}()

axes(S::AbstractSphericalHarmonic{T}) where T = (Inclusion{ZSphericalCoordinate{real(T)}}(UnitSphere{real(T)}()), blockedrange(1:2:∞))

associatedlegendre(m) = ((-1)^m*prod(1:2:(2m-1)))*(UltrasphericalWeight((m+1)/2).*Ultraspherical(m+1/2))

function getindex(S::SphericalHarmonic, x::ZSphericalCoordinate, K::BlockIndex{1})
    ℓ = Int(block(K))
    k = blockindex(K)
    m = k-ℓ
    m̃ = abs(m)
    s = m < 0 ? (-1)^m : 1
    s*sqrt(exp(logabsgamma(ℓ-m̃)[1]-logabsgamma(ℓ+m̃)[1])*(2ℓ-1)/(4π)) * exp(im*m*x.φ) * associatedlegendre(m̃)[x.z,ℓ-m̃]
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
    sph2fourier::FastTransforms.FTPlan{real(T),2,FastTransforms.SPHERE}
    analysis::FastTransforms.FTPlan{real(T),2,FastTransforms.SPHEREANALYSIS}
end

SphericalHarmonicTransform{T}(N::Int) where T = SphericalHarmonicTransform{T}(plan_sph2fourier(T, N), plan_sph_analysis(T, N, 2N-1))

factorize(S::FiniteSphericalHarmonic{T}) where T =
    TransformFactorization(grid(S), SphericalHarmonicTransform{T}(size(S,2)))

end # module
