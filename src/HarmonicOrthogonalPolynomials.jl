module HarmonicOrthogonalPolynomials
using FastTransforms, LinearAlgebra, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
        BlockArrays, BlockBandedMatrices, InfiniteArrays, StaticArrays, QuasiArrays, Base, SpecialFunctions
import Base: OneTo, axes, getindex, convert, to_indices, _maybetail, tail, eltype, *, ==, ^, copy, -, abs
import BlockArrays: block, blockindex, unblock, BlockSlice
import DomainSets: indomain, Sphere
import LinearAlgebra: norm, factorize
import QuasiArrays: to_quasi_index, SubQuasiArray, *
import ContinuumArrays: TransformFactorization, @simplify, ProjectionFactorization
import ClassicalOrthogonalPolynomials: checkpoints, _sum, cardinality
import BlockBandedMatrices: BlockRange1, _BandedBlockBandedMatrix
import FastTransforms: Plan, interlace
import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle

export SphericalHarmonic, UnitSphere, SphericalCoordinate, RadialCoordinate, Block, associatedlegendre, RealSphericalHarmonic, sphericalharmonicy, Laplacian, AbsLaplacianPower, abs, -, ^, AngularMomentum
cardinality(::Sphere) = ℵ₁

include("multivariateops.jl")
include("spheretrav.jl")
include("coordinates.jl")


checkpoints(::UnitSphere{T}) where T = [SphericalCoordinate{T}(0.1,0.2), SphericalCoordinate{T}(0.3,0.4)]

abstract type AbstractSphericalHarmonic{T} <: MultivariateOrthogonalPolynomial{3,T} end
struct RealSphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
struct SphericalHarmonic{T} <: AbstractSphericalHarmonic{T} end
SphericalHarmonic() = SphericalHarmonic{ComplexF64}()
RealSphericalHarmonic() = RealSphericalHarmonic{Float64}()
copy(a::AbstractSphericalHarmonic) = a

axes(S::AbstractSphericalHarmonic{T}) where T = (Inclusion{SphericalCoordinate{real(T)}}(UnitSphere{real(T)}()), blockedrange(1:2:∞))

associatedlegendre(m) = ((-1)^m*prod(1:2:(2m-1)))*(UltrasphericalWeight((m+1)/2).*Ultraspherical(m+1/2))
lgamma(n) = logabsgamma(n)[1]


function sphericalharmonicy(ℓ, m, θ, φ)
    m̃ = abs(m)
    exp((lgamma(ℓ+m̃+1)+lgamma(ℓ-m̃+1)-2lgamma(ℓ+1))/2)*sqrt((2ℓ+1)/(4π)) * exp(im*m*φ) * sin(θ/2)^m̃ * cos(θ/2)^m̃ * jacobip(ℓ-m̃,m̃,m̃,cos(θ))
end

function getindex(S::SphericalHarmonic{T}, x::SphericalCoordinate, K::BlockIndex{1}) where T
    ℓ = Int(block(K))
    k = blockindex(K)
    m = k-ℓ
    convert(T, sphericalharmonicy(ℓ-1, m, x.θ, x.φ))::T
end

==(::SphericalHarmonic{T},::SphericalHarmonic{T}) where T = true
==(::RealSphericalHarmonic{T},::RealSphericalHarmonic{T}) where T = true

# function getindex(S::RealSphericalHarmonic{T}, x::ZSphericalCoordinate, K::BlockIndex{1}) where T
#     # sorts entries by ...-2,-1,0,1,2... scheme
#     ℓ = Int(block(K))
#     k = blockindex(K)
#     m = k-ℓ
#     m̃ = abs(m)
#     indepm = (-1)^m̃*exp((lgamma(ℓ-m̃)-lgamma(ℓ+m̃))/2)*sqrt((2ℓ-1)/(2π))*associatedlegendre(m̃)[x.z,ℓ-m̃]
#     m>0 && return cos(m*x.φ)*indepm
#     m==0 && return cos(m*x.φ)/sqrt(2)*indepm
#     m<0 && return sin(m̃*x.φ)*indepm
# end

function getindex(S::RealSphericalHarmonic{T}, x::SphericalCoordinate, K::BlockIndex{1}) where T
    # starts with m=0, then alternates between sin and cos terms (beginning with sin).
    ℓ = Int(block(K))
    m = blockindex(K)-1
    z = cos(x.θ)
    if iszero(m)
        return sqrt((2ℓ-1)/(4*π))*associatedlegendre(0)[z,ℓ]
    elseif isodd(m)
        m = (m+1)÷2
        return sin(m*x.φ)*(-1)^m*exp((lgamma(ℓ-m)-lgamma(ℓ+m))/2)*sqrt((2ℓ-1)/(2*π))*associatedlegendre(m)[z,ℓ-m]
    else
        m = m÷2
        return cos(m*x.φ)*(-1)^m*exp((lgamma(ℓ-m)-lgamma(ℓ+m))/2)*sqrt((2ℓ-1)/(2*π))*associatedlegendre(m)[z,ℓ-m]
    end
end

getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, K::BlockIndex{1}) = S[SphericalCoordinate(x), K]
getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, K::Block{1}) = S[x, axes(S,2)[K]]
getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, KR::BlockOneTo) = mortar([S[x, K] for K in KR])
getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, k::Int) = S[x, findblockindex(axes(S,2), k)]
getindex(S::AbstractSphericalHarmonic, x::StaticVector{3}, kr::AbstractUnitRange{Int}) = [S[x, k] for k in kr]

# @simplify *(Ac::QuasiAdjoint{<:Any,<:SphericalHarmonic}, B::SphericalHarmonic) =


##
# Expansion
##

const FiniteSphericalHarmonic{T} = SubQuasiArray{T,2,SphericalHarmonic{T},<:Tuple{<:Inclusion,<:BlockSlice{BlockRange1{OneTo{Int}}}}}
const FiniteRealSphericalHarmonic{T} = SubQuasiArray{T,2,RealSphericalHarmonic{T},<:Tuple{<:Inclusion,<:BlockSlice{BlockRange1{OneTo{Int}}}}}
copy(a::FiniteRealSphericalHarmonic) = a
copy(a::FiniteSphericalHarmonic) = a

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
function grid(S::FiniteRealSphericalHarmonic)
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
struct RealSphericalHarmonicTransform{T} <: Plan{T}
    sph2fourier::FastTransforms.FTPlan{T,2,FastTransforms.SPHERE}
    analysis::FastTransforms.FTPlan{T,2,FastTransforms.SPHEREANALYSIS}
end

SphericalHarmonicTransform{T}(N::Int) where T<:Complex = SphericalHarmonicTransform{T}(plan_spinsph2fourier(T, N, 0), plan_spinsph_analysis(T, N, 2N-1, 0))
RealSphericalHarmonicTransform{T}(N::Int) where T<:Real = RealSphericalHarmonicTransform{T}(plan_sph2fourier(T, N), plan_sph_analysis(T, N, 2N-1))

*(P::SphericalHarmonicTransform{T}, f::Matrix{T}) where T = SphereTrav(P.sph2fourier \ (P.analysis * f))
*(P::RealSphericalHarmonicTransform{T}, f::Matrix{T}) where T = RealSphereTrav(P.sph2fourier \ (P.analysis * f))

factorize(S::FiniteSphericalHarmonic{T}) where T =
    TransformFactorization(grid(S), SphericalHarmonicTransform{T}(blocksize(S,2)))
factorize(S::FiniteRealSphericalHarmonic{T}) where T =
    TransformFactorization(grid(S), RealSphericalHarmonicTransform{T}(blocksize(S,2)))

function _sum(A::AbstractSphericalHarmonic{T}, dims) where T
    @assert dims == 1
    PseudoBlockArray(Hcat(sqrt(4convert(T, π)), Zeros{T}(1,∞)), (Base.OneTo(1),axes(A,2)))
end

include("laplace.jl")
include("angularmomentum.jl")

end # module
