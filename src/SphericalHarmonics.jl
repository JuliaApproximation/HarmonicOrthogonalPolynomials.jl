module SphericalHarmonics
using FastTransforms, LinearAlgebra, OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, BlockArrays, InfiniteArrays, StaticArrays, QuasiArrays, Base
import Base: OneTo, axes, getindex, convert, to_indices, _maybetail, tail, eltype
import BlockArrays: block, blockindex, unblock
import DomainSets: indomain
import LinearAlgebra: norm
import QuasiArrays: to_quasi_index

export SphericalHarmonic, UnitSphere, SphericalCoordinate, Block, associatedlegendre


###
# BlockQuasiArray support
###

@inline to_indices(A::AbstractQuasiArray, inds, I::Tuple{Block{1}, Vararg{Any}}) =
    (unblock(A, inds, I), to_indices(A, _maybetail(inds), tail(I))...)


associatedlegendre(m) = ((-1)^m*prod(1:2:(2m-1)))*(UltrasphericalWeight((m+1)/2).*Ultraspherical(m+1/2))

abstract type AbstractSphericalCoordinate{T} <: StaticVector{3,T} end
norm(::AbstractSphericalCoordinate{T}) where T = real(one(T))

struct SphericalCoordinate{T} <: StaticVector{3,T}
    θ::T
    φ::T
end

struct ZSphericalCoordinate{T} <: StaticVector{3,T}
    θ::T
    z::T
    function ZSphericalCoordinate{T}(θ::T, z::T) where T 
        -1 ≤ z ≤ 1 || throw(ArgumentError("z must be between -1 and 1"))
        new{T}(θ, z)
    end
end
ZSphericalCoordinate(θ::T, z::V) where {T,V} = ZSphericalCoordinate{promote_type(T,V)}(θ,z)
ZSphericalCoordinate(S::SphericalCoordinate) = ZSphericalCoordinate(S.θ, cos(S.φ))
ZSphericalCoordinate{T}(S::SphericalCoordinate) where T = ZSphericalCoordinate{T}(S.θ, cos(S.φ))

SphericalCoordinate(S::ZSphericalCoordinate) = SphericalCoordinate(S.θ, acos(S.z))
SphericalCoordinate{T}(S::ZSphericalCoordinate) where T = SphericalCoordinate{T}(S.θ, acos(S.z))


function getindex(S::SphericalCoordinate, k::Int)
    k == 1 && return sin(S.φ) * cos(S.θ)
    k == 2 && return sin(S.φ) * sin(S.θ)
    k == 3 && return cos(S.φ)
    throw(BoundsError(S, k))
end
function getindex(S::ZSphericalCoordinate, k::Int) 
    k == 1 && return sqrt(1-S.z^2) * cos(S.θ)
    k == 2 && return sqrt(1-S.z^2) * sin(S.θ)
    k == 3 && return S.z
    throw(BoundsError(S, k))
end

convert(::Type{SVector{3,T}}, S::SphericalCoordinate) where T = SVector{3,T}(sin(S.φ)*cos(S.θ), sin(S.φ)*sin(S.θ), cos(S.φ))
convert(::Type{SVector{3,T}}, S::ZSphericalCoordinate) where T = SVector{3,T}(sqrt(1-S.z^2)*cos(S.θ), sqrt(1-S.z^2)*sin(S.θ), S.z)
convert(::Type{SVector{3}}, S::SphericalCoordinate) = SVector(sin(S.φ)*cos(S.θ), sin(S.φ)*sin(S.θ), cos(S.φ))
convert(::Type{SVector{3}}, S::ZSphericalCoordinate) = SVector(sqrt(1-S.z^2)*cos(S.θ), sqrt(1-S.z^2)*sin(S.θ), S.z)

convert(::Type{SphericalCoordinate}, S::ZSphericalCoordinate) = SphericalCoordinate(S)
convert(::Type{SphericalCoordinate{T}}, S::ZSphericalCoordinate) where T = SphericalCoordinate{T}(S)
convert(::Type{ZSphericalCoordinate}, S::SphericalCoordinate) = ZSphericalCoordinate(S)
convert(::Type{ZSphericalCoordinate{T}}, S::SphericalCoordinate) where T = ZSphericalCoordinate{T}(S)


struct SphericalHarmonic{T} <: Basis{T} end
SphericalHarmonic() = SphericalHarmonic{ComplexF64}()

axes(S::SphericalHarmonic{T}) where T = (Inclusion{ZSphericalCoordinate{real(T)}}(UnitSphere{real(T)}()), blockedrange(1:2:∞))

function getindex(S::SphericalHarmonic, x::ZSphericalCoordinate, K::BlockIndex{1})
    ℓ = Int(block(K))
    k = blockindex(K)
    m = k-ℓ
    exp(im*m*x.θ) * associatedlegendre(abs(m))[x.z,ℓ]
end

getindex(S::SphericalHarmonic, x::StaticVector{3}, K::BlockIndex{1}) = 
    S[ZSphericalCoordinate(x), K]

getindex(S::SphericalHarmonic, x::StaticVector{3}, k::Int) = S[x, findblockindex(axes(S,2), k)]

end # module
