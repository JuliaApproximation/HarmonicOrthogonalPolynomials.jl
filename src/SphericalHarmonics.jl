module SphericalHarmonics
using FastTransforms, LinearAlgebra, OrthogonalPolynomialsQuasi, ContinuumArrays, DomainSets, BlockArrays, InfiniteArrays, StaticArrays, Base
import Base: OneTo


export SphericalHarmonic, UnitSphere, SphericalCoordinate

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
ZSphericalCoordinate(S::SphericalCoordinate) = ZSphericalCoordinate(S.θ, cos(S.φ))

convert(::Type{SVector{3,T}}, S::SphericalCoordinate) where T = convert(SVector{3,T}, ZSphericalCoordinate(S))
convert(::Type{SVector{3,T}}, S::SphericalCoordinate{T}) where T = SVector(S.z*cos(S.θ), S.z*sin(S.θ), S.z)


struct SphericalHarmonic{T} <: Basis{T} end
SphericalHarmonic() = SphericalHarmonic{ComplexF64}()

axes(S::SphericalHarmonic{T}) where T = (UnitSphere{real(T)}(), blockedrange(1:2:∞))

end # module
