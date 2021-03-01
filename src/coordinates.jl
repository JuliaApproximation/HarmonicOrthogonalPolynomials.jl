###
# RadialCoordinate
####

"""
   RadialCoordinate(r, θ)

represents the 2-vector [r*cos(θ),r*sin(θ)]
"""
struct RadialCoordinate{T} <: StaticVector{2,T}
    r::T
    θ::T
    RadialCoordinate{T}(r::T, θ::T) where T = new{T}(r, θ)
end

RadialCoordinate{T}(r, θ) where T = RadialCoordinate{T}(convert(T,r), convert(T,θ))
RadialCoordinate(r::T, θ::V) where {T<:Real,V<:Real} = RadialCoordinate{float(promote_type(T,V))}(r, θ)

function RadialCoordinate(xy::StaticVector{2})
    x,y = xy
    RadialCoordinate(norm(xy), atan(y,x))
end

StaticArrays.SVector(rθ::RadialCoordinate) = SVector(rθ.r * cos(rθ.θ), rθ.r * sin(rθ.θ))
getindex(R::RadialCoordinate, k::Int) = SVector(R)[k]

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
    SphericalCoordinate{T}(θ::T, φ::T) where T = new{T}(θ, φ)
end

SphericalCoordinate{T}(θ, φ) where T = SphericalCoordinate{T}(convert(T,θ), convert(T,φ))
SphericalCoordinate(θ::V, φ::T) where {T<:Real,V<:Real} = SphericalCoordinate{float(promote_type(T,V))}(θ, φ)
SphericalCoordinate(S::SphericalCoordinate) = S

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
