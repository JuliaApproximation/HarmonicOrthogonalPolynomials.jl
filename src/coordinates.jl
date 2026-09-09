###
# RadialCoordinate
####

"""
   RadialCoordinate(r, θ)

represents the 2-vector [r*cos(θ),r*sin(θ)]
"""
struct RadialCoordinate{T<:Real} <: StaticVector{2,T}
    r::T
    θ::T
    RadialCoordinate{T}(r::T, θ::T) where T = new{T}(r, θ)
end

RadialCoordinate{T}(r, θ) where T = RadialCoordinate{T}(convert(T,r), convert(T,θ))
RadialCoordinate(r::T, θ::V) where {T<:Real,V<:Real} = RadialCoordinate{float(promote_type(T,V))}(r, θ)

RadialCoordinate(𝐱::RadialCoordinate) = 𝐱
RadialCoordinate{T}(𝐱::RadialCoordinate{T}) where T = 𝐱
RadialCoordinate{T}(𝐱::RadialCoordinate) where T = RadialCoordinate(convert(T, 𝐱.r), convert(T, 𝐱.θ))
function RadialCoordinate{T}(𝐱::StaticVector{2}) where T
    x,y = 𝐱
    RadialCoordinate{T}(norm(𝐱), atan(y,x))
end

RadialCoordinate(𝐱::StaticVector{2,T}) where T = RadialCoordinate{T}(𝐱)

StaticArrays.SVector(𝐱::RadialCoordinate) = SVector(𝐱.r * cos(𝐱.θ), 𝐱.r * sin(𝐱.θ))

convert(::Type{RadialCoordinate}, 𝐱::StaticVector) = RadialCoordinate(𝐱)
convert(::Type{RadialCoordinate{T}}, 𝐱::StaticVector) where T = RadialCoordinate{T}(𝐱)

getindex(R::RadialCoordinate, k::Int) = SVector(R)[k]
norm(𝐱::RadialCoordinate) = 𝐱.r
LinearAlgebra.norm_sqr(𝐱::RadialCoordinate) = 𝐱.r^2

zero(::Type{RadialCoordinate{T}}) where T = RadialCoordinate{T}(0,0)
zero(r::RadialCoordinate) = zero(typeof(r))


###
# SphericalCoordinate
###

abstract type AbstractSphericalCoordinate{T<:Real} <: StaticVector{3,T} end
norm(S::AbstractSphericalCoordinate{T}) where T = real(S.r)
LinearAlgebra.norm_sqr(𝐱::AbstractSphericalCoordinate) = norm(𝐱)^2
Base.in(S::AbstractSphericalCoordinate, ::UnitSphere{T}) where T = isone(norm(S))

"""
   SphericalCoordinate(r, φ, θ)

represents a point in ℝ^3 as a `StaticVector{3}` in
spherical coordinates where the pole is `SphericalCoordinate(r,φ,0) == SVector(0,0,r)`
and `SphericalCoordinate(r,0,π/2) == SVector(r,0,0)`. 
"""
struct SphericalCoordinate{T<:Real} <: AbstractSphericalCoordinate{T}
    r::T
    φ::T
    θ::T
    SphericalCoordinate{T}(r::T, φ::T, θ::T) where T = new{T}(r, φ, θ)
end

SphericalCoordinate{T}(r, φ, θ) where T = SphericalCoordinate{T}(convert(T,r), convert(T,φ), convert(T,θ))
SphericalCoordinate(r::T, φ::T, θ::T) where T = SphericalCoordinate{real(float(T))}(r, φ, θ)
SphericalCoordinate(r, φ, θ) = SphericalCoordinate(promote(r, φ, θ)...)
SphericalCoordinate{T}(φ, θ) where T = SphericalCoordinate(one(T), φ, θ)
SphericalCoordinate(φ, θ) = SphericalCoordinate(1, φ, θ)
SphericalCoordinate(S::SphericalCoordinate) = S

"""
   ZSphericalCoordinate(r, φ, z)

represents a point in ℝ^3 as a `StaticVector{3}` in
where `z` is specified while the angle coordinate is given by spherical coordinates where the pole is `SVector(0,0,1)`.
"""
struct ZSphericalCoordinate{T<:Real} <: AbstractSphericalCoordinate{T}
    r::T
    φ::T
    z::T
    function ZSphericalCoordinate{T}(r::T, φ::T, z::T) where T 
        -r ≤ z ≤ r || throw(ArgumentError("z must be between -r and r"))
        new{T}(r, φ, z)
    end
end
ZSphericalCoordinate(r::T, φ::T, z::T) where T = ZSphericalCoordinate{T}(r, φ, z)
ZSphericalCoordinate(r, φ, z) = ZSphericalCoordinate(promote(r, φ, z)...)
ZSphericalCoordinate{T}(φ, z) where T = ZSphericalCoordinate(one(T), φ, z)
ZSphericalCoordinate(φ, z) = ZSphericalCoordinate(1, φ, z)
ZSphericalCoordinate(S::SphericalCoordinate) = ZSphericalCoordinate(S.r, S.φ, cos(S.θ))
ZSphericalCoordinate{T}(S::SphericalCoordinate) where T = ZSphericalCoordinate{T}(S.r, S.φ, cos(S.θ))

SphericalCoordinate(S::ZSphericalCoordinate) = SphericalCoordinate(S.r, S.φ, acos(S.z/S.r))
SphericalCoordinate{T}(S::ZSphericalCoordinate) where T = SphericalCoordinate{T}(S.r, S.φ, acos(S.z))


ZSphericalCoordinate{T}(r, φ, z) where T = ZSphericalCoordinate{T}(T(r), T(φ), T(z))
function ZSphericalCoordinate{T}(𝐱::StaticVector{3}) where T
    x,y,z = 𝐱
    ZSphericalCoordinate{T}(norm(𝐱), atan(y,x), z)
end

ZSphericalCoordinate{T}(𝐱::AbstractVector) where T = ZSphericalCoordinate{T}(convert(SVector{3,T}, 𝐱))

ZSphericalCoordinate(𝐱::AbstractVector{T}) where T = ZSphericalCoordinate{float(T)}(𝐱)
ZSphericalCoordinate(𝐱::StaticVector{3,T}) where T = ZSphericalCoordinate{float(T)}(𝐱)

SphericalCoordinate(𝐱::AbstractVector) = SphericalCoordinate(ZSphericalCoordinate(𝐱))
SphericalCoordinate(𝐱::StaticVector{3}) = SphericalCoordinate(ZSphericalCoordinate(𝐱))
SphericalCoordinate{T}(𝐱::AbstractVector) where T = SphericalCoordinate(ZSphericalCoordinate{T}(𝐱))
SphericalCoordinate{T}(𝐱::StaticVector{3}) where T = SphericalCoordinate(ZSphericalCoordinate{T}(𝐱))

zero(::Type{SphericalCoordinate{T}}) where T = SphericalCoordinate(zero(T), zero(T), zero(T))
zero(S::Type{ZSphericalCoordinate{T}}) where T = ZSphericalCoordinate(zero(T), zero(T), zero(T))
zero(S::AbstractSphericalCoordinate) = zero(typeof(S))

function getindex(S::SphericalCoordinate, k::Int)
    r,φ,θ = S.r, S.φ, S.θ
    k == 1 && return r * sin(θ) * cos(φ)
    k == 2 && return r * sin(θ) * sin(φ)
    k == 3 && return r * cos(θ)
    throw(BoundsError(S, k))
end
function getindex(S::ZSphericalCoordinate, k::Int) 
    r,φ,z = S.r, S.φ, S.z
    k == 1 && return sqrt(r^2-z^2) * cos(φ)
    k == 2 && return sqrt(r^2-z^2) * sin(φ)
    k == 3 && return z
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

convert(::Type{SphericalCoordinate{T}}, S::StaticVector{3}) where T = SphericalCoordinate{T}(S)
function convert(::Type{ZSphericalCoordinate{T}}, S::StaticVector{3}) where T
    ZSphericalCoordinate{T}(S)
end
convert(::Type{SphericalCoordinate}, S::StaticVector{3}) = SphericalCoordinate(S)
convert(::Type{ZSphericalCoordinate}, S::StaticVector{3}) = ZSphericalCoordinate(S)
