# Laplacian

struct Laplacian{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Laplacian{T}(axis::Inclusion{<:Any,D}) where {T,D} = Laplacian{T,D}(axis)
# Laplacian{T}(domain) where T = Laplacian{T}(Inclusion(domain))
# Laplacian(axis) = Laplacian{eltype(axis)}(axis)

axes(D::Laplacian) = (D.axis, D.axis)
==(a::Laplacian, b::Laplacian) = a.axis == b.axis
copy(D::Laplacian) = Laplacian(copy(D.axis))

@simplify function *(Δ::Laplacian, P::AbstractSphericalHarmonic)
     # Spherical harmonics are the eigenfunctions of the Laplace operator on the unit sphere
     P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2), 1:2:∞)))
end

# Negative fractional Laplacian (-Δ)^α or equiv. abs(Δ)^α

struct AbsLaplacianPower{T,D,A} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
    α::A
end

AbsLaplacianPower{T}(axis::Inclusion{<:Any,D},α) where {T,D} = AbsLaplacianPower{T,D,typeof(α)}(axis,α)

axes(D:: AbsLaplacianPower) = (D.axis, D.axis)
==(a:: AbsLaplacianPower, b:: AbsLaplacianPower) = a.axis == b.axis && a.α == b.α
copy(D:: AbsLaplacianPower) = AbsLaplacianPower(copy(D.axis), D.α)

abs(Δ::Laplacian) = AbsLaplacianPower(axes(Δ,1),1)
-(Δ::Laplacian) = abs(Δ)

^(D::AbsLaplacianPower, k) = AbsLaplacianPower(D.axis, D.α*k)