struct Laplacian{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Laplacian{T}(axis::Inclusion{<:Any,D}) where {T,D} = Laplacian{T,D}(axis)
Laplacian{T}(domain) where T = Laplacian{T}(Inclusion(domain))
Laplacian(axis) = Laplacian{eltype(axis)}(axis)

axes(D::Laplacian) = (D.axis, D.axis)
==(a::Laplacian, b::Laplacian) = a.axis == b.axis

^(D::Laplacian, k::Integer) = ApplyQuasiMatrix(^, D, k)

@simplify function *(Δ::Laplacian, P::AbstractSphericalHarmonic)
     # Spherical harmonics are the eigenfunctions of the Laplace operator on the unit sphere
     P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2), 1:2:∞)))
end