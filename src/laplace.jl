struct Laplacian{T,D} <: LazyQuasiMatrix{T}
    axis::Inclusion{T,D}
end

Laplacian{T}(axis::Inclusion{<:Any,D}) where {T,D} = Laplacian{T,D}(axis)
Laplacian{T}(domain) where T = Laplacian{T}(Inclusion(domain))
Laplacian(axis) = Laplacian{eltype(axis)}(axis)

axes(D::Laplacian) = (D.axis, D.axis)
==(a::Laplacian, b::Laplacian) = a.axis == b.axis

^(D::Laplacian, k::Integer) = ApplyQuasiArray(^, D, k)

@simplify function *(Δ::Laplacian, P::SphericalHarmonic)
     # Spherical harmonics are the eigenfunctions of the Laplace operator on the unit sphere
     P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2), 1:2:∞)))
end
@simplify function *(Δ::Laplacian, P::RealSphericalHarmonic)
    P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2), 1:2:∞)))
end

# this does Δ^k and should probably work automatically but currently doesn't.
function *(D::ApplyQuasiArray{SphericalCoordinate{T},2,typeof(^),Tuple{Laplacian{SphericalCoordinate{T},DomainSets.FixedUnitSphere{StaticArrays.SArray{Tuple{3},T,1,3}}},A}}, P::SphericalHarmonic) where {T,A<:Integer}
    return P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2).^D.args[2], 1:2:∞)))
end
function *(D::ApplyQuasiArray{SphericalCoordinate{T},2,typeof(^),Tuple{Laplacian{SphericalCoordinate{T},DomainSets.FixedUnitSphere{StaticArrays.SArray{Tuple{3},T,1,3}}},A}}, P::RealSphericalHarmonic) where {T,A<:Integer}
    return P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2).^D.args[2], 1:2:∞)))
end
