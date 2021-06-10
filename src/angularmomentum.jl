#########
# AngularMomentum
# Applies the partial derivative with respect to the last angular variable in the coordinate system.
# For example, in polar coordinates (r, θ) in ℝ² or cylindrical coordinates (r, θ, z) in ℝ³, we apply ∂ / ∂θ = (x ∂ / ∂y - y ∂ / ∂x).
# In spherical coordinates (ρ, θ, φ) in ℝ³, we apply ∂ / ∂φ = (x ∂ / ∂y - y ∂ / ∂x).
#########

struct AngularMomentum{T,Ax<:Inclusion} <: LazyQuasiMatrix{T}
    axis::Ax
end

AngularMomentum{T}(axis::Inclusion) where T = AngularMomentum{T,typeof(axis)}(axis)
AngularMomentum{T}(domain) where T = AngularMomentum{T}(Inclusion(domain))
AngularMomentum(axis) = AngularMomentum{eltype(eltype(axis))}(axis)

axes(A::AngularMomentum) = (A.axis, A.axis)
==(a::AngularMomentum, b::AngularMomentum) = a.axis == b.axis
copy(A::AngularMomentum) = AngularMomentum(copy(A.axis))

^(A::AngularMomentum, k::Integer) = ApplyQuasiArray(^, A, k)

@simplify function *(A::AngularMomentum, P::SphericalHarmonic)
    # Spherical harmonics are the eigenfunctions of the angular momentum operator on the unit sphere
    T = real(eltype(P))
    k = mortar(Base.OneTo.(1:2:∞))
    m = T.((-1) .^ (k .- 1) .* k .÷ 2)
    P * Diagonal(im .* m)
end

#=
@simplify function *(A::AngularMomentum, P::RealSphericalHarmonic)
    # The angular momentum operator applied to real spherical harmonics negates orders
    T = real(eltype(P))
    n = mortar(Fill.(oneto(∞),1:2:∞))
    k = mortar(Base.OneTo.(0:2:∞))
    m = T.(iseven.(k) .* (k .÷ 2))
    dat = PseudoBlockArray(Vcat(
        (-m)', # n, k-1
        (0 .* m)', # n, k
        (m)', # n, k+1
        ), (blockedrange(Fill(3, 1)), axes(n,1)))
    P * _BandedBlockBandedMatrix(dat', axes(k,1), (0,0), (1,1))
end
=#
