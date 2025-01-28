"""
    AngularMomentum
    
Applies the partial derivative with respect to the last angular variable in the coordinate system.
For example, in polar coordinates (r, θ) in ℝ² or cylindrical coordinates (r, θ, z) in ℝ³, we apply ∂ / ∂θ = (x ∂ / ∂y - y ∂ / ∂x).
In spherical coordinates (ρ, θ, φ) in ℝ³, we apply ∂ / ∂φ = (x ∂ / ∂y - y ∂ / ∂x).
"""
struct AngularMomentum{T,Ax<:Inclusion,Order} <: AbstractDifferentialQuasiMatrix{T}
    axis::Ax
    order::Order
end

AngularMomentum{T, D}(axis::D, order) where {T,D<:Inclusion} = AngularMomentum{T,D,typeof(order)}(axis, order)
AngularMomentum{T, D}(axis::D) where {T,D<:Inclusion} = AngularMomentum{T,D,Nothing}(axis, nothing)
AngularMomentum{T}(axis::Inclusion, order...) where T = AngularMomentum{T,typeof(axis)}(axis, order...)
AngularMomentum{T}(domain, order...) where T = AngularMomentum{T}(Inclusion(domain), order...)
AngularMomentum(domain, order...) = AngularMomentum(Inclusion(domain), order...)
AngularMomentum(ax::AbstractQuasiVector{T}, order...) where T = AngularMomentum{eltype(eltype(ax))}(ax, order...)
AngularMomentum(L::AbstractQuasiMatrix, order...) = AngularMomentum(axes(L,1), order...)


operatorcall(::AngularMomentum) = angularmomentum
similaroperator(D::AngularMomentum, ax, ord) = AngularMomentum{eltype(D)}(ax, ord)

simplifiable(::typeof(*), A::AngularMomentum, B::AngularMomentum) = Val(true)
*(D1::AngularMomentum, D2::AngularMomentum) = similaroperator(convert(AbstractQuasiMatrix{promote_type(eltype(D1),eltype(D2))}, D1), D1.axis, operatororder(D1)+operatororder(D2))

angularmomentum(A, order...; dims...) = angularmomentum_layout(MemoryLayout(A), A, order...; dims...)

angularmomentum_layout(::AbstractBasisLayout, Vm, order...; dims...) = error("Overload angularmomentum(::$(typeof(Vm)))")
function angularmomentum_layout(::AbstractBasisLayout, a, order::Int; dims...)
    order < 0 && throw(ArgumentError("order must be non-negative"))
    order == 0 && return a
    isone(order) ? angularmomentum(a) : angularmomentum(angularmomentum(a), order-1)
end

angularmomentum_layout(::ExpansionLayout, A, order...; dims...) = angularmomentum_layout(ApplyLayout{typeof(*)}(), A, order...; dims...)


function angularmomentum_layout(::SubBasisLayout, Vm, order...; dims::Integer=1)
    dims == 1 || error("not implemented")
    angularmomentum(parent(Vm), order...)[:,parentindices(Vm)[2]]
end

function angularmomentum_layout(LAY::ApplyLayout{typeof(*)}, V::AbstractQuasiVecOrMat, order...; dims=1)
    a = arguments(LAY, V)
    dims == 1 || throw(ArgumentError("cannot take angularmomentum a vector along dimension $dims"))
    *(angularmomentum(a[1], order...), tail(a)...)
end

function angularmomentum(P::SphericalHarmonic)
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
    dat = BlockedArray(Vcat(
        (-m)', # n, k-1
        (0 .* m)', # n, k
        (m)', # n, k+1
        ), (blockedrange(Fill(3, 1)), axes(n,1)))
    P * _BandedBlockBandedMatrix(dat', axes(k,1), (0,0), (1,1))
end
=#
