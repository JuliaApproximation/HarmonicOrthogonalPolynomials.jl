
#########
# PartialDerivative{k}
# takes a partial derivative in the k-th index.
#########


struct PartialDerivative{k,T,Ax<:Inclusion} <: LazyQuasiMatrix{T}
    axis::Ax
end

PartialDerivative{k,T}(axis::Inclusion) where {k,T} = PartialDerivative{k,T,typeof(axis)}(axis)
PartialDerivative{k,T}(domain) where {k,T} = PartialDerivative{k,T}(Inclusion(domain))
PartialDerivative{k}(axis) where k = PartialDerivative{k,eltype(eltype(axis))}(axis)

axes(D::PartialDerivative) = (D.axis, D.axis)
==(a::PartialDerivative{k}, b::PartialDerivative{k}) where k = a.axis == b.axis
copy(D::PartialDerivative{k}) where k = PartialDerivative{k}(copy(D.axis))

^(D::PartialDerivative, k::Integer) = ApplyQuasiArray(^, D, k)

abstract type MultivariateOrthogonalPolynomial{d,T} <: Basis{T} end
const BivariateOrthogonalPolynomial{T} = MultivariateOrthogonalPolynomial{2,T}

struct MultivariateOPLayout{d} <: AbstractBasisLayout end
MemoryLayout(::Type{<:MultivariateOrthogonalPolynomial{d}}) where d = MultivariateOPLayout{d}()


const BlockOneTo = BlockRange{1,Tuple{OneTo{Int}}}

copy(P::MultivariateOrthogonalPolynomial) = P

getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, JR::BlockOneTo) where D = error("Overload")
getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, J::Block{1}) where D = P[xy, Block.(OneTo(Int(J)))][J]
getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, JR::BlockRange{1}) where D = P[xy, Block.(OneTo(Int(maximum(JR))))][JR]
getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, Jj::BlockIndex{1}) where D = P[xy, block(Jj)][blockindex(Jj)]
getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, j::Integer) where D = P[xy, findblockindex(axes(P,2), j)]
getindex(P::MultivariateOrthogonalPolynomial{D}, xy::StaticVector{D}, jr::AbstractVector{<:Integer}) where D = P[xy, Block.(OneTo(Int(findblock(axes(P,2), maximum(jr)))))][jr]

const FirstInclusion = BroadcastQuasiVector{<:Any, typeof(first), <:Tuple{Inclusion}}
const LastInclusion = BroadcastQuasiVector{<:Any, typeof(last), <:Tuple{Inclusion}}

function Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::FirstInclusion, P::MultivariateOrthogonalPolynomial)
    axes(x,1) == axes(P,1) || throw(DimensionMismatch())
    P*jacobimatrix(Val(1), P)
end

function Base.broadcasted(::LazyQuasiArrayStyle{2}, ::typeof(*), x::LastInclusion, P::MultivariateOrthogonalPolynomial)
    axes(x,1) == axes(P,1) || throw(DimensionMismatch())
    P*jacobimatrix(Val(2), P)
end

"""
   forwardrecurrence!(v, A, B, C, (x,y))

evaluates the bivaraite orthogonal polynomials at points `(x,y)` ,
where `A`, `B`, and `C` are `AbstractVector`s containing the recurrence coefficients
matrices. In particular, note that for any OPs we have

    P[N+1,x] = (A[N]* [x*I; y*I] + B[N]) * P[N,x] - C[N] * P[N-1,x]

where A[N] is (N+1) x 2N, B[N] and C[N] are (N+1) x N.

"""
# function forwardrecurrence!(v::AbstractBlockVector{T}, A::AbstractVector, B::AbstractVector, C::AbstractVector, xy) where T
#     N = blocklength(v)
#     N == 0 && return v
#     length(A)+1 ≥ N && length(B)+1 ≥ N && length(C)+1 ≥ N || throw(ArgumentError("A, B, C must contain at least $(N-1) entries"))
#     v[Block(1)] .= one(T)
#     N = 1 && return v
#     p_1 = view(v,Block(2))
#     p_0 = view(v,Block(1))
#     mul!(p_1, B[1], p_0)
#     xy_muladd!(xy, A[1], p_0, one(T), p_1)

#     @inbounds for n = 2:N-1
#         p_2 = view(v,Block(n+1))
#         mul!(p_2, B[n], p_1)
#         xy_muladd!(xy, A[n], p_1, one(T), p_2)
#         muladd!(-one(T), C[n], p_0, one(T), p_2)
#         p_1,p_0 = p_2,p_1
#     end    
#     v
# end


# forwardrecurrence(N::Block{1}, A::AbstractVector, B::AbstractVector, C::AbstractVector, xy) =
#     forwardrecurrence!(PseudoBlockVector{promote_type(eltype(eltype(A)),eltype(eltype(B)),eltype(eltype(C)),eltype(xy))}(undef, 1:Int(N)), A, B, C, xy)

# use block expansion
ContinuumArrays.transform_ldiv(V::SubQuasiArray{<:Any,2,<:MultivariateOrthogonalPolynomial,<:Tuple{Inclusion,BlockSlice{BlockOneTo}}}, B::AbstractQuasiArray, _) =
    factorize(V) \ B

ContinuumArrays._sub_factorize(::Tuple{Any,Any}, (kr,jr)::Tuple{Any,BlockSlice{BlockRange1{OneTo{Int}}}}, L, dims...; kws...) =
    TransformFactorization(plan_grid_transform(parent(L), PseudoBlockArray{eltype(L)}(undef, (jr.indices, oneto.(dims)...)), 1)...)

# function factorize(V::SubQuasiArray{<:Any,2,<:MultivariateOrthogonalPolynomial,<:Tuple{Inclusion,AbstractVector{Int}}})
#     P = parent(V)
#     _,jr = parentindices(V)
#     J = findblock(axes(P,2),maximum(jr))
#     ProjectionFactorization(factorize(P[:,Block.(OneTo(Int(J)))]), jr)
# end

# Make sure block structure matches. Probably should do this for all block mul
QuasiArrays.mul(A::MultivariateOrthogonalPolynomial, b::AbstractVector) =
    ApplyQuasiArray(*, A, PseudoBlockVector(b, (axes(A,2),)))