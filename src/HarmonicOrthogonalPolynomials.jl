module HarmonicOrthogonalPolynomials
using FastTransforms, LinearAlgebra, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
        BlockArrays, BlockBandedMatrices, InfiniteArrays, StaticArrays, QuasiArrays, Base, SpecialFunctions, FillArrays
import Base: OneTo, oneto, axes, getindex, convert, to_indices, tail, eltype, *, ==, ^, copy, -, abs, resize!, zero
import BlockArrays: block, blockindex, unblock, BlockSlice
import DomainSets: indomain, Sphere
import LinearAlgebra: norm, factorize
import QuasiArrays: to_quasi_index, SubQuasiArray, *, AbstractQuasiVecOrMat, Inclusion
import ContinuumArrays: TransformFactorization, @simplify, ProjectionFactorization, plan_grid_transform, plan_transform, grid, grid_layout, plotgrid_layout,
                        AbstractBasisLayout, MemoryLayout, abslaplacian, laplacian, AbstractDifferentialQuasiMatrix, operatorcall, similaroperator, SubBasisLayout,
                        ApplyLayout, arguments, ExpansionLayout, basis_axes, grammatrix
import ClassicalOrthogonalPolynomials: checkpoints, _sum, cardinality, increasingtruncations
import BlockBandedMatrices: BlockRange1, _BandedBlockBandedMatrix
import FastTransforms: Plan, interlace
import QuasiArrays: LazyQuasiMatrix, LazyQuasiArrayStyle, _getindex
import InfiniteArrays: InfStepRange, RangeCumsum
using FillArrays: SquareEye

export SphericalHarmonic, UnitSphere, SphericalCoordinate, RadialCoordinate, Block, associatedlegendre, RealSphericalHarmonic, sphericalharmonicy, abs, -, ^, AngularMomentum, Laplacian, AbsLaplacian
cardinality(::Sphere) = ℵ₁

include("multivariateops.jl")
include("spheretrav.jl")
include("coordinates.jl")

# roughly try to double the computational time each iteration
increasingtruncations(::BlockedOneTo{Int,<:RangeCumsum{Int,<:AbstractRange}}) = broadcast(n -> Block.(oneto((2^(n ÷ 2)) ÷ 2)), 4:2:∞)

checkpoints(S::UnitSphere) = [SphericalCoordinate{eltype(eltype(S))}(0.1,0.2), SphericalCoordinate{eltype(eltype(S))}(0.3,0.4)]

include("sphericalharmonics.jl")
include("laplace.jl")
include("angularmomentum.jl")

end # module
