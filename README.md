# HarmonicOrthogonalPolynomials.jl
A Julia package for working with spherical harmonic expansions and
harmonic polynomials in  balls.


[![Build Status](https://github.com/JuliaApproximation/HarmonicOrthogonalPolynomials.jl/workflows/CI/badge.svg)](https://github.com/JuliaApproximation/HarmonicOrthogonalPolynomials.jl/actions)
[![codecov](https://codecov.io/gh/JuliaApproximation/HarmonicOrthogonalPolynomials.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/HarmonicOrthogonalPolynomials.jl)


A [harmonic polynomial](https://en.wikipedia.org/wiki/Harmonic_polynomial) is a multivariate polynomial that solves Laplace's equation. 
[Spherical harmonics](https://en.wikipedia.org/wiki/Spherical_harmonics) are restrictions of harmonic polynomials to the sphere. Importantly they
are orthogonal. This package is primarily an implementation of spherical harmonics (in 2D and 3D) but exploiting their 
polynomial features.

Currently this package focusses on support for
3D spherical harmonics. We use the convention of [FastTransforms](https://mikaelslevinsky.github.io/FastTransforms/transforms.html) for real spherical harmonics:
```julia
julia> θ,φ = 0.1,0.2 # θ is polar, φ is azimuthal (physics convention)

julia> sphericalharmonicy(ℓ, m, θ, φ)
0.07521112971423363 + 0.015246050775019674im
```
But we also allow function approximation, building on top of  [ContinuumArrays.jl](https://github.com/JuliaApproximation/ContinuumArrays.jl) and [ClassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl):
```julia
julia> S = SphericalHarmonic() # A quasi-matrix representation of spherical harmonics
SphericalHarmonic{Complex{Float64}}

julia> S[SphericalCoordinate(θ,φ),Block(ℓ+1)] # evaluate all spherical harmonics with specified ℓ
5-element Array{Complex{Float64},1}:
 0.003545977402630546 - 0.0014992151996309556im
  0.07521112971423363 - 0.015246050775019674im
    0.621352880681805 + 0.0im
  0.07521112971423363 + 0.015246050775019674im
 0.003545977402630546 + 0.0014992151996309556im

julia> 𝐱 = axes(S,1) # represent the unit sphere as a quasi-vector
Inclusion(the 3-dimensional unit sphere)

julia> f = 𝐱 -> ((x,y,z) = 𝐱; exp(x)*cos(y*sin(z))); # function to be approximation

julia> S \ f.(𝐱) # expansion coefficients, adaptively computed
∞-blocked ∞-element BlockedArray{Complex{Float64},1,LazyArrays.CachedArray{Complex{Float64},1,Array{Complex{Float64},1},Zeros{Complex{Float64},1,Tuple{InfiniteArrays.OneToInf{Int64}}}},Tuple{BlockedOneTo{Int,ArrayLayouts.RangeCumsum{Int64,InfiniteArrays.InfStepRange{Int64,Int64}}}}}:
        4.05681442931116 + 0.0im                   
 ──────────────────────────────────────────────────
      1.5777291816142751 + 3.19754060061646e-16im  
  -8.006900295635809e-17 + 0.0im                   
      1.5777291816142751 - 3.539535261006306e-16im 
 ──────────────────────────────────────────────────
      0.3881560551355611 + 5.196884701505137e-17im 
  -7.035627371746071e-17 + 2.5784941810054987e-18im
    -0.30926350498081934 + 0.0im                   
   -6.82462130695514e-17 - 3.515332651034677e-18im 
      0.3881560551355611 - 6.271963079558218e-17im 
 ──────────────────────────────────────────────────
     0.06830566496722756 - 8.852861226980248e-17im 
 -2.3672451919730833e-17 + 2.642173739237023e-18im 
     -0.0514592471634392 - 1.5572791163000952e-17im
  1.1972144648274198e-16 + 0.0im                   
    -0.05145924716343915 + 1.5264133695821818e-17im
                         ⋮

julia> f̃ = S * (S \ f.(𝐱)); # expansion of f in spherical harmonics

julia> f̃[SphericalCoordinate(θ,φ)] # approximates f
1.1026374731849062 + 4.004893695029451e-16im
```