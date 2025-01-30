using HarmonicOrthogonalPolynomials, StaticArrays, Test, InfiniteArrays, LinearAlgebra, BlockArrays, ClassicalOrthogonalPolynomials, QuasiArrays
import HarmonicOrthogonalPolynomials: ZSphericalCoordinate, associatedlegendre, grid, SphereTrav, RealSphereTrav, plotgrid

# @testset "associated legendre" begin
#     m = 2
#     θ = 0.1
#     x = cos(θ)
#     @test associatedlegendre(m)[x,1:10] ≈ -(2:11) .* (sin(θ/2)^m * cos(θ/2)^m * Jacobi(m,m)[x,1:10])
# end

@testset "SphereTrav" begin
    A = SphereTrav([1 2 3; 4 0 0])
    @test A == [1, 2, 4, 3]
    @test A[Block(2)] == [2,4,3]
    B = SphereTrav([1 2 3 4 5; 6 7 8 0 0; 9 0 0 0 0 ])
    @test B == [1, 2, 6, 3, 4, 7, 9, 8, 5]
end

@testset "RadialCoordinate" begin
    rθ = RadialCoordinate(0.1,0.2)
    @test rθ ≈ SVector(rθ) ≈ [0.1cos(0.2),0.1sin(0.2)]
    @test RadialCoordinate(1,0.2) isa RadialCoordinate{Float64}
    @test RadialCoordinate(1,1) isa RadialCoordinate{Float64}
    @test_throws BoundsError rθ[3]
end

@testset "SphericalCoordinate" begin
    θφ = SphericalCoordinate(0.2,0.1)
    @test θφ ≈ ZSphericalCoordinate(0.1,cos(0.2))
    @test θφ == SVector(θφ)
    @test SphericalCoordinate(1,1) isa SphericalCoordinate{Float64}
    @test_throws BoundsError θφ[4]

    φz = ZSphericalCoordinate(0.1,cos(0.2))
    @test φz == SVector(φz)

    @test norm(θφ) === norm(φz) === 1.0
    @test θφ in UnitSphere()
    @test ZSphericalCoordinate(0.1,cos(0.2)) in UnitSphere()

    @test convert(SVector{3,Float64}, θφ) ≈ convert(SVector{3}, θφ) ≈
            convert(SVector{3,Float64}, φz) ≈ convert(SVector{3}, φz) ≈ θφ

    @test ZSphericalCoordinate(θφ) ≡ convert(ZSphericalCoordinate, θφ) ≡ φz
    @test SphericalCoordinate(φz) ≡ convert(SphericalCoordinate, φz) ≡ θφ
end

@testset "Evaluation" begin
    S = SphericalHarmonic()
    @test copy(S) == S
    @test eltype(axes(S,1)) == SphericalCoordinate{Float64}

    θ,φ = 0.1,0.2
    x = SphericalCoordinate(θ,φ)
    @test S[x, Block(1)[1]] == S[x,1] == sqrt(1/(4π))
    @test view(S,x, Block(1)).indices[1] isa SphericalCoordinate
    @test S[x, Block(1)] == [sqrt(1/(4π))]

    @test associatedlegendre(0)[0.1,1:2] ≈ [1.0,0.1]
    @test associatedlegendre(1)[0.1,1:2] ≈ [-0.9949874371066201,-0.29849623113198603]
    @test associatedlegendre(2)[0.1,1] ≈ 2.97

    @test S[x,Block(2)] ≈ 0.5sqrt(3/π)*[1/sqrt(2)*sin(θ)exp(-im*φ),cos(θ),1/sqrt(2)*sin(θ)exp(im*φ)]
    @test S[x,Block(3)] ≈ [0.25sqrt(15/2π)sin(θ)^2*exp(-2im*φ),
                            0.5sqrt(15/2π)sin(θ)cos(θ)exp(-im*φ),
                            0.25sqrt(5/π)*(3cos(θ)^2-1),
                            0.5sqrt(15/2π)sin(θ)cos(θ)exp(im*φ),
                            0.25sqrt(15/2π)sin(θ)^2*exp(2im*φ)]
    @test S[x,Block(4)] ≈ [0.125sqrt(35/π)sin(θ)^3*exp(-3im*φ),
                          0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(-2im*φ),
                          0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(-im*φ),
                          0.25sqrt(7/π)*(5cos(θ)^3-3cos(θ)),
                          0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(im*φ),
                          0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(2im*φ),
                          0.125sqrt(35/π)sin(θ)^3*exp(3im*φ)]

    @test S[x,Block.(1:4)] == [S[x,Block(1)]; S[x,Block(2)]; S[x,Block(3)]; S[x,Block(4)]]
end

@testset "Real Evaluation" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    @test eltype(axes(R,1)) == SphericalCoordinate{Float64}
    θ,φ = 0.1,0.2
    x = SphericalCoordinate(θ,φ)
    @test R[x, Block(1)[1]] ≈ R[x,1] ≈ sqrt(1/(4π))
    @test R[x, Block(2)][1] ≈ S[x, Block(2)][2]
    # Careful here with the (-1) conventions?
    @test R[x, Block(2)][3] ≈ 1/sqrt(2)*(S[x, Block(2)][3]+S[x, Block(2)][1])
    @test R[x, Block(2)][2] ≈ im/sqrt(2)*(S[x, Block(2)][1]-S[x, Block(2)][3])
end

@testset "Expansion" begin
    @testset "grid" begin
        N = 2
        S = SphericalHarmonic()[:,Block.(Base.OneTo(N))]

        @test size(S,2) == 4
        g = grid(S)
        @test eltype(g) == SphericalCoordinate{Float64}
        @test plotgrid(S) == grid(SphericalHarmonic(), Block(4))

        # compare with FastTransforms.jl/examples/sphere.jl
        # The colatitudinal grid (mod $\pi$):
        N = 2
        θ = (0.5:N-0.5)/N
        # The longitudinal grid (mod $\pi$):
        M = 2*N-1
        φ = (0:M-1)*2/M
        X = [sinpi(θ)*cospi(φ) for θ in θ, φ in φ]
        Y = [sinpi(θ)*sinpi(φ) for θ in θ, φ in φ]
        Z = [cospi(θ) for θ in θ, φ in φ]
        @test g ≈ SVector.(X, Y, Z)
    end

    @testset "transform" begin
        N = 2
        S = SphericalHarmonic()[:,Block.(Base.OneTo(N))]
        xyz = axes(S,1)

        P = factorize(S)
        @test eltype(P) == ComplexF64
        c = P \ (xyz -> 1).(xyz)
        @test blocksize(c,1) == blocksize(S,2)
        @test c == S \ (xyz -> 1).(xyz)
        @test (S * c)[SphericalCoordinate(0.1,0.2)] ≈ 1

        f = (x,y,z) -> 1 + x + y + z
        c = S \ splat(f).(xyz)
        u = S * c
        p = SphericalCoordinate(0.1,0.2)
        @test u[p] ≈ 1+sum(p)

        x = grid(SphericalHarmonic(), 5)
        P = plan_transform(SphericalHarmonic(), 5)
        @test P * splat(f).(x) ≈ [c; zeros(5)]
    end

    @testset "adaptive" begin
        S = SphericalHarmonic()
        xyz = axes(S,1)
        u = S * (S \ (xyz -> 1).(xyz))
        @test u[SphericalCoordinate(0.1,0.2)] ≈ 1

        f = c -> exp(-100*c.θ^2)
        u = S * (S \ f.(xyz))
        r = SphericalCoordinate(0.1,0.2)
        @test u[r] ≈ f(r)

        f = c -> ((x,y,z) = c; 1 + x + y + z)
        u = S * (S \ f.(xyz))
        p = SphericalCoordinate(0.1,0.2)
        @test u[p] ≈ 1+sum(p)

        f = c -> ((x,y,z) = c; exp(x)*cos(y*sin(z)))
        u = S * (S \ f.(xyz))
        @test u[p] ≈ f(p)
    end
end

@testset "Real Expansion" begin
    @testset "grid" begin
        N = 2
        S = RealSphericalHarmonic()[:,Block.(Base.OneTo(N))]

        @test size(S,2) == 4
        g = grid(S)
        @test eltype(g) == SphericalCoordinate{Float64}

        # compare with FastTransforms.jl/examples/sphere.jl
        # The colatitudinal grid (mod $\pi$):
        N = 2
        θ = (0.5:N-0.5)/N
        # The longitudinal grid (mod $\pi$):
        M = 2*N-1
        φ = (0:M-1)*2/M
        X = [sinpi(θ)*cospi(φ) for θ in θ, φ in φ]
        Y = [sinpi(θ)*sinpi(φ) for θ in θ, φ in φ]
        Z = [cospi(θ) for θ in θ, φ in φ]
        @test g ≈ SVector.(X, Y, Z)
    end

    @testset "transform" begin
        N = 2
        S = RealSphericalHarmonic()[:,Block.(Base.OneTo(N))]
        xyz = axes(S,1)

        P = factorize(S)
        @test eltype(P) == Float64
        c = P \ (xyz -> 1).(xyz)
        @test blocksize(c,1) == blocksize(S,2)
        @test c == S \ (xyz -> 1).(xyz)
        @test (S * c)[SphericalCoordinate(0.1,0.2)] ≈ 1

        f = c -> ((x,y,z) = c; 1 + x + y + z)
        u = S * (S \ f.(xyz))
        p = SphericalCoordinate(0.1,0.2)
        @test u[p] ≈ 1+sum(p)
    end

    @testset "adaptive" begin
        S = RealSphericalHarmonic()
        xyz = axes(S,1)
        u = S * (S \ (xyz -> 1).(xyz))
        @test u[SphericalCoordinate(0.1,0.2)] ≈ 1

        f = c -> exp(-100*c.θ^2)
        u = S * (S \ f.(xyz))
        r = SphericalCoordinate(0.1,0.2)
        @test u[r] ≈ f(r)

        f = c -> ((x,y,z) = c; 1 + x + y + z)
        u = S * (S \ f.(xyz))
        p = SphericalCoordinate(0.1,0.2)
        @test u[p] ≈ 1+sum(p)

        f = c -> ((x,y,z) = c; exp(x)*cos(y*sin(z)))
        u = S * (S \ f.(xyz))
        @test u[p] ≈ f(p)
    end
end

@testset "Laplacian basics" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    Sxyz = axes(S,1)
    Rxyz = axes(R,1)
    SΔ = Laplacian(Sxyz)
    RΔ = Laplacian(Rxyz)
    @test SΔ isa Laplacian
    @test RΔ isa Laplacian
    @test SΔ*S isa ApplyQuasiArray
    @test *(RΔ,R) isa ApplyQuasiArray
    @test copy(SΔ) == SΔ == RΔ == copy(RΔ)
    @test axes(SΔ) == axes(RΔ) == (axes(S,1),axes(S,1)) == (axes(R,1),axes(R,1))
    @test axes(SΔ) isa Tuple{Inclusion{SphericalCoordinate{Float64}},Inclusion{SphericalCoordinate{Float64}}}
    @test axes(RΔ) isa Tuple{Inclusion{SphericalCoordinate{Float64}},Inclusion{SphericalCoordinate{Float64}}}
    @test Laplacian{eltype(axes(S,1))}(axes(S,1)) == SΔ
end

@testset "test copy() for SphericalHarmonics" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    @test copy(S) == S
    @test copy(R) == R
    S = SphericalHarmonic()[:,Block.(Base.OneTo(10))]
    R = RealSphericalHarmonic()[:,Block.(Base.OneTo(10))]
    @test copy(S) == S
    @test copy(R) == R
end

@testset "Eigenvalues of spherical Laplacian" begin
    S = SphericalHarmonic()
    xyz = axes(S,1)
    Δ = Laplacian(xyz)
    @test Δ isa Laplacian
    # define some explicit spherical harmonics
    Y_20 = c -> 1/4*sqrt(5/π)*(-1+3*cos(c.θ)^2)
    Y_3m3 = c -> 1/8*exp(-3*im*c.φ)*sqrt(35/π)*sin(c.θ)^3
    Y_41 = c -> 3/8*exp(im*c.φ)*sqrt(5/π)*cos(c.θ)*(-3+7*cos(c.θ)^2)*sin(c.θ) # note phase difference in definitions
    # check that the above correctly represents the respective spherical harmonics
    cfsY20 = S \ Y_20.(xyz)
    @test cfsY20[Block(3)[3]] ≈ 1
    cfsY3m3 = S \ Y_3m3.(xyz)
    @test cfsY3m3[Block(4)[1]] ≈ 1
    cfsY41 = S \ Y_41.(xyz)
    @test cfsY41[Block(5)[6]] ≈ 1
    # Laplacian evaluation and correct eigenvalues
    @test (Δ*S*cfsY20)[SphericalCoordinate(0.7,0.2)] ≈ -6*Y_20(SphericalCoordinate(0.7,0.2))
    @test (Δ*S*cfsY3m3)[SphericalCoordinate(0.1,0.36)] ≈ -12*Y_3m3(SphericalCoordinate(0.1,0.36))
    @test (Δ*S*cfsY41)[SphericalCoordinate(1/3,6/7)] ≈ -20*Y_41(SphericalCoordinate(1/3,6/7))
end

@testset "Laplacian of expansions in complex spherical harmonics" begin
    S = SphericalHarmonic()
    xyz = axes(S,1)
    Δ = Laplacian(xyz)
    @test Δ isa Laplacian
    # define some functions along with the action of the Laplace operator on the unit sphere
    f1  = c -> cos(c.θ)^2
    Δf1 = c -> -1-3*cos(2*c.θ)
    f2  = c -> sin(c.θ)^2-3*cos(c.θ)
    Δf2 = c -> 1+6*cos(c.θ)+3*cos(2*c.θ)
    f3  = c -> 3*cos(c.φ)*sin(c.θ)-cos(c.θ)^2*sin(c.θ)^2
    Δf3 = c -> -1/2-cos(2*c.θ)-5/2*cos(4*c.θ)-6*cos(c.φ)*sin(c.θ)
    f4  = c -> cos(c.θ)^3
    Δf4 = c -> -3*(cos(c.θ)+cos(3*c.θ))
    f5  = c -> 3*cos(c.φ)*sin(c.θ)-2*sin(c.θ)^2
    Δf5 = c -> 1-9*cos(c.θ)^2-6*cos(c.φ)*sin(c.θ)+3*sin(c.θ)^2
    # compare with HarmonicOrthogonalPolynomials Laplacian
    @test (Δ*S*(S\f1.(xyz)))[SphericalCoordinate(2.12,1.993)]  ≈ Δf1(SphericalCoordinate(2.12,1.993))
    @test (Δ*S*(S\f2.(xyz)))[SphericalCoordinate(3.108,1.995)] ≈ Δf2(SphericalCoordinate(3.108,1.995))
    @test (Δ*S*(S\f3.(xyz)))[SphericalCoordinate(0.737,0.239)] ≈ Δf3(SphericalCoordinate(0.737,0.239))
    @test (Δ*S*(S\f4.(xyz)))[SphericalCoordinate(0.162,0.162)] ≈ Δf4(SphericalCoordinate(0.162,0.162))
    @test (Δ*S*(S\f5.(xyz)))[SphericalCoordinate(0.1111,0.999)] ≈ Δf5(SphericalCoordinate(0.1111,0.999))
end

@testset "Laplacian of expansions in real spherical harmonics" begin
    R = RealSphericalHarmonic()
    xyz = axes(R,1)
    Δ = Laplacian(xyz)
    @test Δ isa Laplacian
    # define some functions along with the action of the Laplace operator on the unit sphere
    f1  = c -> cos(c.θ)^2
    Δf1 = c -> -1-3*cos(2*c.θ)
    f2  = c -> sin(c.θ)^2-3*cos(c.θ)
    Δf2 = c -> 1+6*cos(c.θ)+3*cos(2*c.θ)
    f3  = c -> 3*cos(c.φ)*sin(c.θ)-cos(c.θ)^2*sin(c.θ)^2
    Δf3 = c -> -1/2-cos(2*c.θ)-5/2*cos(4*c.θ)-6*cos(c.φ)*sin(c.θ)
    f4  = c -> cos(c.θ)^3
    Δf4 = c -> -3*(cos(c.θ)+cos(3*c.θ))
    f5  = c -> 3*cos(c.φ)*sin(c.θ)-2*sin(c.θ)^2
    Δf5 = c -> 1-9*cos(c.θ)^2-6*cos(c.φ)*sin(c.θ)+3*sin(c.θ)^2
    # compare with HarmonicOrthogonalPolynomials Laplacian
    @test (Δ*R*(R\f1.(xyz)))[SphericalCoordinate(2.12,1.993)]  ≈ Δf1(SphericalCoordinate(2.12,1.993))
    @test (Δ*R*(R\f2.(xyz)))[SphericalCoordinate(3.108,1.995)] ≈ Δf2(SphericalCoordinate(3.108,1.995))
    @test (Δ*R*(R\f3.(xyz)))[SphericalCoordinate(0.737,0.239)] ≈ Δf3(SphericalCoordinate(0.737,0.239))
    @test (Δ*R*(R\f4.(xyz)))[SphericalCoordinate(0.162,0.162)] ≈ Δf4(SphericalCoordinate(0.162,0.162))
    @test (Δ*R*(R\f5.(xyz)))[SphericalCoordinate(0.1111,0.999)] ≈ Δf5(SphericalCoordinate(0.1111,0.999))
end

@testset "Laplacian raised to integer power, adaptive" begin
    S = SphericalHarmonic()
    xyz = axes(S,1)
    @test Laplacian(xyz) isa Laplacian
    @test Laplacian(xyz)^2 isa Laplacian
    @test Laplacian(xyz)^3 isa Laplacian
    f1  = c -> cos(c.θ)^2
    Δ_f1 = c -> -1-3*cos(2*c.θ)
    Δ2_f1 = c -> 6+18*cos(2*c.θ)
    Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
    Δ = Laplacian(xyz)
    Δ2 = Laplacian(xyz)^2
    Δ3 = Laplacian(xyz)^3
    t = SphericalCoordinate(0.122,0.993)
    @test (Δ*S*(S\f1.(xyz)))[t] ≈ Δ_f1(t)
    @test (Δ^2*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ2_f1(t)
    @test (Δ^3*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ3_f1(t)
end

@testset "Finite basis Laplacian, complex" begin
    S = SphericalHarmonic()[:,Block.(Base.OneTo(10))]
    xyz = axes(S,1)
    @test Laplacian(xyz) isa Laplacian
    @test Laplacian(xyz)^2 isa Laplacian
    @test Laplacian(xyz)^3 isa Laplacian
    f1  = c -> cos(c.θ)^2
    Δ_f1 = c -> -1-3*cos(2*c.θ)
    Δ2_f1 = c -> 6+18*cos(2*c.θ)
    Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
    Δ = Laplacian(xyz)
    Δ2 = Laplacian(xyz)^2
    Δ3 = Laplacian(xyz)^3
    t = SphericalCoordinate(0.122,0.993)
    @test (Δ*S*(S\f1.(xyz)))[t] ≈ Δ_f1(t)
    @test (Δ^2*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ2_f1(t)
    @test (Δ^3*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ3_f1(t)
end

@testset "Finite basis Laplacian, real" begin
    S = RealSphericalHarmonic()[:,Block.(Base.OneTo(10))]
    xyz = axes(S,1)
    @test Laplacian(xyz) isa Laplacian
    @test Laplacian(xyz)^2 isa Laplacian
    @test Laplacian(xyz)^3 isa Laplacian
    f1  = c -> cos(c.θ)^2
    Δ_f1 = c -> -1-3*cos(2*c.θ)
    Δ2_f1 = c -> 6+18*cos(2*c.θ)
    Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
    Δ = Laplacian(xyz)
    Δ2 = Laplacian(xyz)^2
    Δ3 = Laplacian(xyz)^3
    t = SphericalCoordinate(0.122,0.993)
    @test (Δ*S*(S\f1.(xyz)))[t] ≈ Δ_f1(t)
    @test (Δ^2*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ2_f1(t)
    @test (Δ^3*S*(S\f1.(xyz)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(xyz)))[t] ≈ Δ3_f1(t)
end

@testset "abs(Δ)^α - Basics of absolute Laplacian powers" begin
    # Set 1
    α = 1/3
    S = SphericalHarmonic()
    Sxyz = axes(S,1)
    SΔα = AbsLaplacian(Sxyz,α)
    Δ = Laplacian(Sxyz)
    @test copy(SΔα) == SΔα
    @test SΔα isa AbsLaplacian
    @test SΔα isa QuasiArrays.LazyQuasiMatrix
    @test axes(SΔα) == (axes(S,1),axes(S,1))
    @test abs(Δ) == -Δ == AbsLaplacian(axes(Δ,1),1)
    @test abs(Δ)^α == SΔα
    # Set 2
    α = 7/13
    S = SphericalHarmonic()
    Sxyz = axes(S,1)
    SΔα = AbsLaplacian(Sxyz,α)
    Δ = Laplacian(Sxyz)
    @test copy(SΔα) == SΔα
    @test SΔα isa AbsLaplacian
    @test SΔα isa QuasiArrays.LazyQuasiMatrix
    @test axes(SΔα) == (axes(S,1),axes(S,1))
    @test abs(Δ) == -Δ == AbsLaplacian(axes(Δ,1),1)
    @test abs(Δ)^α == SΔα
end

@testset "sum" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    @test sum(S; dims=1)[:,1:10] ≈ sum(R; dims=1)[:,1:10] ≈ [sqrt(4π) zeros(1,9)]

    x = axes(S,1)
    @test sum(S * (S \ ones(x))) ≈ sum(R * (R \ ones(x))) ≈ 4π
    f = x -> cos(x[1]*sin(x[2]+x[3]))
    @test sum(S * (S \ f.(x))) ≈ sum(R * (R \ f.(x))) ≈ 11.946489824270322609
end

@testset "Angular momentum" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    ∂θ = AngularMomentum(S)
    @test axes(∂θ) == (axes(S, 1), axes(S, 1))
    @test ∂θ == AngularMomentum(R) == AngularMomentum(axes(S, 1).domain)
    @test copy(∂θ) ≡ ∂θ
    A = S \ (∂θ * S)
    A2 = S \ (∂θ^2 * S)
    @test diag(A[1:9, 1:9]) ≈ [0; 0; -im; im; 0; -im; im; -2im; 2im]
    N = 20
    @test isdiag(A[1:N, 1:N])
    @test A[1:N, 1:N]^2 ≈ A2[1:N, 1:N]
end
