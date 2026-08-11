using HarmonicOrthogonalPolynomials, StaticArrays, Test, InfiniteArrays, LinearAlgebra, BlockArrays, ClassicalOrthogonalPolynomials, QuasiArrays, ContinuumArrays, Rotations, WignerD
import HarmonicOrthogonalPolynomials: ZSphericalCoordinate, associatedlegendre, grid, SphereTrav, RealSphereTrav, plotgrid, BivariateOrthogonalPolynomial
using FastTransforms: pochhammer

@testset "associated legendre" begin
    θ = 0.1
    x = cos(θ)
    for m = 0:5
        @test associatedlegendre(m)[x,1:10] ≈ associatedlegendre.(m:m+9, m, x)
    end
    for ℓ = 0:5, m = 0:ℓ
        @test associatedlegendre(ℓ, m, x) ≈ (-2)^m * pochhammer(1/2, m) * sin(θ)^m * ultrasphericalc(ℓ-m, m+1/2, x) ≈
            (-1)^m * (1-x^2)^(m/2) * diff(Legendre(), m)[x,ℓ+1]
        @test associatedlegendre(ℓ, -m, x) ≈ (-1)^m * factorial(ℓ-m)/factorial(ℓ+m) * associatedlegendre(ℓ, m, x)
    end

    for m = 0:5
        @test associatedlegendre(m, -m, x) ≈ sin(θ)^m/(2^m * factorial(m))
    end
end

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
    @test zero(rθ) ≡ zero(typeof(rθ ))
    @test norm(rθ) == rθ.r
    @test LinearAlgebra.norm_sqr(rθ) == rθ.r^2
end
@testset "SphericalCoordinate" begin
    θφ = SphericalCoordinate(0.1,0.2)
    @test θφ ≈ ZSphericalCoordinate(0.1,cos(0.2))
    @test θφ == SVector(θφ)
    @test SphericalCoordinate(1,1) isa SphericalCoordinate{Float64}
    @test_throws BoundsError θφ[4]

    @test zero(θφ) == zero(typeof(θφ)) == SVector{3,Float64}(0,0,0)

    φz = ZSphericalCoordinate(0.1,cos(0.2))
    @test φz == SVector(φz)

    @test norm(θφ) === norm(φz) === 1.0
    @test LinearAlgebra.norm_sqr(θφ) === 1.0
    @test LinearAlgebra.norm_sqr(φz) === 1.0
    @test θφ in UnitSphere()
    @test ZSphericalCoordinate(0.1,cos(0.2)) in UnitSphere()

    @test convert(SVector{3,Float64}, θφ) ≈ convert(SVector{3}, θφ) ≈
            convert(SVector{3,Float64}, φz) ≈ convert(SVector{3}, φz) ≈ θφ

    @test ZSphericalCoordinate(θφ) ≡ convert(ZSphericalCoordinate, θφ) ≡ φz
    @test SphericalCoordinate(φz) ≡ convert(SphericalCoordinate, φz) ≡ θφ

    @test ZSphericalCoordinate{Float64}(0.1, cos(0.2)) ≡ φz
    @test ZSphericalCoordinate{Float64}(1.0, 0.1, cos(0.2)) ≡ φz
    @test SphericalCoordinate{Float64}(φz) ≡ θφ
    @test convert(ZSphericalCoordinate{Float64}, θφ) ≡ φz
    @test convert(SphericalCoordinate{Float64}, φz) ≡ θφ

    # test SphericalCoordinate from ZSphericalCoordinate with non-unit radius
    𝐫 = ZSphericalCoordinate(2.0, 0.1, 2.0*cos(0.2))
    @test SphericalCoordinate(𝐫) ≈ SphericalCoordinate(2.0, 0.1, 0.2)

    @test zero(φz) == zero(ZSphericalCoordinate{Float64}) == SVector{3,Float64}(0,0,0)

    𝐱 = SVector(θφ)
    @test ZSphericalCoordinate(𝐱) ≡ φz
    @test SphericalCoordinate(𝐱) ≡ θφ
    @test convert(ZSphericalCoordinate, 𝐱) ≡ φz
    @test convert(ZSphericalCoordinate{Float64}, 𝐱) ≡ φz
    @test convert(SphericalCoordinate, 𝐱) ≡ θφ
    @test convert(SphericalCoordinate{Float64}, 𝐱) ≡ θφ
    𝐯 = collect(𝐱)
    @test ZSphericalCoordinate(𝐯) ≡ φz
    @test ZSphericalCoordinate{Float64}(𝐯) ≡ φz
    @test SphericalCoordinate(𝐯) ≡ θφ
    @test SphericalCoordinate{Float64}(𝐯) ≡ θφ
end

@testset "SphericalHarmonic" begin
    @testset "Evaluation" begin
        S = SphericalHarmonic()
        @test copy(S) == S
        @test eltype(S) == ComplexF64
        @test eltype(axes(S,1)) == SphericalCoordinate{Float64}

        θ,φ = 0.1,0.2
        𝐱 = SphericalCoordinate(φ, θ)
        @test S[𝐱, Block(1)[1]] == S[𝐱,1] == sqrt(1/(4π))
        @test view(S,𝐱, Block(1)).indices[1] isa SphericalCoordinate
        @test S[𝐱, Block(1)] == [sqrt(1/(4π))]

        @test associatedlegendre(0)[0.1,1:2] ≈ [1.0,0.1]
        @test associatedlegendre(1)[0.1,1:2] ≈ [-0.9949874371066201,-0.29849623113198603]
        @test associatedlegendre(2)[0.1,1] ≈ 2.97

        for ℓ=0:5, m=-ℓ:ℓ
            @test S[𝐱, Block(ℓ+1)[m+ℓ+1]] ≈ sphericalharmonicy(ℓ, m, θ, φ) ≈ sqrt(factorial(ℓ-m) * (2ℓ+1)/(4π*factorial(ℓ+m))) * associatedlegendre(ℓ, m, cos(θ)) * exp(im*m*φ)
            @test sphericalharmonicy(ℓ, m, 𝐱) ≈ sphericalharmonicy(ℓ, m, θ, φ)
            @test sphericalharmonicy(ℓ, m, SVector(𝐱)) ≈ sphericalharmonicy(ℓ, m, θ, φ)
        end

        𝐬 = SVector(𝐱)
        @test S[𝐬, Block(1)[1]] ≈ sqrt(1/(4π))

        @test S[𝐱,Block(2)] ≈ 0.5sqrt(3/π)*[1/sqrt(2)*sin(θ)exp(-im*φ),cos(θ),-1/sqrt(2)*sin(θ)exp(im*φ)]
        @test S[𝐱,Block(3)] ≈ [0.25sqrt(15/2π)sin(θ)^2*exp(-2im*φ),
                                0.5sqrt(15/2π)sin(θ)cos(θ)exp(-im*φ),
                                0.25sqrt(5/π)*(3cos(θ)^2-1),
                                -0.5sqrt(15/2π)sin(θ)cos(θ)exp(im*φ),
                                0.25sqrt(15/2π)sin(θ)^2*exp(2im*φ)]
        @test S[𝐱,Block(4)] ≈ [0.125sqrt(35/π)sin(θ)^3*exp(-3im*φ),
                            0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(-2im*φ),
                            0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(-im*φ),
                            0.25sqrt(7/π)*(5cos(θ)^3-3cos(θ)),
                            -0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(im*φ),
                            0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(2im*φ),
                            -0.125sqrt(35/π)sin(θ)^3*exp(3im*φ)]

        @test S[𝐱,Block.(1:4)] == S[𝐱,Block.(Base.OneTo(4))] == [S[𝐱,Block(1)]; S[𝐱,Block(2)]; S[𝐱,Block(3)]; S[𝐱,Block(4)]] 
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
            N = 5
            S = SphericalHarmonic()
            𝐠,T = ContinuumArrays.plan_grid_transform(S, Block(N))

            f = S * [1:20; zeros(∞)]
            V = f[𝐠]
            @test T * V ≈ [1:20; zeros(5)]

            N = 2
            S = SphericalHarmonic()[:,Block.(Base.OneTo(N))]
            𝐱 = axes(S,1)

            P = factorize(S)
            @test eltype(P) == ComplexF64
            c = P \ (𝐱 -> 1).(𝐱)
            @test blocksize(c,1) == blocksize(S,2)
            @test c == S \ (𝐱 -> 1).(𝐱)
            @test (S * c)[SphericalCoordinate(0.1,0.2)] ≈ 1


            @test S \ (S * [zeros(3); 1]) ≈ [zeros(3); 1]

            f = (x,y,z) -> 1 + x + y + z
            c = S \ splat(f).(𝐱)
            u = S * c
            p = SphericalCoordinate(0.1,0.2)
            @test u[p] ≈ 1+sum(p)

            x = grid(SphericalHarmonic(), 5)
            P = plan_transform(SphericalHarmonic(), 5)
            @test P * splat(f).(x) ≈ [c; zeros(5)]
        end

        @testset "adaptive" begin
            S = SphericalHarmonic()
            𝐱 = axes(S,1)
            u = S * (S \ (𝐱 -> 1).(𝐱))
            @test u[SphericalCoordinate(0.1,0.2)] ≈ 1

            f = c -> exp(-100*c.θ^2)
            u = S * (S \ f.(𝐱))
            r = SphericalCoordinate(0.1,0.2)
            @test u[r] ≈ f(r)

            f = c -> ((x,y,z) = c; 1 + x + y + z)
            u = S * (S \ f.(𝐱))
            p = SphericalCoordinate(0.1,0.2)
            @test u[p] ≈ 1+sum(p)

            f = c -> ((x,y,z) = c; exp(x)*cos(y*sin(z)))
            u = S * (S \ f.(𝐱))
            @test u[p] ≈ f(p)
        end
    end
end

@testset "RealSphericalHarmonic" begin
    @testset "Real Evaluation" begin
        S = SphericalHarmonic()
        R = RealSphericalHarmonic()
        @test eltype(axes(R,1)) == SphericalCoordinate{Float64}
        θ,φ = 0.1,0.2
        x = SphericalCoordinate(φ, θ)
        @test R[x, Block(1)[1]] ≈ R[x,1] ≈ sqrt(1/(4π))
        @test R[x, Block(2)][1] ≈ S[x, Block(2)][2]
        # Careful here with the (-1) conventions?
        @test R[x, Block(2)][3] ≈ 1/sqrt(2)*(S[x, Block(2)[1]]-S[x, Block(2)[3]])
        @test R[x, Block(2)][2] ≈ im/sqrt(2)*(S[x, Block(2)][1]+S[x, Block(2)][3])
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
            𝐱 = axes(S,1)

            P = factorize(S)
            @test eltype(P) == Float64
            c = P \ (𝐱 -> 1).(𝐱)
            @test blocksize(c,1) == blocksize(S,2)
            @test c == S \ (𝐱 -> 1).(𝐱)
            @test (S * c)[SphericalCoordinate(0.1,0.2)] ≈ 1

            f = c -> ((x,y,z) = c; 1 + x + y + z)
            u = S * (S \ f.(𝐱))
            p = SphericalCoordinate(0.1,0.2)
            @test u[p] ≈ 1+sum(p)
        end

        @testset "adaptive" begin
            S = RealSphericalHarmonic()
            𝐱 = axes(S,1)
            u = S * (S \ (𝐱 -> 1).(𝐱))
            @test u[SphericalCoordinate(0.1,0.2)] ≈ 1

            f = c -> exp(-100*c.θ^2)
            u = S * (S \ f.(𝐱))
            r = SphericalCoordinate(0.1,0.2)
            @test u[r] ≈ f(r)

            f = c -> ((x,y,z) = c; 1 + x + y + z)
            u = S * (S \ f.(𝐱))
            p = SphericalCoordinate(0.1,0.2)
            @test u[p] ≈ 1+sum(p)

            f = c -> ((x,y,z) = c; exp(x)*cos(y*sin(z)))
            u = S * (S \ f.(𝐱))
            @test u[p] ≈ f(p)
        end
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
end


@testset "PDEs" begin
    @testset "Laplacian basics" begin
        S = SphericalHarmonic()
        R = RealSphericalHarmonic()
        S𝐱 = axes(S,1)
        R𝐱 = axes(R,1)
        SΔ = Laplacian(S𝐱)
        RΔ = Laplacian(R𝐱)
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


    @testset "Eigenvalues of spherical Laplacian" begin
        S = SphericalHarmonic()
        𝐱 = axes(S,1)
        Δ = Laplacian(𝐱)
        @test Δ isa Laplacian
        # define some explicit spherical harmonics
        Y_20 = c -> 1/4*sqrt(5/π)*(-1+3*cos(c.θ)^2)
        Y_3m3 = c -> 1/8*exp(-3*im*c.φ)*sqrt(35/π)*sin(c.θ)^3
        Y_41 = c -> -3/8*exp(im*c.φ)*sqrt(5/π)*cos(c.θ)*(-3+7*cos(c.θ)^2)*sin(c.θ) # note phase difference in definitions
        # check that the above correctly represents the respective spherical harmonics
        cfsY20 = S \ Y_20.(𝐱)
        @test cfsY20[Block(3)[3]] ≈ 1
        cfsY3m3 = S \ Y_3m3.(𝐱)
        @test cfsY3m3[Block(4)[1]] ≈ 1
        cfsY41 = S \ Y_41.(𝐱)
        @test cfsY41[Block(5)[6]] ≈ 1
        # Laplacian evaluation and correct eigenvalues
        @test (Δ*S*cfsY20)[SphericalCoordinate(0.7,0.2)] ≈ -6*Y_20(SphericalCoordinate(0.7,0.2))
        @test (Δ*S*cfsY3m3)[SphericalCoordinate(0.1,0.36)] ≈ -12*Y_3m3(SphericalCoordinate(0.1,0.36))
        @test (Δ*S*cfsY41)[SphericalCoordinate(1/3,6/7)] ≈ -20*Y_41(SphericalCoordinate(1/3,6/7))
    end

    @testset "Laplacian of expansions in complex spherical harmonics" begin
        S = SphericalHarmonic()
        𝐱 = axes(S,1)
        Δ = Laplacian(𝐱)
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
        @test (Δ*S*(S\f1.(𝐱)))[SphericalCoordinate(2.12,1.993)]  ≈ Δf1(SphericalCoordinate(2.12,1.993))
        @test (Δ*S*(S\f2.(𝐱)))[SphericalCoordinate(3.108,1.995)] ≈ Δf2(SphericalCoordinate(3.108,1.995))
        @test (Δ*S*(S\f3.(𝐱)))[SphericalCoordinate(0.737,0.239)] ≈ Δf3(SphericalCoordinate(0.737,0.239))
        @test (Δ*S*(S\f4.(𝐱)))[SphericalCoordinate(0.162,0.162)] ≈ Δf4(SphericalCoordinate(0.162,0.162))
        @test (Δ*S*(S\f5.(𝐱)))[SphericalCoordinate(0.1111,0.999)] ≈ Δf5(SphericalCoordinate(0.1111,0.999))
    end

    @testset "Laplacian of expansions in real spherical harmonics" begin
        R = RealSphericalHarmonic()
        𝐱 = axes(R,1)
        Δ = Laplacian(𝐱)
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
        @test (Δ*R*(R\f1.(𝐱)))[SphericalCoordinate(2.12,1.993)]  ≈ Δf1(SphericalCoordinate(2.12,1.993))
        @test (Δ*R*(R\f2.(𝐱)))[SphericalCoordinate(3.108,1.995)] ≈ Δf2(SphericalCoordinate(3.108,1.995))
        @test (Δ*R*(R\f3.(𝐱)))[SphericalCoordinate(0.737,0.239)] ≈ Δf3(SphericalCoordinate(0.737,0.239))
        @test (Δ*R*(R\f4.(𝐱)))[SphericalCoordinate(0.162,0.162)] ≈ Δf4(SphericalCoordinate(0.162,0.162))
        @test (Δ*R*(R\f5.(𝐱)))[SphericalCoordinate(0.1111,0.999)] ≈ Δf5(SphericalCoordinate(0.1111,0.999))
    end

    @testset "Laplacian raised to integer power, adaptive" begin
        S = SphericalHarmonic()
        𝐱 = axes(S,1)
        @test Laplacian(𝐱) isa Laplacian
        @test Laplacian(𝐱)^2 isa Laplacian
        @test Laplacian(𝐱)^3 isa Laplacian
        f1  = c -> cos(c.θ)^2
        Δ_f1 = c -> -1-3*cos(2*c.θ)
        Δ2_f1 = c -> 6+18*cos(2*c.θ)
        Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
        Δ = Laplacian(𝐱)
        Δ2 = Laplacian(𝐱)^2
        Δ3 = Laplacian(𝐱)^3
        t = SphericalCoordinate(0.122,0.993)
        @test (Δ*S*(S\f1.(𝐱)))[t] ≈ Δ_f1(t)
        @test (Δ^2*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ2_f1(t)
        @test (Δ^3*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ3_f1(t)
    end

    @testset "Finite basis Laplacian, complex" begin
        S = SphericalHarmonic()[:,Block.(Base.OneTo(10))]
        𝐱 = axes(S,1)
        @test Laplacian(𝐱) isa Laplacian
        @test Laplacian(𝐱)^2 isa Laplacian
        @test Laplacian(𝐱)^3 isa Laplacian
        f1  = c -> cos(c.θ)^2
        Δ_f1 = c -> -1-3*cos(2*c.θ)
        Δ2_f1 = c -> 6+18*cos(2*c.θ)
        Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
        Δ = Laplacian(𝐱)
        Δ2 = Laplacian(𝐱)^2
        Δ3 = Laplacian(𝐱)^3
        t = SphericalCoordinate(0.122,0.993)
        @test (Δ*S*(S\f1.(𝐱)))[t] ≈ Δ_f1(t)
        @test (Δ^2*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ2_f1(t)
        @test (Δ^3*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ3_f1(t)
    end

    @testset "Finite basis Laplacian, real" begin
        S = RealSphericalHarmonic()[:,Block.(Base.OneTo(10))]
        𝐱 = axes(S,1)
        @test Laplacian(𝐱) isa Laplacian
        @test Laplacian(𝐱)^2 isa Laplacian
        @test Laplacian(𝐱)^3 isa Laplacian
        f1  = c -> cos(c.θ)^2
        Δ_f1 = c -> -1-3*cos(2*c.θ)
        Δ2_f1 = c -> 6+18*cos(2*c.θ)
        Δ3_f1 = c -> -36*(1+3*cos(2*c.θ))
        Δ = Laplacian(𝐱)
        Δ2 = Laplacian(𝐱)^2
        Δ3 = Laplacian(𝐱)^3
        t = SphericalCoordinate(0.122,0.993)
        @test (Δ*S*(S\f1.(𝐱)))[t] ≈ Δ_f1(t)
        @test (Δ^2*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ2_f1(t)
        @test (Δ^3*S*(S\f1.(𝐱)))[t] ≈ (Δ*Δ*Δ*S*(S\f1.(𝐱)))[t] ≈ Δ3_f1(t)
    end

    @testset "abs(Δ)^α - Basics of absolute Laplacian powers" begin
        # Set 1
        α = 1/3
        S = SphericalHarmonic()
        S𝐱 = axes(S,1)
        SΔα = AbsLaplacian(S𝐱,α)
        Δ = Laplacian(S𝐱)
        @test copy(SΔα) == SΔα
        @test SΔα isa AbsLaplacian
        @test SΔα isa QuasiArrays.LazyQuasiMatrix
        @test axes(SΔα) == (axes(S,1),axes(S,1))
        @test abs(Δ) == -Δ == AbsLaplacian(axes(Δ,1),1)
        @test abs(Δ)^α == SΔα
        # Set 2
        α = 7/13
        S = SphericalHarmonic()
        S𝐱 = axes(S,1)
        SΔα = AbsLaplacian(S𝐱,α)
        Δ = Laplacian(S𝐱)
        @test copy(SΔα) == SΔα
        @test SΔα isa AbsLaplacian
        @test SΔα isa QuasiArrays.LazyQuasiMatrix
        @test axes(SΔα) == (axes(S,1),axes(S,1))
        @test abs(Δ) == -Δ == AbsLaplacian(axes(Δ,1),1)
        @test abs(Δ)^α == SΔα
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
end

@testset "sum" begin
    S = SphericalHarmonic()
    R = RealSphericalHarmonic()
    @test sum(S; dims=1)[:,1:10] ≈ sum(R; dims=1)[:,1:10] ≈ [sqrt(4π) zeros(1,9)]

    x = axes(S,1)
    @test sum(S * (S \ ones(x))) ≈ sum(R * (R \ ones(x))) ≈ 4π
    f = x -> cos(x[1]*sin(x[2]+x[3]))
    @test sum(S * (S \ f.(x))) ≈ sum(R * (R \ f.(x))) ≈ 11.946489824270322609

    @test sum(1 for (x,y,z) in UnitSphere()) ≈ 4π

    @test [sum(S[𝐱,k]'S[𝐱,j] for 𝐱 in UnitSphere()) for k=1:10, j=1:10] ≈ [sum(R[𝐱,k]'R[𝐱,j] for 𝐱 in UnitSphere()) for k=1:10, j=1:10] ≈ I
    @test S'S == R'R == Eye(∞)
end

@testset "representation theory" begin
    α,β,γ = 0.1,0.2,0.3
    ρ = RotZYZ(α,β,γ)
    𝐱 = SphericalCoordinate(0,0)
    S = SphericalHarmonic()
    for ℓ = 0:5
        @test S[ρ*𝐱, Block(ℓ+1)] ≈ conj(wignerD(ℓ, α, β, γ)) * S[𝐱, Block(ℓ+1)]
    end
end


struct IncompleteMultivariateOP <: BivariateOrthogonalPolynomial{Float64} end
Base.axes(::IncompleteMultivariateOP) = Inclusion((-1.0..1)^2), blockedrange(Base.oneto(∞))

@test_throws "Overload" IncompleteMultivariateOP()[SVector(0.1,0.2),2]
@test_throws "Overload" IncompleteMultivariateOP()[SVector(0.1,0.2),Block(2)]
@test_throws "Overload" IncompleteMultivariateOP()[SVector(0.1,0.2),Block(2)[2]]
@test_throws "Overload" IncompleteMultivariateOP()[SVector(0.1,0.2),[1,2]]

