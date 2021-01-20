using SphericalHarmonics, StaticArrays, Test, InfiniteArrays, LinearAlgebra, BlockArrays
import SphericalHarmonics: ZSphericalCoordinate, associatedlegendre, grid, SphereTrav

@testset "SphereTrav" begin
    A = SphereTrav([1 2 3; 4 0 0])
    @test A == [1, 2, 4, 3]
    @test A[Block(2)] == [2,4,3]
    B = SphereTrav([1 2 3 4 5; 6 7 8 0 0; 9 0 0 0 0 ])
    @test B == [1, 2, 6, 3, 4, 7, 9, 8, 5]
end

@testset "SphericalCoordinate" begin
    @test SphericalCoordinate(0.2,0.1) ≈ ZSphericalCoordinate(0.1,cos(0.2))
    @test SphericalCoordinate(0.2,0.1) == SVector(SphericalCoordinate(0.2,0.1))
    @test ZSphericalCoordinate(0.1,0.2) == SVector(ZSphericalCoordinate(0.1,0.2))

    @test norm(SphericalCoordinate(0.1,0.2)) === norm(ZSphericalCoordinate(0.1,cos(0.2))) === 1.0
    @test SphericalCoordinate(0.1,0.2) in UnitSphere()
    @test ZSphericalCoordinate(0.1,cos(0.2)) in UnitSphere()
end

@testset "Evaluation" begin
    S = SphericalHarmonic()
    @test eltype(axes(S,1)) == SphericalCoordinate{Float64}

    θ,φ = 0.1,0.2
    x = SphericalCoordinate(θ,φ)
    @test S[x, Block(1)[1]] == S[x,1] == sqrt(1/(4π))
    @test view(S,x, Block(1)).indices[1] isa SphericalCoordinate
    @test S[x, Block(1)] == [sqrt(1/(4π))]

    @test associatedlegendre(0)[0.1,1:2] ≈ [1.0,0.1]
    @test associatedlegendre(1)[0.1,1:2] ≈ [-0.9949874371066201,-0.29849623113198603]
    @test associatedlegendre(2)[0.1,1] ≈ 2.9699999999999998

    @test S[x,Block(2)] ≈ 0.5sqrt(3/π)*[1/sqrt(2)*sin(θ)exp(-im*φ),cos(θ),-1/sqrt(2)*sin(θ)exp(im*φ)]
    @test S[x,Block(3)] ≈ [0.25sqrt(15/2π)sin(θ)^2*exp(-2im*φ),0.5sqrt(15/2π)sin(θ)cos(θ)exp(-im*φ),
                            0.25sqrt(5/π)*(3cos(θ)^2-1),-0.5sqrt(15/2π)sin(θ)cos(θ)exp(im*φ),
                            0.25sqrt(15/2π)sin(θ)^2*exp(2im*φ)]
    @test S[x,Block(4)] ≈ [0.125sqrt(35/π)sin(θ)^3*exp(-3im*φ),
        0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(-2im*φ),
        0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(-im*φ),
        0.25sqrt(7/π)*(5cos(θ)^3-3cos(θ)),
        -0.125sqrt(21/π)sin(θ)*(5cos(θ)^2-1)*exp(im*φ),
        0.25sqrt(105/2π)sin(θ)^2*cos(θ)*exp(2im*φ),
        -0.125sqrt(35/π)sin(θ)^3*exp(3im*φ)]

    @test S[x,Block.(1:4)] == [S[x,Block(1)]; S[x,Block(2)]; S[x,Block(3)]; S[x,Block(4)]]
end

@testset "Expansion" begin
    N = 2
    S = SphericalHarmonic()[:,Block.(Base.OneTo(N))]
    
    @test size(S,2) == 4
    g = grid(S)
    @test eltype(g) == SphericalCoordinate{Float64}
    @testset "compare with FastTransforms.jl/examples/sphere.jl" begin
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
        
    P = factorize(S)
    @test eltype(P) == Float64
    xyz = axes(S,1)
    c = P \ (xyz -> 1).(xyz)
    @test blocksize(c,1) == blocksize(S,2)
    @test c == S \ (xyz -> 1).(xyz)
    @test (S * c)[SphericalCoordinate(0.1,0.2)] ≈ 1
end