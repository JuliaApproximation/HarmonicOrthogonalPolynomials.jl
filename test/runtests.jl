using SphericalHarmonics, StaticArrays, Test, InfiniteArrays
import SphericalHarmonics: ZSphericalCoordinate, associatedlegendre, grid


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
    @test eltype(axes(S,1)) == ZSphericalCoordinate{Float64}

    θ,φ = 0.1,0.2
    x = SphericalCoordinate(θ,φ)
    @test S[x, Block(1)[1]] == S[x,1] == sqrt(1/(4π))
    @test view(S,x, Block(1)).indices[1] isa ZSphericalCoordinate
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
    S = SphericalHarmonic()[:,Block.(1:N)]
    xyz = axes(S,1)
    f = xyz -> ((x,y,z) = xyz; exp(x+y*z))
    @test_broken size(S,2) == 4
    @test_broken grid(S)  # Should return grid with more than 4 points
    @test_broken factorize(S) # Should return a Factorization
end