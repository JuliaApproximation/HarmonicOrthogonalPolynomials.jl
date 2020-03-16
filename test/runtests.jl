using SphericalHarmonics, Test
import SphericalHarmonics: ZSphericalCoordinate

@test SphericalCoordinate(0.1,0.2) ≈ ZSphericalCoordinate(0.1,cos(0.2))
@test SphericalCoordinate(0.1,0.2) == SVector(SphericalCoordinate(0.1,0.2))
@test ZSphericalCoordinate(0.1,0.2) == SVector(ZSphericalCoordinate(0.1,0.2))


@test norm(SphericalCoordinate(0.1,0.2)) === norm(ZSphericalCoordinate(0.1,cos(0.2))) === 1.0
@test SphericalCoordinate(0.1,0.2) in UnitSphere()
@test ZSphericalCoordinate(0.1,cos(0.2)) in UnitSphere()

S = SphericalHarmonic()
@test eltype(axes(S,1)) == ZSphericalCoordinate{Float64}

x = SphericalCoordinate(0.1,0.2)
@test S[x, Block(1)[1]] == S[x,1] == 1
@test view(S,x, Block(1)).indices[1] isa ZSphericalCoordinate
@test S[x, Block(1)] == [1.0]

S[x,Block(2)] == 
S[x, Block.(1:5)]
