using HarmonicOrthogonalPolynomials, Plot
plotly()

G = HarmonicOrthogonalPolynomials.grid(SphericalHarmonic()[:,Block.(Base.OneTo(10))])
n,m = size(G)
g = append!(vec(G), [SphericalCoordinate(0,0), SphericalCoordinate(π,0)])
x,y,z = ntuple(k ->getindex.(g,k), 3); 
N = length(g)


i = [range(0; length=m, step=n); ]
j = [range(n; length=m-1, step=n); 0]
k = fill(N-2, m)

for ν = range(0; length=m-1, step=n)
    append!(i, range(ν; length=n-1))
    append!(i, range(ν+n; length=n-1))
    append!(j, range(ν+1; length=n-1))
    append!(j, range(ν+1; length=n-1))
    append!(k, range(ν+n; length=n-1))
    append!(k, range(ν+n+1; length=n-1))
end

append!(i, range(n*(m-1); length=n-1))
append!(i, range(0; length=n-1))
append!(j, range(n*(m-1)+1; length=n-1))
append!(j, range(n*(m-1)+1; length=n-1))
append!(k, range(0; length=n-1))
append!(k, range(1; length=n-1))

append!(i, [range(2n-1; length=m-1, step=n); n-1])
append!(j, range(n-1; length=m, step=n))
append!(k, fill(N-1, m))

mesh3d(x,y,z; connections = (i,j,k))