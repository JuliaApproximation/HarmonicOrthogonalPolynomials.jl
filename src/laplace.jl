function laplacian(P::AbstractSphericalHarmonic; dims...)
     # Spherical harmonics are the eigenfunctions of the Laplace operator on the unit sphere
     P * Diagonal(mortar(Fill.((-(0:∞)-(0:∞).^2), 1:2:∞)))
end
