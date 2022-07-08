# Iterative Methods for Sparse Linear Systems
Implementation and testing of iterative methods for sparse systems of linear equations. 

Linear Systems:

1. Finite element discretization of 2D Poisson's equation with variable coefficients on regular grid (same as FD in this particular case) (Poisson1)          symmetric positive definite matrix
2. Finite element method Poisson solver with constant coefficients on irregular grid (Poisson2) - symmetric matrix
3. Finite difference using Discrete Exterior Calculus (DEC) - non-symmetric matrix

Methods:

  1. Jacobi iterations
  2. Weighted Jacobi iterations
  3. Successive overrelaxation
  4. Gauss-Seidel method 
  5. Steepest Descent
  6. Conjugate gradient (CG) and its preconditioned version
  7. Biconjugate gradient (BiCG) and its preconditioned version
  8. Biconjugate Gradient Stabilized (BiCGStab) + preconditioned
  9. GMRES - in progress

Preconditioners:

  1. Jacobi
  2. Incomplete Cholesky
  3. Incomplete LU
