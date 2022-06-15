import numpy as np
import scipy as sp
from scipy import sparse

def get_system(Nx=10, Ny=10):

    def a(x, y):
        assert (y.shape[1] == 1 and x.shape[1] == 1)  # outer product - the other dimensions of the vectors are not necessarily the same
        return np.cos(y) * np.cos(x.T)

    def u(x, y):
        return np.sin(4 * np.pi * y) * np.sin(2 * np.pi * x.T)

    def f(x, y):
        a1 = np.multiply(np.cos(y), np.sin(4 * np.pi * y))
        a2 = np.multiply(np.sin(x.T), np.cos(2 * np.pi * x.T))
        a3 = np.multiply(np.cos(x.T), np.sin(2 * np.pi * x.T))
        a4 = np.multiply(np.sin(y), np.cos(4 * np.pi * y))
        a5 = np.multiply(np.cos(y), np.sin(4 * np.pi * y))
        a6 = np.multiply(np.cos(x.T), np.sin(2 * np.pi * x.T))
        return 2 * np.pi * a1 * (a2 + 2 * np.pi * a3) + 4 * np.pi * (a4 + 4 * np.pi * a5) * a6

    def a1(i, j):
        return 0.5 * (a(x[i], y[j]) + a(x[i - 1], y[j]))

    def a2(i, j):
        return 0.5 * (a(x[i], y[j]) + a(x[i], y[j - 1]))

    dx = 1 / (Nx + 1)
    x = np.linspace(0, 1, Nx + 2).reshape(Nx + 2, 1)
    xi = x[1:-1].reshape(Nx, 1)  # interior points N

    # in y direction
    dy = dx
    y = x.copy()
    yi = xi.copy()

    A1 = 0.5 * a(x[:-1], yi) + 0.5 * a(x[1:], yi)
    A2 = 0.5 * a(xi, y[:-1]) + 0.5 * a(xi, y[1:])

    Dx = sp.sparse.diags([1, -1], [0, -1], shape=(Nx + 1, Nx))
    Dy = sp.sparse.diags([1, -1], [0, -1], shape=(Ny + 1, Ny))

    Ix = sp.sparse.diags([1], [0], shape=(Nx, Nx))
    Iy = sp.sparse.diags([1], [0], shape=(Ny, Ny))

    A1_diag = sp.sparse.diags(A1.T.reshape(1, (Nx + 1) * Ny), [0], shape=((Nx + 1) * Ny, (Nx + 1) * Ny))
    A2_diag = sp.sparse.diags(A2.T.reshape(1, (Ny + 1) * Nx), [0], shape=((Ny + 1) * Nx, (Ny + 1) * Nx))

    L = sp.sparse.kron(Dx.T, Iy) @ A1_diag @ sp.sparse.kron(Dx, Iy)  # symmetric matrix
    R = sp.sparse.kron(Ix, Dy.T) @ A2_diag @ sp.sparse.kron(Ix, Dy)  # symmetric matrix
    A = L / dx ** 2 + R / dy ** 2  # defined for the internal points only

    # right hand side (internal grid points only)
    F = f(xi, yi)
    Fvec = F.reshape(((Nx) * (Ny), 1))

    # exact solution
    Uexact = u(xi.reshape(Nx, 1), yi.reshape(Ny, 1))
    Uexactvec = Uexact.reshape(Nx * Ny, 1)

    return A, Fvec, Uexactvec

