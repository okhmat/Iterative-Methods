import numpy as np
import scipy as sp
from scipy import sparse
import pandas as pd
import csv

# discretization of 2D Poisson's equation with variable coefficient a(x, y)
def get_system_Poisson1(Nx=10, Ny=10):
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


# assembles .csv files with the mtx and rhs elements from the solver's output
def DEC_sys_assembler(M_row, M_col, M_v, b_row, b_v):
    # Matrix
    cols = pd.read_csv(M_col, sep=" ", header=None)
    rows = pd.read_csv(M_row, sep=" ", header=None)
    vals = pd.read_csv(M_v, sep=" ", header=None)
    N = np.max(cols[:][0])  # dimensions of the matrix
    Nnnz = len(vals)

    A = sp.sparse.csc_matrix((vals[:][0], (rows[:][0] - 1, cols[:][0] - 1)), shape=(N, N), dtype='float64')
    # -1 in idx since in python enumeration starts from 0

    # convert those vectors to CSV
    with open('linear_systems/DEC/A.csv', 'w') as f:
        f.truncate()
        writer = csv.writer(f, delimiter=" ")
        for i in range(Nnnz):
            writer.writerow([rows[0][i] - 1, cols[0][i] - 1, vals[0][i]])

    # RHS
    rows = pd.read_csv(b_row, sep=" ", header=None)
    vals = pd.read_csv(b_v, sep=" ", header=None)

    b = np.zeros((N, 1), dtype='float64')
    b[rows[0][0] - 1, 0] = vals[0][0]

    # convert those vectors to CSV
    with open('linear_systems/DEC/b.csv', 'w') as f:
        f.truncate()  # clear the file
        writer = csv.writer(f, delimiter=" ")
        writer.writerow([rows[0][0] - 1, cols[0][0] - 1, vals[0][0]])


# A.csv and b.csv that were generated by the DEC_sys_assembler()
def get_system_DEC(matrixtxt, rhstxt):
    matrix = pd.read_csv(matrixtxt, sep=" ", header=None)
    matrix.columns = ["row_idx", "col_idx", "val"]
    N = np.max([np.max(matrix.row_idx), np.max(matrix.row_idx)])
    spmatrix = sp.sparse.csc_matrix((matrix.values[:, 2], (matrix.row_idx, matrix.col_idx)))

    rhs = pd.read_csv(rhstxt, sep=" ", header=None)
    rhs.columns = ["row_idx", "col_idx", "val"]
    print('DEC size of the system: ', N)

    sprhs = np.zeros((N, 1))
    sprhs[int(rhs.values[0, 0]), int(rhs.values[0, 1])] = rhs.values[0, 2]
    # sprhs = sp.sparse.csc_matrix(rhs.values[:, 1], (N, 1))

    return spmatrix, sprhs, N


def get_system_Poisson2(matrixtxt, rhstxt):
    matrix = pd.read_csv(matrixtxt, sep=" ", header=None)
    matrix.columns = ["row_idx", "col_idx", "val"]
    N = np.max([np.max(matrix.row_idx), np.max(matrix.row_idx)])+1
    spmatrix = sp.sparse.csc_matrix((matrix.values[:, 2], (matrix.row_idx, matrix.col_idx)))

    rhs = pd.read_csv(rhstxt, sep=" ", header=None)
    rhs.columns = ["row_idx", "val", "glb_idx"]
    aux = rhs.values[:, 0]
    for i in range(N):
        aux[i] = int(aux[i])
    sprhs = sp.sparse.csc_matrix(rhs.values[:, 1]).T  # in the local index
    # it's the row vector, need to return column vector

    assert (spmatrix.shape[0] == spmatrix.shape[1])
    assert (spmatrix.shape[1] == N)
    assert (sprhs.shape[0] == N)
    assert (sprhs.shape[1] == 1)

    print('Poisson2 size of the system: ', N)
    return spmatrix, sprhs.todense(), N
