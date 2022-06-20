import numpy as np
import scipy as sp
from scipy import sparse

# x is the exact solution for the relative error
def jacobi(A, b, x0, x=None, tol=1e-15, maxiter=50): # A is a sparse matrix; better to use csc format
    err = list(); res = list()
    # D = sp.sparse.spdiags(A.diagonal(), 0, A.shape[0], A.shape[1], format='csc')
    Dinv = sp.sparse.spdiags(1/(A.diagonal()), 0, A.shape[0], A.shape[1], format='csc')
    L = -sp.sparse.tril(A, -1)
    U = -sp.sparse.triu(A, 1)
    LU = L + U
    bnorm = np.linalg.norm(b)
    if (x is not None):
        xnorm = np.linalg.norm(x)
    xk = x0
    iter_count = 0
    rk = np.linalg.norm(b - A*xk) / bnorm
    res.append(rk)
    if (x is not None):
        err.append(np.linalg.norm(x - xk) / xnorm)
    while (iter_count < maxiter and rk > tol):
        xk = LU*xk + b
        xk = Dinv*xk
        rk = np.linalg.norm(b - A*xk) / bnorm
        res.append(rk)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        iter_count += 1
    return xk, res, err
