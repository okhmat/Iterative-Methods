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
    xk = x0

    iter_count = 0
    rk = np.linalg.norm(b - A*xk) / bnorm
    res.append(rk)
    if (x is not None):
        xnorm = np.linalg.norm(x)
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

def weight_jacobi(A, b, x0, x, tol=1e-15, maxiter=100, w=2/3): # it's just Jacobi for w=1
    err = list();
    res = list()
    Dw = sp.sparse.spdiags(A.diagonal()/w, 0, A.shape[0], A.shape[1], format='csc')
    Dwinv = sp.sparse.spdiags(w/A.diagonal(), 0, A.shape[0], A.shape[1], format='csc')
    M = Dw - A # not completely without diagonal, but with a normalized diagonal
    bnorm = np.linalg.norm(b)
    xk = x0
    iter_count = 0
    rk = np.linalg.norm(b - A * xk) / bnorm
    res.append(rk)
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err.append(np.linalg.norm(x - xk) / xnorm)
    while (iter_count < maxiter and rk > tol):
        xk = M*xk + b
        xk = Dwinv*xk
        rk = np.linalg.norm(b - A * xk) / bnorm
        res.append(rk)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        iter_count += 1
    return xk, res, err

def gauss_seidel(A, b, x0, x, tol=1e-15, maxiter=100):
    err = list();
    res = list()
    D = sp.sparse.spdiags(A.diagonal(), 0, A.shape[0], A.shape[1], format='csc')
    L = -sp.sparse.tril(A, -1)
    U = -sp.sparse.triu(A, 1)
    DLinv = sp.sparse.linalg.inv(D - L)

    bnorm = np.linalg.norm(b)
    xk = x0
    iter_count = 0
    rk = np.linalg.norm(b - A * xk) / bnorm
    res.append(rk)
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err.append(np.linalg.norm(x - xk) / xnorm)

    while (iter_count < maxiter and rk > tol):
        xk = U*xk + b
        xk = DLinv*xk
        rk = np.linalg.norm(b - A*xk) / bnorm
        res.append(rk)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        iter_count += 1

    return xk, res, err

def sor(A, b, x0, x, tol=1e-15, maxiter=100, w=2/3):
    err = list();
    res = list()
    D = sp.sparse.spdiags(A.diagonal(), 0, A.shape[0], A.shape[1], format='csc')
    L = -sp.sparse.tril(A, -1)
    U = -sp.sparse.triu(A, 1)
    DwLinv = sp.sparse.linalg.inv(D - w*L)
    bnorm = np.linalg.norm(b)

    xk = x0
    iter_count = 0
    rk = np.linalg.norm(b - A * xk) / bnorm
    res.append(rk)
    if (x is not None):
        err.append(np.linalg.norm(x - xk) / xnorm)
        xnorm = np.linalg.norm(x)

    while (iter_count < maxiter and rk > tol):
        xk = ((1-w)*D + w*U)*xk + w*b
        xk = DwLinv*xk
        rk = np.linalg.norm(b - A*xk) / bnorm
        res.append(rk)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        iter_count += 1
    return xk, res, err

def quadratic_cost_f(A, x, b):
    return float(-0.5*(x.T@(A@x)) + x.T@b)

def sd(A, b, x0, x, tol=1e-15, maxiter=100):
    err = list()
    res = list()
    cost = list()
    bnorm = np.linalg.norm(b)
    xk = x0
    iter_count = 0
    rk = b - A@xk
    rknorm = np.linalg.norm(rk)/bnorm
    res.append(rknorm)
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err.append(np.linalg.norm(x - xk) / xnorm)
    cost.append(quadratic_cost_f(A, xk, b))

    while (iter_count < maxiter and rknorm > tol):
        rk = b - A@xk
        rknorm = np.linalg.norm(rk)/bnorm
        aux1 = rk.T@rk
        aux2 = rk.T@(A@rk)
        alphak = aux1 / aux2
        xk = xk + alphak*rk

        res.append(np.linalg.norm(rk) / bnorm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        cost.append(quadratic_cost_f(A, xk, b))
        iter_count += 1

    return xk, res, err, cost

def cg(A, b, x0, x, tol=1e-15, maxiter=10):
    err = list()
    xk = x0 # x0
    rk = A.dot(xk) - b # r0
    pk = -rk # p0
    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk)/ bnorm # normalized residual at the initial iteration

    res = [rk_norm] # accumulates residuals rk_norm at each iteration
    if (x is not None):
        err = [np.linalg.norm(x - xk)/xnorm]
        xnorm = np.linalg.norm(x)
    cost = [quadratic_cost_f(A, xk, b)]

    num_iter = 0 # iteration counter

    while (rk_norm > tol and num_iter < maxiter): # norm of the residual at the current iteration is greater than the preset tolerance

        apk = A.dot(pk) # numpy vector in both sparse and non-sparse cases
        rkrk = np.dot(rk.T, rk)#[0, 0] # since the vectors are numpy arrays, columns
        alpha = rkrk / np.dot(pk.T, apk)#[0][0]

        xk = xk + alpha * pk # x(k+1)

        rk = rk + alpha * apk

        beta = np.dot(rk.T, rk)/rkrk  #[0][0] / rkrk

        pk = -rk + beta * pk # p(k+1)

        num_iter += 1

        rk_norm = np.linalg.norm(rk) / bnorm # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk)/xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost


# for both sparse and non-sparse formats
# for split preconditioner
def pcg(A, b, x0, x, Ml, Mr, atol=1e-15, maxiter=10):
    uk = x0  # u0
    rk = A.dot(uk) - b  # r0
    # with hat
    rk = Ml.dot(rk)
    pk = -rk  # p0

    xk = Mr @ uk
    A = Ml @ A @ Mr

    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk) / bnorm  # normalized residual at the initial iteration

    res = [rk_norm]  # accumulates residuals rk_norm at each iteration
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err = [np.linalg.norm(x - xk) / xnorm]
    cost = [quadratic_cost_f(A, xk, b)]

    num_iter = 0  # iteration counter
    curve_x = [xk]  # to plot the vectors as the algorithm converges
    while rk_norm > atol and num_iter < maxiter:  # norm of the residual at the current iteration is greater than the preset tolerance

        apk = A.dot(pk)  # numpy vector in both sparse and non-sparse cases
        rkrk = np.dot(rk.T, rk)[0, 0]  # since the vectors are numpy arrays, columns
        alpha = rkrk / np.dot(pk.T, apk)[0][0]

        uk = uk + alpha * pk  # u(k+1)
        xk = Mr @ uk  # x(k+1)

        rk = rk + alpha * apk

        beta = np.dot(rk.T, rk)[0][0] / rkrk

        pk = -rk + beta * pk  # p(k+1)

        num_iter += 1

        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk) / np.linalg.norm(b)  # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost

def bicg(A, b, x0, x, tol=1e-15, maxiter=10):
    err = list()
    
    xk = x0
    rk = b - (A @ xk)
    rhk = rk
    pk = rk
    phk = rhk

    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk) / bnorm

    res = [rk_norm] # accumulates residuals rk_norm at each iteration
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err = [np.linalg.norm(x - xk) / xnorm]
    cost = [quadratic_cost_f(A, xk, b)]

    iter_count = 0

    while (rk_norm > tol and iter_count < maxiter):
        apk = A @ pk
        rhk_rk = rhk.T @ rk

        alpha = rhk_rk / (phk.T @ apk)

        xk = xk + alpha * pk
        rk = rk - alpha * apk
        rhk = rhk - alpha * (A.T @ phk)

        beta = (rhk.T @ rk) / rhk_rk

        pk = rk + beta * pk
        phk = rhk + beta * phk

        iter_count += 1

        rk_norm = np.linalg.norm(rk) / bnorm # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk)/xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost

def pbicg(A, b, x0, x, Ml, Mr, tol=1e-15, maxiter=10):
    uk = x0
    rk = b - (A @ uk)

    rk = Ml @ rk
    pk = rk
    rhk = rk
    phk = pk

    xk = Mr @ uk
    A = Ml @ A @ Mr

    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk) / bnorm

    res = [rk_norm]  # accumulates residuals rk_norm at each iteration
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err = [np.linalg.norm(x - xk) / xnorm]
    cost = [quadratic_cost_f(A, xk, b)]

    iter_count = 0

    while (rk_norm > tol and iter_count < maxiter):
        apk = A @ pk
        rhk_rk = rhk.T @ rk

        alpha = rhk_rk / (phk.T @ apk)

        uk = uk + alpha * pk
        xk = Mr @ uk

        rk = rk - alpha * apk
        rhk = rhk - alpha * (A.T @ phk)

        beta = (rhk.T @ rk) / rhk_rk

        pk = rk + beta * pk
        phk = rhk + beta * phk

        iter_count += 1

        rk_norm = np.linalg.norm(rk) / bnorm  # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost

def bicgstab(A, b, x0, x, tol=1e-15, maxiter=10):
    err = list()
    
    xk = x0
    rk = b - (A @ xk)
    pk = rk
    rh = rk  # r^0

    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk) / bnorm

    res = [rk_norm]  # accumulates residuals rk_norm at each iteration
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err = [np.linalg.norm(x - xk) / xnorm]
    cost = [quadratic_cost_f(A, xk, b)]

    iter_count = 0

    while (rk_norm > tol and iter_count < maxiter):
        apk = A @ pk
        rhrk = rh.T @ rk

        alpha = rhrk / (rh.T @ apk)

        sk = rk - alpha * apk
        ask = A @ sk

        wk = (ask.T @ sk) / (ask.T @ ask)

        xk = xk + alpha * pk + wk * sk
        rk = sk - wk * ask

        beta = (rh.T @ rk) / rhrk * (alpha / wk)

        pk = rk + beta * (pk - wk * apk)

        iter_count += 1

        rk_norm = np.linalg.norm(rk) / bnorm  # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost

def pbicgstab(A, b, x0, x, Ml, Mr, tol=1e-15, maxiter=10):
    uk = x0
    rk = b - (A @ uk)

    rk = Ml @ rk
    pk = rk
    rh = rk  # r^0

    xk = Mr @ uk
    A = Ml @ A @ Mr

    bnorm = np.linalg.norm(b)
    rk_norm = np.linalg.norm(rk) / bnorm

    res = [rk_norm]  # accumulates residuals rk_norm at each iteration
    if (x is not None):
        xnorm = np.linalg.norm(x)
        err = [np.linalg.norm(x - xk) / xnorm]
    cost = [quadratic_cost_f(A, xk, b)]

    iter_count = 0

    while (rk_norm > tol and iter_count < maxiter):
        apk = A @ pk
        rhrk = rh.T @ rk

        alpha = rhrk / (rh.T @ apk)

        sk = rk - alpha * apk
        ask = A @ sk

        wk = (ask.T @ sk) / (ask.T @ ask)

        uk = uk + alpha * pk + wk * sk
        xk = Mr @ uk
        rk = sk - wk * ask

        beta = (rh.T @ rk) / rhrk * (alpha / wk)

        pk = rk + beta * (pk - wk * apk)

        iter_count += 1

        rk_norm = np.linalg.norm(rk) / bnorm  # normalized residual at the k-th iteration
        res.append(rk_norm)
        if (x is not None):
            err.append(np.linalg.norm(x - xk) / xnorm)
        cost.append(quadratic_cost_f(A, xk, b))

    return xk, res, err, cost
