if __name__ == "__main__":
    import numpy as np
    import scipy as sp
    from scipy import sparse
    from scipy.sparse import linalg
    import matplotlib.pyplot as plt
    import time

    from jacobi import *
    from gen_linear_sys import get_system



    # create the simplest system to test the Jacobi Iterations
    N = 10
    A, b, x_exact = get_system(Nx=N, Ny=N)

    t0 = time.time()
    xk, res, err = jacobi(A=A, b=b, x0=np.ones((N*N, 1)), x=x_exact, maxiter=int(N**2/2))
    t1 = time.time()

    print("%0.2f", t1-t0, " sec")

    plt.plot(np.log10(res))
    plt.show()




