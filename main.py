if __name__ == "__main__":
    import numpy as np
    import scipy as sp
    from scipy import sparse
    from scipy.sparse import linalg
    import matplotlib.pyplot as plt
    import time

    from methods import *
    from gen_linear_sys import *



    # create the simplest system to test the Jacobi Iterations
    N = 10
    A, b, x_exact = get_system_Poisson1(Nx=N, Ny=N) # notice that the system size here is Nx * Ny
    # this system is real symmetric and positive definite

    # uncomment if need to assemble matrix and rhs for DEC
    # DEC_sys_assembler(M_row="linear_systems/DEC/M_row.txt",
    #                   M_col="linear_systems/DEC/M_col.txt",
    #                   M_v="linear_systems/DEC/M_v.txt",
    #                   b_row = "linear_systems/DEC/b_row.txt",
    #                   b_v = "linear_systems/DEC/b_v.txt")
    A_DEC, b_DEC, N_DEC = get_system_DEC(matrixtxt="linear_systems/DEC/A.csv", rhstxt="linear_systems/DEC/b.csv")
    A_Siem, b_Siem, N_Siem = get_system_Poisson2(matrixtxt="linear_systems/Poisson2/matrix.txt", rhstxt="linear_systems/Poisson2/rhs.txt")
    # check if these matrices are real, symmetric, positive definite
    # DEC
    if (sp.sparse.linalg.norm(A_DEC - A_DEC.T, 'fro') == 0):
        print('matrix A is real symmetric')
    else:
        print('matrix A is NOT real symmetric')
    if (sp.sparse.linalg.eigs(A_DEC, k=1, which='SR', return_eigenvectors=False) > 0):
        print('the matrix is positive definite')
    else:
        print('the matrix is NOT positive definite')

    for i in range(b_DEC.shape[0]):
        if (b_DEC[i] != 0):
            print(b_DEC[i])
            print(i)


    t0 = time.time()
    xk1, res1, err1 = jacobi(A=A_DEC, b=b_DEC, x0=np.ones((N_DEC, 1)), maxiter=100)
    xk2, res2, err2 = weight_jacobi(A=A_DEC, b=b_DEC, x0=np.ones((N_DEC, 1)), x=None, tol=1e-15, maxiter=100, w=2/3)
    t1 = time.time()

    print(t1-t0, " sec")

    plt.plot(np.log10(res1))
    plt.show()




