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
    # A_DEC, b_DEC, N_DEC = get_system_DEC(matrixtxt="linear_systems/DEC/A.csv", rhstxt="linear_systems/DEC/b.csv")
    A_Siem, b_Siem, N_Siem = get_system_Poisson2(matrixtxt="linear_systems/Poisson2/matrix.txt", rhstxt="linear_systems/Poisson2/rhs.txt")


    # check if these matrices are real, symmetric, positive definite
    # DEC
    # if (sp.sparse.linalg.norm(A_DEC - A_DEC.T, 'fro') == 0):
    #     print('matrix A is real symmetric')
    # else:
    #     print('matrix A is NOT real symmetric')
    # if (sp.sparse.linalg.eigs(A_DEC, k=1, which='SR', return_eigenvectors=False) > 0):
    #     print('the matrix is positive definite')
    # else:
    #     print('the matrix is NOT positive definite')
    #
    # for i in range(b_DEC.shape[0]):
    #     if (b_DEC[i] != 0):
    #         print(b_DEC[i])
    #         print(i)


    t0 = time.time()
    xk1, res1, err1 = jacobi(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem, 1)), maxiter=500)
    xk2, res2, err2 = weight_jacobi(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem, 1)), x=None, tol=1e-15, maxiter=500, w=2/3)
    #xk3, res3, err3 = gauss_seidel(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=10)
    #xk4, res4, err4 = sor(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=10)
    xk5, res5, err5, cost5 = sd(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=500)
    xk6, res6, err6, cost6 = cg(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=500)
    xk7, res7, err7, cost7 = bicg(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=500)
    xk8, res8, err8, cost8 = bicgstab(A=A_Siem, b=b_Siem, x0=np.ones((N_Siem,1)), x=None, tol=1e-15, maxiter=500)
    t1 = time.time()

    print(t1-t0, " sec")

    plt.figure(figsize=(17, 11))
    plt.plot(np.log10(res5), label='steepest descent')
    plt.plot(np.log10(res6), label='conjugate gradient')
    plt.plot(np.log10(res7), label='biconjugate gradient', linestyle='dotted')
    plt.plot(np.log10(res8), label='biconjugate gradient stabilized')

    plt.ylabel('relative residue',fontsize=18)
    plt.xlabel('iteration number',fontsize=18)
    plt.legend()




