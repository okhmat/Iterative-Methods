#ifndef method_h
#define method_h

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void jacobi(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, float w, VectorXd& res, SparseMatrix<double>& xk);
void sor(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, float w, VectorXd& res, SparseMatrix<double>& xk);
void sd(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);
void cg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);
void bicg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);
void bicg_stab(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);

void pcg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);
void pbicg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);
void pbicg_stab(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk);

void precond_jacobi(const SparseMatrix<double>& A, SparseMatrix<double>& M);
void precond_ichol(const SparseMatrix<double>& A, SparseMatrix<double>& Ml, SparseMatrix<double>& Mr);

#endif