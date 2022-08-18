#ifndef method_h
#define method_h

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void jacobi(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w);
void sor(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w);
void sd(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter);
void cg(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter);
void bicg(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter);
void bicg_stab(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter);

void pcg(SparseMatrix<double> A, VectorXd b, VectorXd x0, SparseMatrix<double> Ml, SparseMatrix<double> Mr, double tol, int maxIter);
void pbicg(SparseMatrix<double> A, VectorXd b, VectorXd x0, SparseMatrix<double> Ml, SparseMatrix<double> Mr, double tol, int maxIter);
void pbicg_stab(SparseMatrix<double> A, VectorXd b, VectorXd x0, SparseMatrix<double> Ml, SparseMatrix<double> Mr, double tol, int maxIter);

SparseMatrix<double> precond_jacobi(SparseMatrix<double> A);
 
#endif