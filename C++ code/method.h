#ifndef method_h
#define method_h

#include <Eigen/Dense>
#include <Eigen/Sparse>

using namespace Eigen;

void jacobi(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w);
void sor(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w);
void sd(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter);

#endif