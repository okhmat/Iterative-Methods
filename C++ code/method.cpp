#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "method.h"

using namespace std;
using namespace Eigen;

void jacobi(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w = 1) {

	MatrixXd Dinv = A.diagonal().asDiagonal();
	for (int i = 0; i < A.cols(); ++i) {
		if (Dinv(i, i) != 0) {
			Dinv(i, i) = w / Dinv(i, i);
		}
	}
	SparseMatrix<double> Dinvs = Dinv.sparseView();

	MatrixXd D = A.diagonal().asDiagonal();
	for (int i = 0; i < A.cols(); ++i) {
		if (Dinv(i, i) != 0) {
			Dinv(i, i) = D(i, i) / w;
		}
	}
	SparseMatrix<double> Ds = D.sparseView();

	SparseMatrix<double> M = Ds - A;

	/*
	MatrixXd L = -A.triangularView<StrictlyLower>();
	MatrixXd U = -A.triangularView<StrictlyUpper>();
	MatrixXd LU = L + U;
	SparseMatrix<double> LUs = LU.sparseView();
	*/

	double bnorm = b.norm();

	SparseMatrix<double> xk = x0.sparseView();

	int iter = 0;
	double rk = (b - A * xk).norm() / bnorm;

	VectorXd res(maxIter);
	res(iter) = rk;

	while (iter < maxIter && rk > tol) {
		iter++;

		xk = M * xk + b;
		xk = Dinvs * xk;

		rk = (b - A * xk).norm() / bnorm;

		res(iter) = rk;
	}
}
