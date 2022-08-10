#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "method.h"

using namespace std;
using namespace Eigen;

void jacobi(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w) {

	MatrixXd Dinv = A.diagonal().asDiagonal();
	for (int i = 0; i < A.cols(); ++i) {
		if (Dinv(i, i) != 0) {
			Dinv(i, i) = w / Dinv(i, i);
		}
	}
	SparseMatrix<double> Dinvs = Dinv.sparseView();

	MatrixXd D = A.diagonal().asDiagonal();
	for (int i = 0; i < A.cols(); ++i) {
		if (D(i, i) != 0) {
			D(i, i) = D(i, i) / w;
		}
	}
	SparseMatrix<double> Ds = D.sparseView();

	SparseMatrix<double> M = Ds - A;

	double bnorm = b.norm();

	SparseMatrix<double> xk = x0.sparseView();

	int iter = 0;
	double rknorm = (b - A * xk).norm() / bnorm;

	VectorXd res(maxIter);
	res(iter) = rknorm;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		xk = M * xk + b;
		xk = Dinvs * xk;

		rknorm = (b - A * xk).norm() / bnorm;
		res(iter) = rknorm;
	}

	//cout << endl << res << endl;
}

void sor(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter, float w) {
	
	MatrixXd D = A.diagonal().asDiagonal();
	MatrixXd L = -A.triangularView<StrictlyLower>();
	MatrixXd U = -A.triangularView<StrictlyUpper>();
	SparseMatrix<double> Ds = D.sparseView();
	SparseMatrix<double> Ls = L.sparseView();
	SparseMatrix<double> Us = U.sparseView();

	MatrixXd DLinv = (D - w * L).inverse();
	SparseMatrix<double> DLinvs = DLinv.sparseView();

	double bnorm = b.norm();

	SparseMatrix<double> xk = x0.sparseView();

	int iter = 0;
	double rknorm = (b - A * xk).norm() / bnorm;

	VectorXd res(maxIter);
	res(iter) = rknorm;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		xk = ((1 - w) * Ds + w * Us) * xk + w * b;
		xk = DLinvs * xk;

		rknorm = (b - A * xk).norm() / bnorm;
		res(iter) = rknorm;
	}

	//cout << endl << res << endl;
}

void sd(SparseMatrix<double> A, VectorXd b, VectorXd x0, double tol, int maxIter) {
	
	double bnorm = b.norm();

	SparseMatrix<double> xk = x0.sparseView();

	int iter = 0;
	SparseMatrix<double> rk = b - A * xk;
	double rknorm = rk.norm() / bnorm;

	VectorXd res(maxIter);
	res(iter) = rknorm;

	double aux1, aux2, alphak;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		aux1 = (rk.transpose() * rk).norm();
		aux2 = (rk.transpose() * (A * rk)).norm();
		alphak = aux1 / aux2;
		
		xk = xk + alphak * rk;

		rk = b - A * xk;
		rknorm = rk.norm() / bnorm;
		res(iter) = rknorm;
	}

	//cout << endl << res << endl;
}

