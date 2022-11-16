#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "method.h"
#include "gnuplot-iostream.h"


using namespace std;
using namespace Eigen;

void jacobi(const SparseMatrix<double> &A, const VectorXd &b, const VectorXd &x0, const double tol, const int maxIter, float w, VectorXd &res, SparseMatrix<double> &xk) {

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

	xk = x0.sparseView();

	int iter = 0;
	double rknorm = (b - A * xk).norm() / bnorm;

	res[iter] = rknorm;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		xk = M * xk + b;
		xk = Dinvs * xk;

		rknorm = (b - A * xk).norm() / bnorm;
		res[iter] = rknorm;
	}
}

void sor(const SparseMatrix<double> &A, const VectorXd &b, const VectorXd &x0, const double tol, const int maxIter, float w, VectorXd &res, SparseMatrix<double> &xk) {
	
	MatrixXd D = A.diagonal().asDiagonal();
	MatrixXd L = -A.triangularView<StrictlyLower>();
	MatrixXd U = -A.triangularView<StrictlyUpper>();
	SparseMatrix<double> Ds = D.sparseView();
	SparseMatrix<double> Ls = L.sparseView();
	SparseMatrix<double> Us = U.sparseView();

	MatrixXd DLinv = (D - w * L).inverse();
	SparseMatrix<double> DLinvs = DLinv.sparseView();

	double bnorm = b.norm();

	xk = x0.sparseView();

	int iter = 0;
	double rknorm = (b - A * xk).norm() / bnorm;

	res[iter] = rknorm;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		xk = ((1 - w) * Ds + w * Us) * xk + w * b;
		xk = DLinvs * xk;

		rknorm = (b - A * xk).norm() / bnorm;
		res[iter] = rknorm;
	}
}

void sd(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
	
	double bnorm = b.norm();

	xk = x0.sparseView();

	int iter = 0;
	SparseMatrix<double> rk = b - A * xk;
	double rknorm = rk.norm() / bnorm;

	res[iter] = rknorm;

	double aux1, aux2, alphak;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		aux1 = (rk.transpose() * rk).norm();
		aux2 = (rk.transpose() * (A * rk)).norm();
		alphak = aux1 / aux2;
		
		xk = xk + alphak * rk;

		rk = b - A * xk;
		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}

	//cout << endl << res << endl;
}

void cg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {

	xk = x0.sparseView();
	SparseMatrix<double> rk = A * xk - b;
	SparseMatrix<double> pk = -rk;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> apk;
	double rkrk, alpha, beta;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A * pk;
		rkrk = (rk.transpose() * rk).norm();
		alpha = rkrk / (pk.transpose() * apk).norm();

		xk = xk + alpha * pk;
		rk = rk + alpha * apk;

		beta = (rk.transpose() * rk).norm() / rkrk;

		pk = -rk + beta * pk;

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void pcg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
	SparseMatrix<double> uk = x0.sparseView();
	SparseMatrix<double> rk = Ml * (A * uk - b);
	SparseMatrix<double> pk = -rk;
	xk = Mr * uk;
	SparseMatrix<double> A1 = Ml * A * Mr;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> apk;
	double rkrk, alpha, beta;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A1 * pk;
		rkrk = (rk.transpose() * rk).norm();
		alpha = rkrk / (pk.transpose() * apk).norm();

		uk = uk + alpha * pk;
		xk = Mr * uk;

		rk = rk + alpha * apk;

		beta = (rk.transpose() * rk).norm() / rkrk;

		pk = -rk + beta * pk;

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void bicg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
    xk = x0.sparseView();
	SparseMatrix<double> rk = b - A * xk;
	SparseMatrix<double> rhk = rk;
	SparseMatrix<double> pk = rk;
	SparseMatrix<double> phk = rhk;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> apk;
	double rhkrk, alpha, beta;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A * pk;
		rhkrk = (rhk.transpose() * rk).norm();
		alpha = rhkrk / (phk.transpose() * apk).norm();

		xk = xk + alpha * pk;
		rk = rk - alpha * apk;
		rhk = rhk - alpha * (A.transpose() * phk);

		beta = (rhk.transpose() * rk).norm() / rhkrk;

		pk = rk + beta * pk;
		phk = rhk + beta * phk;

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void pbicg(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
	SparseMatrix<double> uk = x0.sparseView();
	SparseMatrix<double> rk = Ml * (b - A * uk);
	SparseMatrix<double> rhk = rk;
	SparseMatrix<double> pk = rk;
	SparseMatrix<double> phk = rhk;

	xk = Mr * uk;
	SparseMatrix<double> A1 = Ml * A * Mr;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> apk;
	double rhkrk, alpha, beta;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A1 * pk;
		rhkrk = (rhk.transpose() * rk).norm();
		
		alpha = rhkrk / (phk.transpose() * apk).norm();

		uk = uk + alpha * pk;
		xk = Mr * uk;

		rk = rk - alpha * apk;
		rhk = rhk - alpha * (A1.transpose() * phk);

		beta = (rhk.transpose() * rk).norm() / rhkrk;

		pk = rk + beta * pk;
		phk = rhk + beta * phk;

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void bicg_stab(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
	xk = x0.sparseView();
	SparseMatrix<double> rk = b - A * xk;
	SparseMatrix<double> pk = rk;
	SparseMatrix<double> rh = rk;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> sk, apk, ask;
	double rhrk, alpha, beta, wk;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A * pk;
		rhrk = (rh.transpose() * rk).norm();

		alpha = rhrk / (rh.transpose() * apk).norm();
		
		sk = rk - alpha * apk;
		ask = A * sk;

		wk = (ask.transpose() * sk).norm() / (ask.transpose() * ask).norm();

		xk = xk + alpha * pk + wk * sk;
		rk = sk - wk * ask;
		
		beta = (rh.transpose() * rk).norm() / rhrk * (alpha / wk);

		pk = rk + beta * (pk - wk * apk);

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void pbicg_stab(const SparseMatrix<double>& A, const VectorXd& b, const VectorXd& x0, const SparseMatrix<double>& Ml, const SparseMatrix<double>& Mr, const double tol, const int maxIter, VectorXd& res, SparseMatrix<double>& xk) {
	SparseMatrix<double> uk = x0.sparseView();
	SparseMatrix<double> rk = Ml * (b - A * uk);
	SparseMatrix<double> pk = rk;
	SparseMatrix<double> rh = rk;

	xk = Mr * uk;
	SparseMatrix<double> A1 = Ml * A * Mr;

	double bnorm = b.norm();
	double rknorm = rk.norm() / bnorm;

	int iter = 0;
	res[iter] = rknorm;

	SparseMatrix<double> sk, apk, ask;
	double rhrk, alpha, beta, wk;

	while (iter < maxIter && rknorm > tol) {
		iter++;

		apk = A1 * pk;
		rhrk = (rh.transpose() * rk).norm();

		alpha = rhrk / (rh.transpose() * apk).norm();

		sk = rk - alpha * apk;
		ask = A1 * sk;

		wk = (ask.transpose() * sk).norm() / (ask.transpose() * ask).norm();

		uk = uk + alpha * pk + wk * sk;
		xk = Mr * uk;
		rk = sk - wk * ask;

		beta = (rh.transpose() * rk).norm() / rhrk * (alpha / wk);

		pk = rk + beta * (pk - wk * apk);

		rknorm = rk.norm() / bnorm;
		res[iter] = rknorm;
	}
}

void precond_jacobi(const SparseMatrix<double>& A, SparseMatrix<double>& M) {
	MatrixXd D = A.diagonal().asDiagonal();

	for (int i = 0; i < A.cols(); ++i) {
		if (D(i, i) != 0) {
			D(i, i) = 1 / D(i, i);
		}
	}

	M = D.sparseView();
}

// matrix A must be possitive definite in order for incomplete Cholesky to work!!!
void precond_ichol(const SparseMatrix<double>& A, SparseMatrix<double>& Ml, SparseMatrix<double>& Mr) {
	int n = A.cols();
	MatrixXd L(n, n);
	L = MatrixXd::Zero(n, n);

	for (int j = 0; j < n; j++) {		
		for (int i = j; i < n; i++) {
			double sumk = 0;
			if (i == j) {
				for (int k = 0; k < j; k++) {
					sumk += pow(L(i, k), 2.0);
				}
				L(i, j) = sqrt(A.coeff(i,j) - sumk);
			}
			else {
				double sumk = 0;
				for (int k = 0; k < j; k++) {
					sumk += L(i, k) * L(j, k);
				}
				L(i, j) = (A.coeff(i,j) - sumk) / L(j, j);
			}
		}
	}

	Ml = L.inverse().sparseView();
	Mr = L.transpose().inverse().sparseView();
}