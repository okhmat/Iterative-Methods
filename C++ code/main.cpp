#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include "method.h"

using namespace std;
using namespace Eigen;

int main()
{
	int N = 3000;

	MatrixXd m = MatrixXd::Random(N,N);
	
	SparseMatrix<double> A;
	A = m.sparseView();

	VectorXd x0(N, 1);
	x0.setOnes();

	VectorXd b(N, 1);
	b.setOnes();

	double tol = 1E-15;
	int maxIter = 1000;

	auto start = std::chrono::steady_clock::now();

	jacobi(A, b, x0, tol, maxIter, 1);

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;

	cout << endl << elapsed_seconds.count() << endl;
	
	return 0;
}
