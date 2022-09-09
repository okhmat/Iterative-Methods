#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include "method.h"
#include "gnuplot-iostream.h"

using namespace std;
using namespace Eigen;

int main()
{
	//----------opening csv files of the system----------
	ifstream A, b;
	
	A.open("C:\\Users\\Puye\\project\\Iterative-Methods\\linear_systems\\DEC\\A.csv"); //location of file 'A'
	b.open("C:\\Users\\Puye\\project\\Iterative-Methods\\linear_systems\\DEC\\b.csv"); //location of file 'b'


	if (!A) {
		cout << "csv file 'A' open failed" << endl;
	}
	else {
		cout << "csv file 'A' open success" << endl;
	}
	
	if (!b) {
		cout << "csv file 'b' open failed" << endl;
	}
	else {
		cout << "csv file 'b' open success" << endl;
	}

	//----------get number of rows in csv files----------
	int N = 0, el_num = 0, b_num = 0;
	string line;
	
	while (getline(A, line)) {
		el_num++;
	}
	cout << el_num << endl;
	
	A.clear();
	A.seekg(0, ios_base::beg);
	
	
	while (getline(b, line)) {
		b_num++;
	}
	cout << b_num << endl;

	b.clear();
	b.seekg(0, ios_base::beg);
	
	N = 4211 + 1; // NEED TO CHANGE THIS PART

	//----------initialize x0, bvec, tol, maxIter----------
	VectorXd x0(N, 1);
	x0.setOnes();
	
	VectorXd bvec(N, 1);

	double tol = 1E-15;
	int maxIter = 1000;

	//----------get matrix A and right-hand side vector b from csv files----------
	int i = 0, j = 0;
	string col, row, val, b_val;

	typedef Eigen::Triplet<double> T;
	std::vector<T> tripletList;
	tripletList.reserve(el_num);

	/*
	while(i < N) { 
		i++;

		getline(b, b_val, '\n');
		bvec(i - 1) = stod(b_val);
	}
	*/

	for (i = 0; i < b_num; i++) {
		getline(b, row, ' ');
		getline(b, col, ' ');
		getline(b, b_val, '\n');

		bvec(stoi(row)) = stoi(b_val);
	}
	
	while (j < el_num) {
		j++;

		getline(A, row, ',');
		getline(A, col, ',');
		getline(A, val, '\n');

		tripletList.push_back(T(stoi(col), stoi(row), stod(val)));
	}

	SparseMatrix<double> mat(N,N);

	mat.setFromTriplets(tripletList.begin(), tripletList.end());

	//----------initialize results----------
	VectorXd res1(maxIter), res2(maxIter), res3(maxIter), res4(maxIter), res5(maxIter), res6(maxIter), res7(maxIter), res8(maxIter);
	SparseMatrix<double> xk1, xk2, xk3, xk4, xk5, xk6, xk7, xk8;

	//SparseMatrix<double> M;
	//precond_jacobi(mat, M);
	//----------run all the iterative solvers----------
	auto start = std::chrono::steady_clock::now();
	
	jacobi(mat, bvec, x0, tol, maxIter, 1, res1, xk1);
	auto t1 = std::chrono::steady_clock::now();
	chrono::duration<double> elapsed_seconds = t1 - start;
	cout << endl << "jacobi run time: " << elapsed_seconds.count() << "sec" << endl;

	jacobi(mat, bvec, x0, tol, maxIter, 2.0/3.0, res2, xk2);
	auto t2 = std::chrono::steady_clock::now();
	elapsed_seconds = t2 - t1;
	cout << endl << "weighted jacobi run time: " << elapsed_seconds.count() << "sec" << endl;

	sor(mat, bvec, x0, tol, maxIter, 1, res3, xk3);

	sor(mat, bvec, x0, tol, maxIter, 2.0/3.0, res4, xk4);

	cg(mat, bvec, x0, tol, maxIter, res5, xk5);
	bicg(mat, bvec, x0, tol, maxIter, res6, xk6);
	bicg_stab(mat, bvec, x0, tol, maxIter, res7, xk7);
	
	//pcg(mat, bvec, x0, M, M, tol, maxIter, res8, xk8);

	auto end = std::chrono::steady_clock::now();
	elapsed_seconds = end - start;
	cout << endl << "time: " << elapsed_seconds.count() << "sec" << endl;

	//----------plot residual vs iteration graph----------
	Gnuplot gp;

	vector<pair<int, double>> pts_jacobi, pts_weighted_jacobi, pts_gauss_seidel, pts_sor, pts_cg, pts_bicg, pts_bicg_stab, pts_pcg_jacobi;
	for (int x = 0; x < res1.size(); x++) {
		pts_jacobi.push_back(make_pair(x + 1, res1[x]));
		pts_weighted_jacobi.push_back(make_pair(x + 1, res2[x]));
		pts_gauss_seidel.push_back(make_pair(x + 1, res3[x]));
		pts_sor.push_back(make_pair(x + 1, res4[x]));
		pts_cg.push_back(make_pair(x + 1, res5[x]));
		pts_bicg.push_back(make_pair(x + 1, res6[x]));
		pts_bicg_stab.push_back(make_pair(x + 1, res7[x]));
		pts_pcg_jacobi.push_back(make_pair(x + 1, res8[x]));
	}

	gp << "set format y '10^{% L}'\n";
	gp << "set logscale y\n";
	gp << "plot" << gp.file1d(pts_jacobi) << "with lines title 'jacobi'," 
				 << gp.file1d(pts_weighted_jacobi) << "with lines title 'weighted jacobi',"
				 << gp.file1d(pts_gauss_seidel) << "with lines title 'gauss seidel',"
				 << gp.file1d(pts_sor) << "with lines title 'sor',"
				 << gp.file1d(pts_cg) << "with lines title 'cg',"
			     //<< gp.file1d(pts_bicg) << "with lines title 'bicg',"
				 //<< gp.file1d(pts_bicg_stab) << "with lines title 'bicg stab',"
			     //<< gp.file1d(pts_pcg_jacobi) << "with lines title 'pcg jacobi',"
				 <<	endl;

	return 0;
}

