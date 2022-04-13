//1D unsteady heat conduction 
//Formulation taken from Prof Wagner ME424 course
#include<Eigen/Sparse>
#include<vector>
#include <iostream>
#include<Eigen/IterativeLinearSolvers>
#include<Eigen/SparseLU>
#include<fstream>
using namespace Eigen;
typedef Eigen::SparseMatrix<double> SparseMatrixXd; // declares a column-major sparse matrix type of double
//typedef Eigen::Triplet<double> T;

int main()
{
	double L = 1.0; //length
	double D = 1.0; //diffusion coefficient
	double rho = 1.0; // density
	double phiIC = 0.0; //initial condition
	double phiL = 1.0; //left BC
	double phiR = 0.0; //Right BC
	double dA = 1.0; //Area
	double totalTime = 0.5; //Total simulation time

	//Finite volume mesh solution parameters
	int nc = 20; //number of cell
	double dt = 0.0005; //timestep
	int theta = 0; // 0 for explicit and 1 for implicit

	//set up mesh
	double dx = L / double(nc); //cell width
	VectorXd xn(nc); //location of the FV center
	for (int ic = 0; ic < nc; ic++)
	{
		xn(ic) = ic * dx + dx / 2;
	}

	//std::cout << xn << std::endl;
	//define the sparse matrices using Eigen
	SparseMatrixXd matM(nc, nc);
	SparseMatrixXd matK(nc, nc);
	SparseMatrixXd LHS(nc, nc);
	matM.reserve(VectorXd::Constant(nc, 3));
	matK.reserve(VectorXd::Constant(nc, 3));
	LHS.reserve(VectorXd::Constant(nc, 3));

	VectorXd RHS0(nc); RHS0.setZero();
	VectorXd RHS(nc);

	//for the center cell
	double dV = dA * dx;
	for (int i = 1; i < (nc - 1); i++) {
		matK.coeffRef(i, i - 1) = -D * dA / dx;
		matK.coeffRef(i, i ) = 2* D * dA / dx;
		matK.coeffRef(i, i + 1) = -D * dA / dx;
		matM.coeffRef(i, i ) = rho*dV;
	}

	//for the left BC
	matK.coeffRef(0, 0) = 3*D * dA / dx;
	matK.coeffRef(0, 1) = -D * dA / dx;
	matM.coeffRef(0, 0) = rho * dV;
	RHS0(0) = 2 * D * dA  * phiL / dx;


	//for the right BC
	matK.coeffRef(nc-1, nc-2) = -D * dA / dx;
	matK.coeffRef(nc-1, nc-1) =  3 * D * dA / dx;
	//std::cout << matK << std::endl;
	matM.coeffRef(nc-1, nc-1) = rho * dV;
	//std::cout << matM << std::endl;
	RHS0(nc-1) = 2 * D * dA * phiR / dx;

	//std::cout << matK << std::endl;
	//std::cout << matM << std::endl;
	//std::cout << RHS << std::endl;

	LHS = matM / dt + theta * matK;
	//std::cout << LHS << std::endl;

	//set initial condition
	VectorXd phi(nc); phi.setZero();//assuming initial condition zero

	std::ofstream write_output("output.dat");
	//begin the time loop
	double time = 0;
	while (time < totalTime) {
		time = time + dt;
		RHS = (matM / dt - (1 - theta) * matK) * phi + RHS0;
		//std::cout << RHS << std::endl;
		BiCGSTAB<SparseMatrix<double> > solver;
		solver.compute(LHS);
		phi = solver.solve(RHS);
		std::cout << "#iterations:     " << solver.iterations() << std::endl;
		std::cout << "estimated error: " << solver.error() << std::endl;
		/* ... update b ... */
		phi = solver.solve(RHS); // solve again
		//std::cout << phi << std::endl;
		write_output << "current timestep: " << time << "\n";
		for (int ic = 0; ic < nc; ic++)
		{
			write_output << xn(ic)<< " " << phi(ic) << "\n";
		}

	}
	write_output.close();
	return 0;
}

