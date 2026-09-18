// Standalone fast-tier CI target extracted from test_fem_poisson_2d.cpp
// (test_manufactured_solution, unchanged): verifies FEM Poisson convergence
// against a manufactured solution. The full test suite in
// test_fem_poisson_2d.cpp runs as its own (slow-tier) target. The few shared
// helpers it needs are copied below to keep this file self-contained.
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include "../../src/field_solver/fem_solver/tri_mesh.hpp"
#include "../../src/field_solver/fem_solver/FEM_Poisson_solver_2d.hpp"
#include "../../src/field_solver/fem_solver/fem_boundary.hpp"

using namespace std;
using namespace psum;
using namespace psum::field_solver;
using namespace psum::field_solver::fem_solver;
using namespace fem_boundary_creator;

const string test_backend = "eigen_sparselu_cpu";

inline void solve_with_backend(FEM_Poisson_solver_2d& solver, const string& backend_name, Eigen::VectorXd& phi, const Eigen::VectorXd& source) {
	if (backend_name == "native" || backend_name.size() < 3 ||
		backend_name.substr(backend_name.size() - 3) == "cpu") {
		solver.solve(phi, source);
		return;
	}

	sycl::queue q{sycl::default_selector_v};
	double* phi_dev = sycl::malloc_device<double>(phi.size(), q);
	double* source_dev = sycl::malloc_device<double>(source.size(), q);
	if (phi_dev == nullptr || source_dev == nullptr) {
		if (phi_dev) sycl::free(phi_dev, q);
		if (source_dev) sycl::free(source_dev, q);
		throw runtime_error("failed to allocate SYCL device arrays for FEM test");
	}

	q.memcpy(source_dev, source.data(), sizeof(double) * source.size()).wait();
	solver.solve(phi_dev, source_dev);
	q.memcpy(phi.data(), phi_dev, sizeof(double) * phi.size()).wait();
	sycl::free(phi_dev, q);
	sycl::free(source_dev, q);
}

inline tri_mesh make_rect_trimesh(double x0, double y0, double x1, double y1, int nx, int ny) {
	vector<Eigen::Vector2d> nodes;
	nodes.reserve((nx + 1) * (ny + 1));
	for (int j = 0; j <= ny; j++)
		for (int i = 0; i <= nx; i++)
			nodes.push_back(Eigen::Vector2d(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny));
	vector<array<size_t, 3>> elements;
	elements.reserve(2 * nx * ny);
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++) {
			size_t n00 = j * (nx + 1) + i;
			size_t n10 = j * (nx + 1) + i + 1;
			size_t n01 = (j + 1) * (nx + 1) + i;
			size_t n11 = (j + 1) * (nx + 1) + i + 1;
			elements.push_back({n00, n10, n11});
			elements.push_back({n00, n11, n01});
		}
	return tri_mesh(nodes, elements);
}

int main()
{
	cout << "\n=== Manufactured solution test (FEM) ===" << endl;

	vector<double> errL2;

	for(int k=1;k<=6;k++)
	{
		int N = 40*k;
		tri_mesh mesh = make_rect_trimesh(0, 0, 1, 1, N, N);

		FEM_Poisson_solver_2d solver;
		solver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cartesian,
			{},
			{
				Dirichlet_region(mesh, [](double, double y) { return y > 1 - 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double, double y) { return y < 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double x, double) { return x > 1 - 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double x, double) { return x < 1e-10; }) = 0
			},
			8.854e-12
		);

		Eigen::VectorXd Phi(mesh.n_nodes()), rho(mesh.n_nodes());
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double lap = -2*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y);
			rho(i) = -8.854e-12 * lap;
		}
		solve_with_backend(solver, test_backend, Phi, rho);

		double L2=0;
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double exact = sin(M_PI*x)*sin(M_PI*y);
			double e = abs(Phi(i) - exact);
			L2 += e*e;
		}
		L2 = sqrt(L2 / Phi.size());
		errL2.push_back(L2);
		cout<<"N="<<N<<"\tL2="<<L2<<endl;
	}

	double order = log(errL2.front() / errL2.back()) / log(6);
	cout << "Estimated order ~ " << order << endl;

	// Convergence acceptance criteria: linear FEM on this uniform refinement
	// path is 2nd order; the error must stay finite, decrease and remain small.
	for (double e : errL2) {
		if (!isfinite(e)) { cerr << "FAIL: non-finite L2 error" << endl; return 1; }
	}
	if (!(errL2.front() > errL2.back())) { cerr << "FAIL: L2 error did not decrease" << endl; return 1; }
	if (errL2.back() >= 1e-3) { cerr << "FAIL: final L2 error too large: " << errL2.back() << endl; return 1; }
	if (!(order > 1.5 && order < 2.5)) { cerr << "FAIL: convergence order " << order << " not ~2" << endl; return 1; }
	cout << "FEM convergence test PASSED" << endl;
	return 0;
}
