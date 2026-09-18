#include <sycl/sycl.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <functional>
#include <Eigen/Core>
#include "../../src/field/multi_patch/multi_patch_grid.hpp"
#include "../../src/field/multi_patch/duplicate_map.hpp"
#include "../../src/field_solver/fem_solver/multi_patch_to_tri_mesh.hpp"
#include "../../src/field_solver/fem_solver/FEM_Poisson_solver_2d.hpp"
#include "../../src/field_solver/fem_solver/MPG_Poisson_solver_2d.hpp"
#include "../../src/field_solver/fem_solver/fem_boundary.hpp"

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace psum::field_solver::fem_solver;
using namespace psum::field::multi_patch;
using namespace fem_boundary_creator;

const string test_backend = "eigen_sparselu_cpu";

bool backend_uses_device(const string& backend_name) {
	if (backend_name == "native") return false;
	if (backend_name.size() < 3) return false;
	return backend_name.substr(backend_name.size() - 3) != "cpu";
}

void solve_with_backend(FEM_Poisson_solver_2d& solver, const string& backend_name, Eigen::VectorXd& phi, const Eigen::VectorXd& source) {
	if (!backend_uses_device(backend_name)) {
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

void solve_with_backend(MPG_Poisson_solver_2d& solver, const string& backend_name, vector<double>& phi, const vector<double>& source) {
	if (!backend_uses_device(backend_name)) {
		solver.solve(phi, source);
		return;
	}

	sycl::queue q{sycl::default_selector_v};
	double* phi_dev = sycl::malloc_device<double>(phi.size(), q);
	double* source_dev = sycl::malloc_device<double>(source.size(), q);
	if (phi_dev == nullptr || source_dev == nullptr) {
		if (phi_dev) sycl::free(phi_dev, q);
		if (source_dev) sycl::free(source_dev, q);
		throw runtime_error("failed to allocate SYCL device arrays for MPG test");
	}

	q.memcpy(source_dev, source.data(), sizeof(double) * source.size()).wait();
	solver.solve(phi_dev, source_dev);
	q.memcpy(phi.data(), phi_dev, sizeof(double) * phi.size()).wait();
	sycl::free(phi_dev, q);
	sycl::free(source_dev, q);
}

void solve_with_backend(MPG_Poisson_solver_2d& solver, const string& backend_name, Eigen::VectorXd& phi, const Eigen::VectorXd& source) {
	if (!backend_uses_device(backend_name)) {
		solver.solve(phi, source);
		return;
	}

	sycl::queue q{sycl::default_selector_v};
	double* phi_dev = sycl::malloc_device<double>(phi.size(), q);
	double* source_dev = sycl::malloc_device<double>(source.size(), q);
	if (phi_dev == nullptr || source_dev == nullptr) {
		if (phi_dev) sycl::free(phi_dev, q);
		if (source_dev) sycl::free(source_dev, q);
		throw runtime_error("failed to allocate SYCL device arrays for MPG test");
	}

	q.memcpy(source_dev, source.data(), sizeof(double) * source.size()).wait();
	solver.solve(phi_dev, source_dev);
	q.memcpy(phi.data(), phi_dev, sizeof(double) * phi.size()).wait();
	sycl::free(phi_dev, q);
	sycl::free(source_dev, q);
}

void write_plt(const string& filename, const tri_mesh& mesh, const Eigen::VectorXd& phi) {
	ofstream fp(filename);
	fp << "TITLE = \"MPG FEM Poisson result\"" << endl;
	fp << "VARIABLES = \"X\", \"Y\", \"phi\"" << endl;
	fp << "ZONE T=\"solution\", N=" << mesh.n_nodes() << ", E=" << mesh.n_elements()
	   << ", F=FEPOINT, ET=TRIANGLE" << endl;
	for (size_t i = 0; i < mesh.n_nodes(); i++)
		fp << mesh.node_position(i).x() << "\t" << mesh.node_position(i).y() << "\t" << phi(i) << endl;
	for (size_t e = 0; e < mesh.n_elements(); e++) {
		auto& tri = mesh.elements()[e];
		fp << (tri[0] + 1) << "\t" << (tri[1] + 1) << "\t" << (tri[2] + 1) << endl;
	}
	fp.close();
}

void test_basic_conversion() {
	cout << "\n=== Basic conversion test ===" << endl;
	sycl::queue q;

	vector<double> lower = {0.0, 0.0};
	vector<double> upper = {2.0, 1.0};
	vector<int> coarse_cells = {2, 1};

	{
		vector<int> res_exp = {3, 4};
		multi_patch_grid<2> mpg(q, lower, upper, coarse_cells, res_exp);
		auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);

		auto coarse = mpg.coarse_grid();
		int n0 = 1 << res_exp[0];
		int n1 = 1 << res_exp[1];
		int expected_nodes = (n0 + 1) * (n0 + 1) + (n1 + 1) * (n1 + 1)
			- (n0 + 1) - ((n1 + 1) - (n0 + 1));
		int expected_elems = 2 * n0 * n0 + 3 * n0 + 2 * n1 * n1;

		cout << "nodes: " << mesh.n_nodes() << " (approx expected: " << expected_nodes << ")" << endl;
		cout << "elements: " << mesh.n_elements() << " (approx expected: " << expected_elems << ")" << endl;
		cout << "boundary_edges: " << mesh.n_boundary_edges() << endl;

		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			auto p = mesh.node_position(i);
			if (p.x() < -1e-10 || p.x() > 2.0 + 1e-10 || p.y() < -1e-10 || p.y() > 1.0 + 1e-10) {
				cout << "ERROR: node " << i << " out of bounds: (" << p.x() << "," << p.y() << ")" << endl;
			}
		}
		cout << "Bounds check passed." << endl;
	}
}

void test_uniform_manufactured() {
	cout << "\n=== Uniform mesh manufactured solution ===" << endl;
	sycl::queue q;

	vector<double> lower = {0.0, 0.0};
	vector<double> upper = {1.0, 1.0};
	vector<int> coarse_cells = {1, 1};

	vector<double> errL2;
	for (int k = 1; k <= 3; k++) {
		vector<int> res_exp = {k + 3};
		multi_patch_grid<2> mpg(q, lower, upper, coarse_cells, res_exp);
		auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);

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
			1.0
		);

		Eigen::VectorXd Phi(mesh.n_nodes()), rho(mesh.n_nodes());
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double lap = -2 * M_PI * M_PI * sin(M_PI * x) * sin(M_PI * y);
			rho(i) = -lap;
		}
		solve_with_backend(solver, test_backend, Phi, rho);

		double L2 = 0;
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double exact = sin(M_PI * x) * sin(M_PI * y);
			double e = abs(Phi(i) - exact);
			L2 += e * e;
		}
		L2 = sqrt(L2 / Phi.size());
		errL2.push_back(L2);
		cout << "N=" << (1 << (k + 3)) << "\tL2=" << L2 << endl;
	}

	double order = log(errL2.front() / errL2.back()) / log(3);
	cout << "Estimated order ~ " << order << endl;
}

void test_mixed_resolution() {
	cout << "\n=== Mixed resolution manufactured solution ===" << endl;
	sycl::queue q;

	vector<double> lower = {0.0, 0.0};
	vector<double> upper = {2.0, 2.0};
	vector<int> coarse_cells = {2, 2};

	vector<double> errL2;
	for (int k = 0; k < 3; k++) {
		vector<int> res_exp = {k + 2, k + 3, k + 3, k + 2};
		multi_patch_grid<2> mpg(q, lower, upper, coarse_cells, res_exp);
		auto [mesh, dm] = multi_patch_to_tri_mesh(mpg);

		FEM_Poisson_solver_2d solver;
		solver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cartesian,
			{},
			{
				Dirichlet_region(mesh, [](double, double y) { return y > 2 - 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double, double y) { return y < 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double x, double) { return x > 2 - 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double x, double) { return x < 1e-10; }) = 0
			},
			1.0
		);

		Eigen::VectorXd Phi(mesh.n_nodes()), rho(mesh.n_nodes());
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double lap = -(M_PI / 2) * (M_PI / 2) * sin(M_PI * x / 2) * sin(M_PI * y / 2)
			             - (M_PI / 2) * (M_PI / 2) * sin(M_PI * x / 2) * sin(M_PI * y / 2);
			rho(i) = -lap;
		}
		solve_with_backend(solver, test_backend, Phi, rho);

		double L2 = 0;
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			double exact = sin(M_PI * x / 2) * sin(M_PI * y / 2);
			double e = abs(Phi(i) - exact);
			L2 += e * e;
		}
		L2 = sqrt(L2 / Phi.size());
		errL2.push_back(L2);
		cout << "exp={" << res_exp[0] << "," << res_exp[1] << "} nodes=" << mesh.n_nodes()
			 << " elems=" << mesh.n_elements() << " L2=" << L2 << endl;

		if (k == 2) {
			std::vector<double> mpg_data(dm.n_index);
			scatter_to_mpg(dm, Phi.data(), mpg_data.data());
			ofstream fp("ans_mpg_scattered.plt");
			mpg.template plot<double, 1>(fp, mpg_data.data(), std::nan(""), var_loc::nodeCentered);
			fp.close();
			cout << "Wrote ans_mpg_scattered.plt" << endl;
		}
	}

	double order = log(errL2.front() / errL2.back()) / log(2);
	cout << "Estimated order ~ " << order << endl;
}

void test_mpg_solver_eigen_matches_vector() {
	cout << "\n=== MPG solver Eigen/vector API equivalence ===" << endl;
	sycl::queue q;

	vector<double> lower = {0.0, 0.0};
	vector<double> upper = {1.0, 1.0};
	vector<int> coarse_cells = {2, 2};
	vector<int> res_exp = {3, 4, 4, 3};
	multi_patch_grid<2> mpg(q, lower, upper, coarse_cells, res_exp);

	MPG_Poisson_solver_2d solver;
	solver.init(
		test_backend,
		mpg,
		MPG_Poisson_solver_2d::Cartesian,
		{
			Dirichlet_region(mpg, [](double, double y) { return y > 1 - 1e-10; }) = 0,
			Dirichlet_region(mpg, [](double, double y) { return y < 1e-10; }) = 0,
			Dirichlet_region(mpg, [](double x, double) { return x > 1 - 1e-10; }) = 0,
			Dirichlet_region(mpg, [](double x, double) { return x < 1e-10; }) = 0
		},
		1.0
	);

	vector<double> source_vec(mpg.contentSize(var_loc::nodeCentered));
	for (size_t i = 0; i < source_vec.size(); i++) {
		auto p = mpg.position<grid_element::node>(i);
		source_vec[i] = 2.0 * M_PI * M_PI * sin(M_PI * p.x()) * sin(M_PI * p.y());
	}

	vector<double> phi_vec(source_vec.size());
	solve_with_backend(solver, test_backend, phi_vec, source_vec);

	Eigen::VectorXd source_eigen(source_vec.size());
	Eigen::VectorXd phi_eigen(source_vec.size());
	for (size_t i = 0; i < source_vec.size(); i++)
		source_eigen(i) = source_vec[i];
	solve_with_backend(solver, test_backend, phi_eigen, source_eigen);

	double max_diff = 0.0;
	for (size_t i = 0; i < source_vec.size(); i++)
		max_diff = max(max_diff, abs(phi_vec[i] - phi_eigen(i)));

	cout << "max vector/eigen diff = " << max_diff << endl;
	if (max_diff > 1e-10)
		throw runtime_error("MPG solver Eigen/vector API mismatch");
}

int main() {
	test_basic_conversion();
	cout << endl;
	test_uniform_manufactured();
	cout << endl;
	test_mixed_resolution();
	cout << endl;
	test_mpg_solver_eigen_matches_vector();
	cout << endl;
	return 0;
}
