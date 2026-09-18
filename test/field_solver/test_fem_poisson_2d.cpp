#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <unordered_map>
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

tri_mesh make_rect_trimesh(double x0, double y0, double x1, double y1, int nx, int ny) {
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

void write_plt(const string& filename, const tri_mesh& mesh, const Eigen::VectorXd& phi) {
	ofstream fp(filename);
	fp << "TITLE = \"FEM Poisson result\"" << endl;
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

double interp_node(const tri_mesh& mesh, const Eigen::VectorXd& phi, double tx, double ty) {
	double best_dist = 1e30;
	double best_val = 0;
	for (size_t i = 0; i < mesh.n_nodes(); i++) {
		auto& p = mesh.node_position(i);
		double d = (p.x() - tx) * (p.x() - tx) + (p.y() - ty) * (p.y() - ty);
		if (d < best_dist) { best_dist = d; best_val = phi(i); }
	}
	return best_val;
}

fem_neumann_boundary make_neumann(const tri_mesh& mesh, double val, const function<bool(double, double)>& fn) {
	fem_neumann_boundary nb = Neumann_region(mesh, fn);
	nb = val;
	return nb;
}

fem_robin_boundary make_robin(const tri_mesh& mesh, double val, const function<pair<double,double>(double,double)>& coeff_fn, const function<bool(double, double)>& region_fn) {
	unordered_map<size_t, array<double, 2>> nodes;
	for (size_t i = 0; i < mesh.n_nodes(); i++) {
		auto& p = mesh.node_position(i);
		if (region_fn(p.x(), p.y())) nodes[i] = {p.x(), p.y()};
	}
	fem_robin_boundary rb;
	rb.set_related_nodes(nodes);
	rb.set_coeff_function(coeff_fn);
	rb = val;
	return rb;
}

struct FEM_Poisson_solver_2d_inspector : public FEM_Poisson_solver_2d {
	using FEM_Poisson_solver_2d::init;

	size_t neumann_edge_count() const {
		size_t count = 0;
		for (auto& info : boundary_edge_infos_)
			if (info.is_neumann) count++;
		return count;
	}

	size_t robin_edge_count() const {
		size_t count = 0;
		for (auto& info : boundary_edge_infos_)
			if (info.is_robin) count++;
		return count;
	}
};

void test_boundary_edge_classification()
{
	cout << "\n=== Boundary edge classification test ===" << endl;
	tri_mesh mesh = make_rect_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2);

	FEM_Poisson_solver_2d_inspector neumann_solver;
	neumann_solver.init(
		test_backend,
		mesh,
		FEM_Poisson_solver_2d::Cartesian,
		{
			make_neumann(mesh, 1.0, [](double, double y) { return y < 1e-10; }),
			make_neumann(mesh, 2.0, [](double x, double) { return x < 1e-10; })
		},
		{
			Dirichlet_region(mesh, [](double x, double y) { return x > 1 - 1e-10 && y > 1 - 1e-10; }) = 0
		},
		1.0
	);
	if (neumann_solver.neumann_edge_count() != 4)
		throw runtime_error("Neumann edge classification mismatch");

	FEM_Poisson_solver_2d_inspector robin_solver;
	robin_solver.init(
		test_backend,
		mesh,
		FEM_Poisson_solver_2d::Cartesian,
		{},
		{
			make_robin(mesh, 1.0, [](double, double) { return pair<double,double>{1.0, 1.0}; }, [](double, double y) { return y < 1e-10; }),
			make_robin(mesh, 2.0, [](double, double) { return pair<double,double>{2.0, 1.0}; }, [](double x, double) { return x < 1e-10; })
		},
		{
			Dirichlet_region(mesh, [](double x, double y) { return x > 1 - 1e-10 && y > 1 - 1e-10; }) = 0
		},
		1.0
	);
	if (robin_solver.robin_edge_count() != 4)
		throw runtime_error("Robin edge classification mismatch");
}

void test_robin_zero_beta_rejected()
{
	cout << "\n=== Robin zero beta rejection test ===" << endl;
	tri_mesh mesh = make_rect_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2);
	bool threw = false;
	try {
		FEM_Poisson_solver_2d solver;
		solver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cartesian,
			{},
			{
				make_robin(mesh, 1.0, [](double, double) { return pair<double,double>{1.0, 0.0}; }, [](double, double y) { return y < 1e-10; })
			},
			{
				Dirichlet_region(mesh, [](double, double y) { return y > 1 - 1e-10; }) = 0
			},
			1.0
		);
	} catch (const runtime_error&) {
		threw = true;
	}
	if (!threw)
		throw runtime_error("Robin zero beta was not rejected");
}

void test_c_rate_simple()
{
	cout << "\n=== Convergence rate test (simple) ===" << endl;
	double c1 = 0.01;
	double c2 = 0.5;
	double maxR = 1.67;
	double eps = 8.854e-12;
	double theoryAns = pow(maxR, c2 + 2) * c1 / (eps * pow(c2 + 2, 2));
	cout << "Theory ans:" << theoryAns << "\nrelative error:" << endl;
	vector<double> err;

	for (int k = 1; k <= 10; k++)
	{
		tri_mesh mesh = make_rect_trimesh(0.0, 0.0, 1.0, maxR, 6*k, 4*k);
		FEM_Poisson_solver_2d psolver;
		psolver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cylindrical,
			{
				make_neumann(mesh, 0, [](double x, double) { return x > 1 - 1e-10; }),
				make_neumann(mesh, 0, [](double x, double) { return x < 1e-10; }),
				make_neumann(mesh, 0, [](double, double y) { return y < 1e-10; })
			},
			{
				Dirichlet_region(mesh, [maxR](double, double y) { return y > maxR - 1e-10; }) = 0
			},
			eps
		);

		Eigen::VectorXd Phi(mesh.n_nodes()), q_dens(mesh.n_nodes());
		for (size_t i = 0; i < mesh.n_nodes(); i++)
			q_dens(i) = c1 * pow(mesh.node_position(i).y(), c2);
		solve_with_backend(psolver, test_backend, Phi, q_dens);

		double ans = interp_node(mesh, Phi, 1.0, 0.0);

		err.push_back(abs(ans - theoryAns) / theoryAns);
		cout << k << "\t" << "err(phi@(1,0))=" << err.back() << endl;
		write_plt("ans_in_cr1.plt", mesh, Phi);
	}
	double order = log(err.front() / err.back()) / log(10);
	cout << "Convergence order:" << order;
	if (order > 1.7 && order < 2.3) cout << " ~= 2" << endl;
	else cout << " != 2" << endl;
}

void test_c_rate_complex()
{
	cout << "\n=== Convergence rate test (complex) ===" << endl;
	double true_ans;
	vector<double> err;
	for (int k = 20; k > 0; k--)
	{
		if (k < 20 && k > 10) continue;
		tri_mesh mesh = make_rect_trimesh(-1.0, -1.0, 1.0, 1.0, 40*k, 32*k);
		double dx = 2.0 / (40 * k);

		auto robin_w = make_robin(mesh, 1e9,
			[dx](double, double) -> pair<double, double> { return {1, dx}; },
			[](double x, double) { return x < -1.0 + 1e-10; });

		FEM_Poisson_solver_2d psolver;
		psolver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cartesian,
			{
				make_neumann(mesh, 0, [](double, double y) { return y > 1 - 1e-10; }),
				make_neumann(mesh, 0, [](double, double y) { return y < -1 + 1e-10; }),
				make_neumann(mesh, 0, [](double x, double) { return x > 1 - 1e-10; })
			},
			{robin_w},
			{
				Dirichlet_region(mesh, [](double x, double y) { return x > 0.5 - 1e-10 && y > 0.5 - 1e-10; }) = 0,
				Dirichlet_region(mesh, [](double x, double y) { return x < -0.5 + 1e-10 && y < -0.5 + 1e-10; }) = 1e9
			},
			1.0
		);

		Eigen::VectorXd Phi(mesh.n_nodes()), q_dens(mesh.n_nodes());
		for (size_t i = 0; i < mesh.n_nodes(); i++) {
			double x = mesh.node_position(i).x();
			double y = mesh.node_position(i).y();
			q_dens(i) = cos(x * 10) * cos(y * 12);
		}
		solve_with_backend(psolver, test_backend, Phi, q_dens);

		double ans = interp_node(mesh, Phi, 0.0, 0.0);

		if(k==20){
			true_ans = ans;
			cout << "Ground truth:" << true_ans << "\nrelative error:" << endl;
			write_plt("ans_in_cr2.plt", mesh, Phi);
		}
		else
		{
			err.push_back(abs(ans - true_ans) / true_ans);
			cout << k << "\t" << "err(phi@(0,0))=" << err.back() << endl;
		}
	}
	double order = log(err.back() / err.front()) / log(10);
	cout << "Convergence order:" << order;
	if (order > 1.7 && order < 2.3) cout << " ~= 2" << endl;
	else cout << " != 2" << endl;
}

void test_fixedZone_func()
{
	cout << "\n=== Fixed zone function test ===" << endl;
	tri_mesh mesh = make_rect_trimesh(0.0, 0.0, 2.0, 2.0, 100, 200);
	FEM_Poisson_solver_2d psolver1;
	psolver1.init(
		test_backend,
		mesh,
		FEM_Poisson_solver_2d::Cylindrical,
		{
			Dirichlet_region(mesh, [](double z, double r) {
				return ((z - 1) * (z - 1) + r * r) > 0.25;
			}) = 0
		},
		8.854e-12
	);
	Eigen::VectorXd Phi(mesh.n_nodes()), q_dens(mesh.n_nodes());
	for (size_t i = 0; i < mesh.n_nodes(); i++) q_dens(i) = 0.01;
	solve_with_backend(psolver1, test_backend, Phi, q_dens);

	double theoryAns = 0.25*0.01/(6*8.854e-12);
	double solvedAns = interp_node(mesh, Phi, 1.0, 0.0);
	cout << "solved ans:" << solvedAns << ";theory ans:" << theoryAns << endl;
	cout << "err=" << abs(solvedAns - theoryAns) / theoryAns << endl;
	write_plt("ans_fvz_1.plt", mesh, Phi);
	cout << "A sphere." << endl;
	cout << "See in output file." << endl;
}

void test_robin()
{
	cout << "\n=== Robin boundary test ===" << endl;
	for (int k = 0; k < 10; k++)
	{
		tri_mesh mesh = make_rect_trimesh(-1.0, -1.0, 1.0, 1.0, 100, 80);
		double dx = 2.0 / 100;
		double grad_coef = pow((k+0.1),3)*dx;

		auto coeff_fn = [grad_coef](double, double) -> pair<double, double> {
			return {1, grad_coef};
		};

		FEM_Poisson_solver_2d psolver;
		psolver.init(
			test_backend,
			mesh,
			FEM_Poisson_solver_2d::Cartesian,
			{},
			{
				make_robin(mesh, 0, coeff_fn, [](double x, double) { return x > 1 - 1e-10; }),
				make_robin(mesh, 0, coeff_fn, [](double x, double) { return x < -1 + 1e-10; }),
				make_robin(mesh, 0, coeff_fn, [](double, double y) { return y > 1 - 1e-10; }),
				make_robin(mesh, 0, coeff_fn, [](double, double y) { return y < -1 + 1e-10; })
			},
			{
				Dirichlet_region(mesh, [](double x, double y) {
					return x >= -0.4 - 1e-10 && x <= -0.3 + 1e-10 && y >= -0.05 - 1e-10 && y <= 0.05 + 1e-10;
				}) = 10,
				Dirichlet_region(mesh, [](double x, double y) {
					return x >= 0.3 - 1e-10 && x <= 0.4 + 1e-10 && y >= -0.05 - 1e-10 && y <= 0.05 + 1e-10;
				}) = -10
			},
			1.0
		);
		Eigen::VectorXd Phi(mesh.n_nodes()), q_dens(mesh.n_nodes());
		q_dens.setZero();
		solve_with_backend(psolver, test_backend, Phi, q_dens);
		write_plt("ans_robin_" + to_string(k) + ".plt", mesh, Phi);
	}
	cout << "As coeff_grad/coeff_val increases, the potentials gradually 'spread outward'." << endl;
	cout << "See in output file." << endl;
}

void test_backend_manufactured()
{
	cout << "\n=== Backend manufactured solution test ===" << endl;
	string backend_name = "native";

	int N = 120;
	tri_mesh mesh = make_rect_trimesh(0, 0, 1, 1, N, N);

	FEM_Poisson_solver_2d solver_native;
	solver_native.init(
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

	Eigen::VectorXd Phi_native(mesh.n_nodes()), rho(mesh.n_nodes());
	for (size_t i = 0; i < mesh.n_nodes(); i++) {
		double x = mesh.node_position(i).x();
		double y = mesh.node_position(i).y();
		double lap = -2*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y);
		rho(i) = -8.854e-12 * lap;
	}
	solve_with_backend(solver_native, test_backend, Phi_native, rho);

	vector<string> backends = {"cuda_sparselu_gpu"};
	for (auto& bk : backends) {
		cout << "Testing backend: " << bk << endl;
		try {
			FEM_Poisson_solver_2d solver_bk;
			solver_bk.init(
				bk,
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

			Eigen::VectorXd Phi_bk(mesh.n_nodes());
			solve_with_backend(solver_bk, bk, Phi_bk, rho);

			double max_diff = 0;
			for (size_t i = 0; i < mesh.n_nodes(); i++)
				max_diff = max(max_diff, abs(Phi_native(i) - Phi_bk(i)));
			cout << "  max_diff vs native: " << max_diff << endl;
		} catch (const exception& e) {
			cout << "  skipped: " << e.what() << endl;
		}
	}
}

int main() {
	test_boundary_edge_classification();
	cout << endl;
	test_robin_zero_beta_rejected();
	cout << endl;
	test_c_rate_simple();
	cout << endl;
	test_c_rate_complex();
	cout << endl;
	test_fixedZone_func();
	cout << endl;
	test_robin();
	cout << endl;
	test_backend_manufactured();
	cout << endl;
	return 0;
}
