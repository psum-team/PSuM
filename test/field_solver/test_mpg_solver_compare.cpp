#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_2d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include "../../src/field_solver/fem_solver/MPG_Poisson_solver_2d.hpp"
#include "../../src/field_solver/fem_solver/fem_boundary.hpp"
#include "../../src/field_solver/fem_solver/multi_patch_to_tri_mesh.hpp"
#include "../../src/field/multi_patch/node_volume_field.hpp"
#include "../../src/field_solver/fixed_boundary.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field::multi_patch;
using namespace psum::field_solver;
using namespace psum::field_solver::fem_solver;
using namespace fem_boundary_creator;
using namespace boundary_creator;

double theory_phi_origin = 0.25;
const string test_backend = "native";

bool backend_uses_device(const string& backend_name) {
	if (backend_name == "native") return false;
	if (backend_name.size() < 3) return false;
	return backend_name.substr(backend_name.size() - 3) != "cpu";
}

sycl::queue make_queue_for_backend(const string& backend_name) {
	if (backend_uses_device(backend_name)) {
		return sycl::queue{sycl::gpu_selector_v};
	}
	return sycl::queue{sycl::default_selector_v};
}

void solve_with_backend(MPG_Poisson_solver_2d& solver, const string& backend_name, vector<double>& phi, const vector<double>& source) {
	if (!backend_uses_device(backend_name)) {
		solver.solve(phi.data(), source.data());
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

void solve_with_backend(MPG_Poisson_solver_2d& solver, const string& backend_name, node_mp_field2D<double>& phi, node_mp_field2D<double>& source) {
	if (!backend_uses_device(backend_name)) {
		auto source_host = source.getContent().to_host();
		vector<double> phi_host(source_host.size());
		solver.solve(phi_host.data(), source_host.data());
		phi.copy(phi_host);
		return;
	}

	solver.solve(phi.data(), source.data());
}

double nearest_value(const grid2D& grid, const vector<double>& values, double x, double y) {
	return values[grid.n2i(grid.nN(Eigen::RowVector<double, 2>(x, y)))];
}

double nearest_value(const multi_patch_grid<2>& mpg, const vector<double>& values, double x, double y) {
	return values[mpg.n2i(mpg.nN(Eigen::RowVector<double, 2>(x, y)))];
}

void test_circle_dirichlet() {
	cout << "\n=== Circle Dirichlet: MPG vs SG ===" << endl;
	sycl::queue q = make_queue_for_backend(test_backend);

	// ── Step 1: Grid + Field ──
	// SG: simple_grid + device_field
	grid2D grid({-1.5, -1.5}, {1.5, 1.5}, {128, 128});
	node_field2D<double> rho_sg(q, grid);
	node_field2D<double> Phi_sg(q, grid);

	// MPG: 8×8 patches, exp 3~5, boundary-refined around unit circle
	vector<int> res_exp = {
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 4, 4, 4, 4, 3, 3,
		3, 4, 4, 4, 4, 4, 4, 3,
		3, 4, 4, 4, 4, 4, 4, 3,
		3, 4, 4, 4, 4, 4, 4, 3,
		3, 4, 4, 4, 4, 4, 4, 3,
		3, 3, 4, 4, 4, 4, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3
	};
	multi_patch_grid<2> mpg(q, {-1.5, -1.5}, {1.5, 1.5}, {8, 8}, res_exp);
	node_mp_field2D<double> rho_mpg(q, mpg.make_copy());
	node_mp_field2D<double> Phi_mpg(q, mpg.make_copy());

	// ── Step 2: Boundary conditions and solver initialization ──
	// SG
	Poisson_solver_2d sg_solver;
	sg_solver.init(grid, Poisson_solver_2d::Cartesian, 
		{
			Dirichlet_func(grid, [](double x, double y) { return x*x + y*y > 1.0; }) = 0
		}, 1.0
	);
	// MPG
	MPG_Poisson_solver_2d mpg_solver;
	mpg_solver.init(test_backend, mpg, MPG_Poisson_solver_2d::Cartesian, 
		{
			Dirichlet_region(mpg, [](double x, double y) { return x*x + y*y > 1.0; }) = 0
		}, 1.0
	);

	// ── Step 3: Set source ──
	rho_sg.setConstant(1.0);
	rho_mpg.setConstant(1.0);

	// ── Step 4: Solve ──
	vector<double> rho_sg_host = rho_sg.getContent().to_host();
	vector<double> phi_sg_host(rho_sg_host.size());
	sg_solver.solve(phi_sg_host.data(), rho_sg_host.data());

	solve_with_backend(mpg_solver, test_backend, Phi_mpg, rho_mpg);
	vector<double> phi_mpg_host = Phi_mpg.getContent().to_host();

	// ── Step 5: Plot ──
	Phi_sg.copy(phi_sg_host);
	Phi_sg.plot("circle_sg.plt", "phi");
	Phi_mpg.plot("circle_mpg.plt", "phi");

	// ── Step 6: Evaluate at origin via nearest node ──
	double phi_sg_origin = nearest_value(grid, phi_sg_host, 0.0, 0.0);
	double phi_mpg_origin = nearest_value(mpg, phi_mpg_host, 0.0, 0.0);

	cout << fixed << setprecision(6);
	cout << "MPG (FEM): phi(0,0) = " << phi_mpg_origin
	     << "  rel err: " << abs(phi_mpg_origin - theory_phi_origin) / theory_phi_origin
	     << " (DOF = " << mpg_solver.duplicate_map().n_dof << ")" << endl;
	cout << "SG  (FD):  phi(0,0) = " << phi_sg_origin
	     << "  rel err: " << abs(phi_sg_origin - theory_phi_origin) / theory_phi_origin
	     << " (DOF = " << grid.contentSize(var_loc::nodeCentered) << ")" << endl;
	cout << "Theory:    phi(0,0) = " << theory_phi_origin << endl;

	assert(abs(phi_mpg_origin - theory_phi_origin) / theory_phi_origin < 0.05);
	assert(abs(phi_sg_origin - theory_phi_origin) / theory_phi_origin < 0.05);

	cout << "✅ Circle Dirichlet test passed." << endl;
}

void test_circle_refined_boundary() {
	cout << "\n=== Circle Dirichlet (boundary-refined MPG): MPG vs SG ===" << endl;
	sycl::queue q = make_queue_for_backend(test_backend);

	constexpr int n_coarse = 20;
	constexpr double L = 1.5;
	constexpr double patch_span = 2.0 * L / n_coarse;
	constexpr int base_exp = 3;
	constexpr int boundary_exp = 6;

	grid2D grid({-L, -L}, {L, L}, {410, 410});
	node_field2D<double> rho_sg(q, grid);
	node_field2D<double> Phi_sg(q, grid);

	// Build res_exp: refine patches near the circle boundary (x²+y²=1),
	// with gradual transition layers so adjacent patches differ by at most 1
	vector<int> res_exp(n_coarse * n_coarse);
	for (int pi = 0; pi < n_coarse; pi++) {
		for (int pj = 0; pj < n_coarse; pj++) {
			double cx = -L + (pi + 0.5) * patch_span;
			double cy = -L + (pj + 0.5) * patch_span;
			double d = sqrt(cx*cx + cy*cy);
			double dist_to_circle = abs(d - 1.0);
			int layers = (boundary_exp - base_exp);
			double ring_width = patch_span * 1.5;
			int ring = max(0, static_cast<int>(layers - dist_to_circle / ring_width));
			ring = min(ring, layers);
			res_exp[pi * n_coarse + pj] = base_exp + ring;
		}
	}

	multi_patch_grid<2> mpg(q, {-L, -L}, {L, L}, {n_coarse, n_coarse}, res_exp);
	node_mp_field2D<double> rho_mpg(q, mpg.make_copy());
	node_mp_field2D<double> Phi_mpg(q, mpg.make_copy());

	auto boundary_fn = [](double x, double y) { return x*x + y*y > 1.0; };

	Poisson_solver_2d sg_solver;
	sg_solver.init(grid, Poisson_solver_2d::Cartesian,
		{ Dirichlet_func(grid, boundary_fn) = 0 }, 1.0
	);

	MPG_Poisson_solver_2d mpg_solver;
	mpg_solver.init(test_backend, mpg, MPG_Poisson_solver_2d::Cartesian,
		{ Dirichlet_region(mpg, boundary_fn) = 0 }, 1.0
	);

	rho_sg.setConstant(1.0);
	rho_mpg.setConstant(1.0);

	vector<double> rho_sg_host = rho_sg.getContent().to_host();
	vector<double> phi_sg_host(rho_sg_host.size());
	sg_solver.solve(phi_sg_host.data(), rho_sg_host.data());

	solve_with_backend(mpg_solver, test_backend, Phi_mpg, rho_mpg);
	vector<double> phi_mpg_host = Phi_mpg.getContent().to_host();

	Phi_sg.copy(phi_sg_host);
	Phi_sg.plot("circle_bnd_sg.plt", "phi");
	Phi_mpg.plot("circle_bnd_mpg.plt", "phi");

	double phi_sg_origin = nearest_value(grid, phi_sg_host, 0.0, 0.0);
	double phi_mpg_origin = nearest_value(mpg, phi_mpg_host, 0.0, 0.0);

	cout << fixed << setprecision(6);
	cout << "MPG (FEM, boundary-refined): phi(0,0) = " << phi_mpg_origin
	     << "  rel err: " << abs(phi_mpg_origin - theory_phi_origin) / theory_phi_origin
	     << " (DOF = " << mpg_solver.duplicate_map().n_dof << ")" << endl;
	cout << "SG  (FD):                     phi(0,0) = " << phi_sg_origin
	     << "  rel err: " << abs(phi_sg_origin - theory_phi_origin) / theory_phi_origin
	     << " (DOF = " << grid.contentSize(var_loc::nodeCentered) << ")" << endl;
	cout << "Theory:                       phi(0,0) = " << theory_phi_origin << endl;

	assert(abs(phi_mpg_origin - theory_phi_origin) / theory_phi_origin < 0.04);
	assert(abs(phi_sg_origin - theory_phi_origin) / theory_phi_origin < 0.02);

	cout << "✅ Circle Dirichlet (boundary-refined) test passed." << endl;
}

void test_random_source_robin() {
	cout << "\n=== Random Point Source + Robin BC: MPG vs SG ===" << endl;
	sycl::queue q = make_queue_for_backend(test_backend);

	// ── Step 1: Grid + Field ──
	constexpr int N = 384;
	constexpr double L = 3.0;
	double dx = 2.0 * L / N;

	grid2D grid({-L, -L}, {L, L}, {N, N});
	node_field2D<double> rho_sg(q, grid);
	node_field2D<double> Phi_sg(q, grid);
	rho_sg.setZero();

	vector<int> res_exp(144);
	for (int j = 0; j < 12; j++)
		for (int i = 0; i < 12; i++)
			res_exp[j * 12 + i] = 4 + ((i + j) % 2);

	multi_patch_grid<2> mpg(q, {-L, -L}, {L, L}, {12, 12}, res_exp);
	node_mp_field2D<double> rho_mpg(q, mpg.make_copy());
	node_mp_field2D<double> Phi_mpg(q, mpg.make_copy());
	rho_mpg.setZero();

	// ── Step 2: Generate 20 random cluster centers, then sample 1000 Gaussian points ──
	mt19937 rng(42);
	uniform_real_distribution<double> center_dist(-L * 0.6, L * 0.6);
	constexpr int n_centers = 20;
	constexpr int n_pts = 1000;
	constexpr double sigma = 0.15;

	vector<Eigen::RowVector<double, 2>> centers(n_centers);
	for (int k = 0; k < n_centers; k++)
		centers[k] = Eigen::RowVector<double, 2>(center_dist(rng), center_dist(rng));

	normal_distribution<double> gauss(0.0, sigma);
	vector<Eigen::RowVector<double, 2>> pts(n_pts);
	for (int k = 0; k < n_pts; k++) {
		auto& c = centers[k % n_centers];
		pts[k] = Eigen::RowVector<double, 2>(c.x() + gauss(rng), c.y() + gauss(rng));
	}

	device_array<Eigen::RowVector<double, 2>> pts_dev(q, pts);

	// ── Step 3: Deposit onto grids via add_back (linear interpolation) ──
	// SG
	pts_dev.for_each([&](sycl::handler& h) {
		auto acc = rho_sg.get_access(h);
		return [=](const Eigen::RowVector<double, 2>& pos) {
			double w = pos.x() > 0 ? 1.0 : -1.0;
			add_back(pos, w, acc);
		};
	});

	// MPG
	pts_dev.for_each([&](sycl::handler& h) {
		auto acc = rho_mpg.get_access(h);
		return [=](const Eigen::RowVector<double, 2>& pos) {
			double w = pos.x() > 0 ? 1.0 : -1.0;
			add_back(pos, w, acc);
		};
	});

	// MPG: merge duplicates via duplicate_map
	auto dm = build_duplicate_map(mpg);
	auto S = build_merge_matrix(dm);
	auto rho_mpg_host = rho_mpg.getContent().to_host();
	Eigen::VectorXd rho_mpg_vec(dm.n_index);
	for (size_t i = 0; i < dm.n_index; i++)
		rho_mpg_vec(i) = rho_mpg_host[i];
	Eigen::VectorXd merged = S * rho_mpg_vec;
	vector<double> rho_mpg_merged(dm.n_index);
	for (size_t i = 0; i < dm.n_index; i++)
		rho_mpg_merged[i] = merged(i);
	rho_mpg.copy(rho_mpg_merged);

	// ── Step 3.5: Divide by node volume to convert accumulated weight -> density ──
	// SG node volumes
	{
		double dy = dx;
		auto rho_sg_host = rho_sg.getContent().to_host();
		for (int j = 0; j <= N; j++) {
			for (int i = 0; i <= N; i++) {
				size_t idx = grid.n2i({i, j});
				double vol = dx * dy;
				if (i == 0 || i == N) vol *= 0.5;
				if (j == 0 || j == N) vol *= 0.5;
				rho_sg_host[idx] /= vol;
			}
		}
		rho_sg.copy(rho_sg_host);
	}

	// MPG node volumes (use build_node_volume_field, then divide)
	{
		auto vol_field = build_node_volume_field(mpg.make_copy());
		auto vol_host = vol_field.getContent().to_host();
		rho_mpg_host = rho_mpg.getContent().to_host();
		for (size_t i = 0; i < dm.n_index; i++)
			rho_mpg_host[i] /= vol_host[i];
		rho_mpg.copy(rho_mpg_host);
	}

	// ── Step 4: Robin BC on all four walls + solver init ──
	// Robin: c_val * phi + c_grad * d(phi)/dn = 0
	// As c_grad/c_val grows, phi on boundary -> 0 (approaches Dirichlet)
	double robin_val_coeff = 1.0;
	double robin_grad_coeff = dx;

	// SG: Robin_Line on each wall (mixed_boundary_2d)
	auto robin_coeff_fn = [robin_val_coeff, robin_grad_coeff](double, double) -> pair<double, double> {
		return {robin_val_coeff, robin_grad_coeff};
	};
	Poisson_solver_2d sg_solver;
	sg_solver.init(grid, Poisson_solver_2d::Cartesian,
		{
			Robin_Line(grid, boundary_direction_2d::S, robin_coeff_fn) = 0,
			Robin_Line(grid, boundary_direction_2d::N, robin_coeff_fn) = 0,
			Robin_Line(grid, boundary_direction_2d::W, robin_coeff_fn) = 0,
			Robin_Line(grid, boundary_direction_2d::E, robin_coeff_fn) = 0
		},
		{}, 1.0
	);

	// MPG: Robin_region (fem_robin_boundary)
	auto robin_region_fn = [robin_val_coeff, robin_grad_coeff](double, double) -> pair<double, double> {
		return {robin_val_coeff, robin_grad_coeff};
	};
	auto boundary_fn = [L](double x, double y) {
		return abs(x) > L - 1e-10 || abs(y) > L - 1e-10;
	};

	MPG_Poisson_solver_2d mpg_solver;
	mpg_solver.init(test_backend, mpg, MPG_Poisson_solver_2d::Cartesian,
		{}, { Robin_region(mpg, robin_region_fn, boundary_fn) = 0 },
		{}, 1.0
	);

	// ── Step 5: Solve ──
	auto rho_sg_host = rho_sg.getContent().to_host();
	vector<double> phi_sg_host(rho_sg_host.size());
	sg_solver.solve(phi_sg_host.data(), rho_sg_host.data());

	solve_with_backend(mpg_solver, test_backend, Phi_mpg, rho_mpg);
	vector<double> phi_mpg_host = Phi_mpg.getContent().to_host();

	// ── Step 6: Plot ──
	Phi_sg.copy(phi_sg_host);
	Phi_sg.plot("randsrc_robin_sg.plt", "phi");
	Phi_mpg.plot("randsrc_robin_mpg.plt", "phi");

	// ── Step 7: Compare at evaluation points ──
	cout << fixed << setprecision(6);

	vector<Eigen::RowVector<double, 2>> eval_pts = {
		{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, -1.0}, {0.5, -0.5}
	};

	vector<double> sg_h(eval_pts.size()), mpg_h(eval_pts.size());
	for (size_t i = 0; i < eval_pts.size(); i++) {
		sg_h[i] = nearest_value(grid, phi_sg_host, eval_pts[i].x(), eval_pts[i].y());
		mpg_h[i] = nearest_value(mpg, phi_mpg_host, eval_pts[i].x(), eval_pts[i].y());
	}

	double l1 = 0, l2 = 0, lmax = 0;
	for (size_t k = 0; k < eval_pts.size(); k++) {
		double diff = abs(sg_h[k] - mpg_h[k]);
		l1 += diff;
		l2 += diff * diff;
		if (diff > lmax) lmax = diff;
		cout << "  pt " << k << ": SG=" << sg_h[k] << "  MPG=" << mpg_h[k]
		     << "  diff=" << scientific << setprecision(4) << diff << endl;
	}

	l1 /= eval_pts.size();
	l2 = sqrt(l2 / eval_pts.size());
	cout << "L1 avg diff = " << scientific << setprecision(4) << l1 << endl;
	cout << "L2 avg diff = " << l2 << endl;
	cout << "Linf diff   = " << lmax << endl;

	cout << "✅ Random point source + Robin BC test passed." << endl;
}

void test_sphere_dirichlet() {
	cout << "\n=== Sphere Dirichlet (Cylindrical): MPG vs SG ===" << endl;
	sycl::queue q = make_queue_for_backend(test_backend);

	constexpr double R_sphere = 0.5;
	constexpr double eps = 8.854e-12;
	constexpr double rho_val = 0.01;
	double theory_ans = R_sphere * R_sphere * rho_val / (6.0 * eps);

	// Domain: z in [0, 2], r in [0, 2]; sphere centered at (z=1, r=0) radius 0.5
	// Dirichlet = 0 where (z-1)^2 + r^2 > 0.25

	constexpr int n_coarse = 20;
	constexpr double Lz = 2.0, Lr = 2.0;
	constexpr double patch_z = Lz / n_coarse;
	constexpr double patch_r = Lr / n_coarse;
	constexpr int base_exp = 3;
	constexpr int bnd_exp = 5;

	// SG: 200x200 uniform (matches FEM test)
	grid2D grid({0.0, 0.0}, {Lz, Lr}, {200, 200});
	node_field2D<double> rho_sg(q, grid);
	node_field2D<double> Phi_sg(q, grid);

	// MPG: 16x16 patches, refine near sphere boundary
	vector<int> res_exp(n_coarse * n_coarse);
	for (int pi = 0; pi < n_coarse; pi++) {
		for (int pj = 0; pj < n_coarse; pj++) {
			double cz = (pi + 0.5) * patch_z;
			double cr = (pj + 0.5) * patch_r;
			double d = sqrt(pow(cz - 1.0, 2) + cr * cr);
			double dist_to_sphere = abs(d - R_sphere);
			int layers = bnd_exp - base_exp;
			double ring_width = max(patch_z, patch_r) * 1.5;
			int ring = max(0, static_cast<int>(layers - dist_to_sphere / ring_width));
			ring = min(ring, layers);
			res_exp[pi * n_coarse + pj] = base_exp + ring;
		}
	}

	multi_patch_grid<2> mpg(q, {0.0, 0.0}, {Lz, Lr}, {n_coarse, n_coarse}, res_exp);
	node_mp_field2D<double> rho_mpg(q, mpg.make_copy());
	node_mp_field2D<double> Phi_mpg(q, mpg.make_copy());

	// ── Step 2: Boundary + solver init ──
	auto sphere_fn = [R_sphere](double z, double r) {
		return (z - 1.0) * (z - 1.0) + r * r > R_sphere * R_sphere;
	};

	Poisson_solver_2d sg_solver;
	sg_solver.init(grid, Poisson_solver_2d::Cylindrical,
		{ Dirichlet_func(grid, sphere_fn) = 0 }, eps
	);

	MPG_Poisson_solver_2d mpg_solver;
	mpg_solver.init(test_backend, mpg, MPG_Poisson_solver_2d::Cylindrical,
		{ Dirichlet_region(mpg, sphere_fn) = 0 }, eps
	);

	// ── Step 3: Source ──
	rho_sg.setConstant(rho_val);
	rho_mpg.setConstant(rho_val);

	// ── Step 4: Solve ──
	vector<double> rho_sg_host = rho_sg.getContent().to_host();
	vector<double> phi_sg_host(rho_sg_host.size());
	sg_solver.solve(phi_sg_host.data(), rho_sg_host.data());

	solve_with_backend(mpg_solver, test_backend, Phi_mpg, rho_mpg);
	vector<double> phi_mpg_host = Phi_mpg.getContent().to_host();

	// ── Step 5: Plot ──
	Phi_sg.copy(phi_sg_host);
	Phi_sg.plot("sphere_sg.plt", "phi");
	Phi_mpg.plot("sphere_mpg.plt", "phi");

	// ── Step 6: Evaluate at sphere center (z=1, r=0) ──
	double phi_sg_ctr = nearest_value(grid, phi_sg_host, 1.0, 0.0);
	double phi_mpg_ctr = nearest_value(mpg, phi_mpg_host, 1.0, 0.0);

	cout << fixed << setprecision(4);
	cout << "MPG (FEM): phi(1,0) = " << phi_mpg_ctr
	     << "  rel err: " << abs(phi_mpg_ctr - theory_ans) / theory_ans
	     << " (DOF = " << mpg_solver.duplicate_map().n_dof << ")" << endl;
	cout << "SG  (FD):  phi(1,0) = " << phi_sg_ctr
	     << "  rel err: " << abs(phi_sg_ctr - theory_ans) / theory_ans
	     << " (DOF = " << grid.contentSize(var_loc::nodeCentered) << ")" << endl;
	cout << "Theory:    phi(1,0) = " << theory_ans << endl;

	assert(abs(phi_mpg_ctr - theory_ans) / theory_ans < 0.05);
	assert(abs(phi_sg_ctr - theory_ans) / theory_ans < 0.05);

	cout << "✅ Sphere Dirichlet test passed." << endl;
}

int main() {
	test_circle_dirichlet();
	test_circle_refined_boundary();
	test_sphere_dirichlet();
	test_random_source_robin();
	return 0;
}
