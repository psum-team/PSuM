#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_2d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include "../../src/random/rander.hpp"
#include "../../src/utils_sycl.hpp"

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace psum::random;
using namespace boundary_creator;

void test_c_rate_simple()
{
	cout << "\n=== Convergence rate test (simple) ===" << endl;
	// q(z,r) = c1*r^c2
	double c1 = 0.01;
	double c2 = 0.5;
	double maxR = 1.67;
	double theoryAns = pow(maxR, c2 + 2) * c1 / (8.854e-12 * pow(c2 + 2, 2));
	cout << "Theory ans:" << theoryAns << "\nrelative error:" << endl;
	vector<double> err;

	for (int k = 1; k <= 10; k++)
	{
		Poisson_solver_2d psolver;
		grid2D g({0.0, 0.0}, {1.0, maxR}, {60*k, 40*k});
		psolver.init(
			g,
			Poisson_solver_2d::Cylindrical,
			{
				Neumann_line(g, boundary_direction_2d::E) = 0,
				Neumann_line(g, boundary_direction_2d::W) = 0,
				Neumann_line(g, boundary_direction_2d::S) = 0
			},
			{
				Dirichlet_line(g, boundary_direction_2d::N) = 0
			}
		);

        host_node_field2D<double> Phi(g);
        host_node_field2D<double> q_dens(g);
        q_dens.for_each(
            [=](size_t i, double& v, host_cell_field2D<double>::Position pos) {
                v = c1 * pow(pos.y(), c2);
            }
        );

        Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
		psolver.solve(phi_vec, q_dens_vec);

        double ans = interp(field::node_field2D<double>::Position{1.0, 0.0}, Phi);

		err.push_back(abs(ans - theoryAns) / theoryAns);
		cout << k << "\t" << "err(phi@(1,0))=" << err.back() << endl;
		Phi.plot("ans_in_cr1.plt", "phi");
    }
	double order = log(err.front() / err.back()) / log(10);
	cout << "Convergence order:" << order;
	if (order > 1.7 && order < 2.3) cout << " ~= 2" << endl;
	else cout << " != 2" << endl;
}

void test_c_rate_complex()
{
	cout << "\n=== Convergence rate test (complex) ===" << endl;
	double true_ans; // computed in k=20
	vector<double> err;
	for (int k = 20; k > 0; k--)
	{
		if (k < 20 && k > 10) continue;
		Poisson_solver_2d psolver;
		grid2D g({-1.0, -1.0}, {1.0, 1.0}, {40*k, 32*k});
		psolver.init(
			g,
			Poisson_solver_2d::Cartesian,
			{
				Neumann_line(g, boundary_direction_2d::N) = 0,
				Neumann_line(g, boundary_direction_2d::S) = 0,
				Neumann_line(g, boundary_direction_2d::E) = 0,
				Robin_Line(g, boundary_direction_2d::W, {1, 1}) = 1e9
			},
			{
				Dirichlet_box(g, 0.5, 0.5, 1, 1) = 0,
                Dirichlet_box(g, -1, -1, -0.5, -0.5) = 1e9,
			}
		);

        host_node_field2D<double> Phi(g);
        host_node_field2D<double> q_dens(g);
        q_dens.for_each(
            [=](size_t i, double& v, host_cell_field2D<double>::Position pos) {
                v = cos(pos.x() * 10) * cos(pos.y() * 12);
            }
        );

        Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
		psolver.solve(phi_vec, q_dens_vec);

        double ans = interp(field::node_field2D<double>::Position{0.0, 0.0}, Phi);

		if(k==20){
			true_ans = ans;
			cout << "Ground truth:" << true_ans << "\nrelative error:" << endl;
			Phi.plot("ans_in_cr2.plt", "phi");
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
	Poisson_solver_2d psolver1;
	field::grid2D g({0.0, 0.0}, {2.0, 2.0}, {100, 200});
	psolver1.init(
		g,
		Poisson_solver_2d::Cylindrical,
		{
			Dirichlet_func(g, [](double z, double r) { return ((z - 1) * (z - 1) + r * r) > 0.25;}) = 0
		}
	);
    field::host_node_field2D<double> Phi(g);
    field::host_node_field2D<double> q_dens(g);
    q_dens.setConstant(0.01);
    Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
    psolver1.solve(phi_vec, q_dens_vec);

	double theoryAns = 0.25*0.01/(6*8.854e-12);
    double solvedAns = interp(field::node_field2D<double>::Position{1.0, 0.0}, Phi);
    cout << "solved ans:" << solvedAns << ";theory ans:" << theoryAns << endl;
    cout << "err=" << abs(solvedAns - theoryAns) / theoryAns << endl;
	Phi.plot("ans_fvz_1.plt", "phi");
	cout << "A shpere." << endl;
	cout << "See in output file." << endl;
}

void test_robin()
{
	cout << "\n=== Robin boundary test ===" << endl;
	for (int k = 0; k < 10; k++)
	{
        Poisson_solver_2d psolver;
        field::grid2D g({-1.0, -1.0}, {1.0, 1.0}, {100, 80});
        psolver.init(
			g,
			Poisson_solver_2d::Cartesian,
			{
				Robin_Line(g, boundary_direction_2d::E, {1, pow((k+0.1),3)*g.del<0>()}) = 0,
				Robin_Line(g, boundary_direction_2d::W, {1, pow((k+0.1),3)*g.del<0>()}) = 0,
				Robin_Line(g, boundary_direction_2d::N, {1, pow((k+0.1),3)*g.del<0>()}) = 0,
				Robin_Line(g, boundary_direction_2d::S, {1, pow((k+0.1),3)*g.del<0>()}) = 0
			},
			{
				Dirichlet_box(g, -0.4, -0.05, 0.-0.3, 0.05) = 10,
				Dirichlet_box(g, 0.3, -0.05, 0.4, 0.05) = -10
			}
		);
        field::host_node_field2D<double> Phi(g);
        field::host_node_field2D<double> q_dens(g);
        q_dens.setConstant(0);
        Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
        psolver.solve(phi_vec, q_dens_vec);
        Phi.plot("ans_robin_" + to_string(k) + ".plt", "phi");
	}
	cout << "As coeff_grad/coeff_val increases, the potentials gradually 'spread outward'." << endl;
	cout << "See in output file." << endl;
}

void test_periodic()
{
	cout << "\n=== Periodic boundary test ===" << endl;
    Poisson_solver_2d psolver;
    field::grid2D g({0.0, 0.0}, {2.0, 2.0}, {100, 100});
    psolver.init(
		g,
		Poisson_solver_2d::Cartesian,
		{},
		{
			Dirichlet_box(g, 0.2, 0.9, 1.0, 1.1) = 10,
			Dirichlet_box(g, 1.3, 0.6, 1.5, 1.4) = -10
		}
	);
    field::host_node_field2D<double> Phi(g);
    field::host_node_field2D<double> q_dens(g);
    q_dens.setConstant(0);
    Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
    psolver.solve(phi_vec, q_dens_vec);
    Phi.plot("ans_periodic_tilex1.plt", "phi");

    Poisson_solver_2d psolver2;
    field::grid2D g2({0.0, 0.0}, {4.0, 4.0}, {200, 200});
    psolver2.init(
		g2,
		Poisson_solver_2d::Cartesian,
		{},
		{
			Dirichlet_box(g2, 0.2, 0.9, 1.0, 1.1) = 10,
			Dirichlet_box(g2, 1.3, 0.6, 1.5, 1.4) = -10,

			Dirichlet_box(g2, 2.2, 0.9, 3.0, 1.1) = 10,
			Dirichlet_box(g2, 3.3, 0.6, 3.5, 1.4) = -10,

			Dirichlet_box(g2, 0.2, 2.9, 1.0, 3.1) = 10,
			Dirichlet_box(g2, 1.3, 2.6, 1.5, 3.4) = -10,
            
			Dirichlet_box(g2, 2.2, 2.9, 3.0, 3.1) = 10,
			Dirichlet_box(g2, 3.3, 2.6, 3.5, 3.4) = -10,
		}
	);
    field::host_node_field2D<double> Phi2(g2);
    field::host_node_field2D<double> q_dens2(g2);
    q_dens2.setConstant(0);
    Eigen::Map<Eigen::VectorXd> q_dens_vec2(q_dens2.data(), q_dens2.size());
    Eigen::Map<Eigen::VectorXd> phi_vec2(Phi2.data(), Phi2.size());
    psolver2.solve(phi_vec2, q_dens_vec2);
    Phi2.plot("ans_periodic_tilex4.plt", "phi");


	cout << "Periodic boundary condition." << endl;
	cout << "See in output file." << endl;
}


int main() {
	test_c_rate_simple();
	cout << endl;
    test_c_rate_complex();
    cout << endl;
    test_fixedZone_func();
    cout << endl;
    test_robin();
    cout << endl;
    test_periodic();
    cout << endl;
	return 0;
}