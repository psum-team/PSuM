#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_2d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace boundary_creator;

void test_fixedZone_convergence()
{
    cout << "\n=== Fixed zone - Native vs Multigrid convergence (Cartesian) ===" << endl;
    cout << setw(12) << "Cycles" << setw(20) << "vs Native Diff" << endl;
    cout << string(32, '-') << endl;

    // Use 2^n grid sizes: 128x128 in Cartesian coordinates
    field::grid2D g({0.0, 0.0}, {2.0, 2.0}, {128, 128});
    int I = g.numCells<0>() + 1;
    int J = g.numCells<1>() + 1;
    int K = 1;

    // Native solver
    Poisson_solver_2d psolver_native;
    psolver_native.init(
        "native",
        g,
        Poisson_solver_2d::Cartesian,
        {
            Dirichlet_func(g, [](double x, double y) { return ((x - 1) * (x - 1) + y * y) > 0.25;}) = 0
        }
    );

    field::host_node_field2D<double> Phi_native(g);
    field::host_node_field2D<double> q_dens(g);
    q_dens.setConstant(0.01);
    Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
    Eigen::Map<Eigen::VectorXd> phi_native_vec(Phi_native.data(), Phi_native.size());
    psolver_native.solve(phi_native_vec, q_dens_vec);

    cout << setw(12) << "Native" << setw(20) << "-" << endl;

    // Test multigrid with increasing cycles
    vector<int> cycles = {1, 2, 5, 10, 20, 50, 100, 150};
    for (int n_cycles : cycles) {
        Poisson_solver_2d psolver_mg;
        psolver_mg.set_options("I " + to_string(I) + " J " + to_string(J) + " K " + to_string(K) + " depth 3 pre_relax 2 post_relax 2 max_cycles " + to_string(n_cycles));
        psolver_mg.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_2d::Cartesian,
            {
                Dirichlet_func(g, [](double x, double y) { return ((x - 1) * (x - 1) + y * y) > 0.25;}) = 0
            }
        );

        field::host_node_field2D<double> Phi_mg(g);
        Eigen::Map<Eigen::VectorXd> phi_mg_vec(Phi_mg.data(), Phi_mg.size());

        psolver_mg.solve(phi_mg_vec, q_dens_vec);

        // Plot multigrid solution
        Phi_mg.plot("mg_fixedzone_cycle" + to_string(n_cycles) + ".plt", "phi");

        // Calculate L2 difference from native solution over entire domain
        double diff_native = (phi_native_vec - phi_mg_vec).norm() / phi_native_vec.norm();

        cout << setw(12) << n_cycles << setw(20) << diff_native << endl;

        if (diff_native < 1e-10) {
            cout << "\nMultigrid converged to native solution at " << n_cycles << " cycles!" << endl;
            break;
        }
    }

    // Plot native solution for comparison
    Phi_native.plot("mg_fixedzone_native.plt", "phi");
}

void test_robin_convergence()
{
    cout << "\n=== Robin boundary - Native vs Multigrid convergence ===" << endl;
    cout << setw(12) << "Cycles" << setw(20) << "vs Native Diff" << endl;
    cout << string(32, '-') << endl;

    const int k = 2;  // Use k=2 for Robin coefficient
    // Use 2^n grid size for multigrid: 128x128
    field::grid2D g({-1.0, -1.0}, {1.0, 1.0}, {128, 128});
    int I = g.numCells<0>() + 1;
    int J = g.numCells<1>() + 1;
    int K_val = 1;

    // Native solver
    Poisson_solver_2d psolver_native;
    psolver_native.init(
        "native",
        g,
        Poisson_solver_2d::Cartesian,
        {
            Robin_Line(g, boundary_direction_2d::E, {1, pow(k,3)*g.del<0>()}) = 0,
            Robin_Line(g, boundary_direction_2d::W, {1, pow(k,3)*g.del<0>()}) = 0,
            Robin_Line(g, boundary_direction_2d::N, {1, pow(k,3)*g.del<0>()}) = 0,
            Robin_Line(g, boundary_direction_2d::S, {1, pow(k,3)*g.del<0>()}) = 0
        },
        {
            Dirichlet_box(g, -0.4, -0.05, 0.-0.3, 0.05) = 10,
            Dirichlet_box(g, 0.3, -0.05, 0.4, 0.05) = -10
        }
    );

    field::host_node_field2D<double> Phi_native(g);
    field::host_node_field2D<double> q_dens(g);
    // Add small source term: rho = 0.001 * sin(pi*x) * cos(pi*y)
    // This should be enough to see non-trivial solution but not dominate
    q_dens.for_each(
        [](size_t i, double& v, host_node_field2D<double>::Position pos) {
            v = 0.001 * sin(M_PI * pos.x()) * cos(M_PI * pos.y());
        }
    );
    Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
    Eigen::Map<Eigen::VectorXd> phi_native_vec(Phi_native.data(), Phi_native.size());
    psolver_native.solve(phi_native_vec, q_dens_vec);

    // Find min and max of native solution
    double min_phi = phi_native_vec.minCoeff();
    double max_phi = phi_native_vec.maxCoeff();

    cout << setw(12) << "Native" << setw(20) << "-" << endl;
    cout << setw(12) << "" << setw(20) << "min(phi) = " << min_phi << endl;
    cout << setw(12) << "" << setw(20) << "max(phi) = " << max_phi << endl;

    // Test multigrid with increasing cycles
    vector<int> cycles = {1, 2, 5, 10, 20, 50, 100, 150};
    for (int n_cycles : cycles) {
        Poisson_solver_2d psolver_mg;
        // Try with more relaxation: pre_relax=4, post_relax=4
        psolver_mg.set_options("I " + to_string(I) + " J " + to_string(J) + " K " + to_string(K_val) + " depth 3 pre_relax 4 post_relax 4 max_cycles " + to_string(n_cycles));
        psolver_mg.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_2d::Cartesian,
            {
                Robin_Line(g, boundary_direction_2d::E, {1, pow(k,3)*g.del<0>()}) = 0,
                Robin_Line(g, boundary_direction_2d::W, {1, pow(k,3)*g.del<0>()}) = 0,
                Robin_Line(g, boundary_direction_2d::N, {1, pow(k,3)*g.del<0>()}) = 0,
                Robin_Line(g, boundary_direction_2d::S, {1, pow(k,3)*g.del<0>()}) = 0
            },
            {
                Dirichlet_box(g, -0.4, -0.05, 0.-0.3, 0.05) = 10,
                Dirichlet_box(g, 0.3, -0.05, 0.4, 0.05) = -10
            }
        );

        field::host_node_field2D<double> Phi_mg(g);
        Eigen::Map<Eigen::VectorXd> phi_mg_vec(Phi_mg.data(), Phi_mg.size());

        psolver_mg.solve(phi_mg_vec, q_dens_vec);

        // Plot multigrid solution
        Phi_mg.plot("mg_robin_cycle" + to_string(n_cycles) + ".plt", "phi");

        // Calculate L2 difference from native solution over entire domain
        double diff_native = (phi_native_vec - phi_mg_vec).norm() / phi_native_vec.norm();

        cout << setw(12) << n_cycles << setw(20) << diff_native << endl;

        if (diff_native < 1e-10) {
            cout << "\nMultigrid converged to native solution at " << n_cycles << " cycles!" << endl;
            break;
        }
    }

    // Plot native solution for comparison
    Phi_native.plot("mg_robin_native.plt", "phi");
}

void test_mixed_boundary_convergence()
{
    cout << "\n=== Mixed boundary (Neumann + non-zero Dirichlet) - Native vs Multigrid (Cartesian) ===" << endl;
    cout << setw(12) << "Cycles" << setw(20) << "vs Native Diff" << endl;
    cout << string(32, '-') << endl;

    double c1 = 0.01;
    double c2 = 0.5;

    // Use 2^n grid sizes: 64x64 in Cartesian coordinates
    field::grid2D g({0.0, 0.0}, {1.0, 1.0}, {64, 64});
    int I = g.numCells<0>() + 1;
    int J = g.numCells<1>() + 1;
    int K = 1;

    // Native solver
    Poisson_solver_2d psolver_native;
    psolver_native.init(
        "native",
        g,
        Poisson_solver_2d::Cartesian,
        {
            Neumann_line(g, boundary_direction_2d::E) = 0,
            Neumann_line(g, boundary_direction_2d::W) = 0,
            Neumann_line(g, boundary_direction_2d::S) = 0
        },
        {
            Dirichlet_line(g, boundary_direction_2d::N) = 0
        }
    );

    field::host_node_field2D<double> Phi_native(g);
    field::host_node_field2D<double> q_dens(g);
    q_dens.for_each(
        [=](size_t i, double& v, host_node_field2D<double>::Position pos) {
            v = c1 * pow(pos.y(), c2);
        }
    );

    Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
    Eigen::Map<Eigen::VectorXd> phi_native_vec(Phi_native.data(), Phi_native.size());
    psolver_native.solve(phi_native_vec, q_dens_vec);

    cout << setw(12) << "Native" << setw(20) << "-" << endl;

    // Test multigrid with increasing cycles
    vector<int> cycles = {1, 2, 5, 10, 20, 50, 100, 150};
    for (int n_cycles : cycles) {
        Poisson_solver_2d psolver_mg;
        // Try with more relaxation: pre_relax=4, post_relax=4
        psolver_mg.set_options("I " + to_string(I) + " J " + to_string(J) + " K " + to_string(K) + " depth 3 pre_relax 4 post_relax 4 max_cycles " + to_string(n_cycles));
        psolver_mg.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_2d::Cartesian,
            {
                Neumann_line(g, boundary_direction_2d::E) = 0,
                Neumann_line(g, boundary_direction_2d::W) = 0,
                Neumann_line(g, boundary_direction_2d::S) = 0
            },
            {
                Dirichlet_line(g, boundary_direction_2d::N) = 0
            }
        );

        field::host_node_field2D<double> Phi_mg(g);
        Eigen::Map<Eigen::VectorXd> phi_mg_vec(Phi_mg.data(), Phi_mg.size());

        psolver_mg.solve(phi_mg_vec, q_dens_vec);

        // Plot multigrid solution
        Phi_mg.plot("mg_mixed_cycle" + to_string(n_cycles) + ".plt", "phi");

        // Calculate L2 difference from native solution over entire domain
        double diff_native = (phi_native_vec - phi_mg_vec).norm() / phi_native_vec.norm();

        cout << setw(12) << n_cycles << setw(20) << diff_native << endl;

        if (diff_native < 1e-10) {
            cout << "\nMultigrid converged to native solution at " << n_cycles << " cycles!" << endl;
            break;
        }
    }

    // Plot native solution for comparison
    Phi_native.plot("mg_mixed_native.plt", "phi");
}

int main()
{
    cout << "\n========================================" << std::endl;
    cout << "Native vs Multigrid Convergence Tests" << std::endl;
    cout << "========================================" << std::endl;

    test_fixedZone_convergence();
    test_robin_convergence();
    test_mixed_boundary_convergence();

    cout << "\n========================================" << std::endl;
    cout << "All convergence tests completed!" << std::endl;
    cout << "========================================" << std::endl;

    return 0;
}
