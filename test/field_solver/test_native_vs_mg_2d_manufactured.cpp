// Fast-tier smoke derived from test_manufactured_solution_convergence() in
// test_native_vs_multigrid_2d.cpp: identical solver setup on a 64x64 grid,
// but with a truncated MG cycle sweep (full sweep lives in the original
// slow-tier target). Verifies that multigrid converges towards the native
// direct solution.
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

int main()
{
    cout << "\n=== Smoke: Manufactured solution - Native vs Multigrid (reduced sweep) ===" << endl;
    cout << setw(12) << "Cycles" << setw(16) << "L2 Error" << setw(16) << "vs Native Err" << endl;
    cout << string(44, '-') << endl;

    const int N = 64;  // Use 2^n grid size for multigrid
    grid2D g({0,0},{1,1},{N,N});

    int I = g.numCells<0>() + 1;
    int J = g.numCells<1>() + 1;
    int K = 1;

    // Native solver (direct method - ground truth)
    Poisson_solver_2d solver_native;
    solver_native.init(
        "native",
        g,
        Poisson_solver_2d::Cartesian,
        {},
        {
            Dirichlet_line(g,boundary_direction_2d::N)=0,
            Dirichlet_line(g,boundary_direction_2d::S)=0,
            Dirichlet_line(g,boundary_direction_2d::E)=0,
            Dirichlet_line(g,boundary_direction_2d::W)=0
        }
    );

    host_node_field2D<double> Phi_native(g);
    host_node_field2D<double> rho(g);

    rho.for_each(
        [](size_t i,double& v,auto pos){
            double x=pos.x();
            double y=pos.y();
            double lap = -2*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y);
            v = -8.854e-12 * lap;
        }
    );

    Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(),rho.size());
    Eigen::Map<Eigen::VectorXd> phi_native_vec(Phi_native.data(),Phi_native.size());

    solver_native.solve(phi_native_vec, rho_vec);

    double L2_native = 0;
    Phi_native.for_each(
        [&](size_t i,double& v,auto pos){
            double exact=sin(M_PI*pos.x())*sin(M_PI*pos.y());
            double e=abs(v-exact);
            L2_native+=e*e;
        }
    );
    L2_native = sqrt(L2_native/Phi_native.size());

    cout << setw(12) << "Native" << setw(16) << L2_native << setw(16) << "-" << endl;

    // Reduced sweep: smoke tier keeps the run well below the CI timeout
    // (full sweep up to 1e-10 agreement lives in the slow-tier target).
    vector<int> cycles = {1, 2, 5};
    bool converged = false;
    for (int n_cycles : cycles) {
        Poisson_solver_2d solver_mg;
        solver_mg.set_options("I " + to_string(I) + " J " + to_string(J) + " K " + to_string(K) + " depth 3 pre_relax 2 post_relax 2 max_cycles " + to_string(n_cycles));
        solver_mg.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_2d::Cartesian,
            {},
            {
                Dirichlet_line(g,boundary_direction_2d::N)=0,
                Dirichlet_line(g,boundary_direction_2d::S)=0,
                Dirichlet_line(g,boundary_direction_2d::E)=0,
                Dirichlet_line(g,boundary_direction_2d::W)=0
            }
        );

        host_node_field2D<double> Phi_mg(g);
        Eigen::Map<Eigen::VectorXd> phi_mg_vec(Phi_mg.data(),Phi_mg.size());

        solver_mg.solve(phi_mg_vec, rho_vec);

        double L2_mg = 0;
        Phi_mg.for_each(
            [&](size_t i,double& v,auto pos){
                double exact=sin(M_PI*pos.x())*sin(M_PI*pos.y());
                double e=abs(v-exact);
                L2_mg+=e*e;
            }
        );
        L2_mg = sqrt(L2_mg/Phi_mg.size());

        double diff_native = (phi_native_vec - phi_mg_vec).norm() / phi_native_vec.norm();

        cout << setw(12) << n_cycles << setw(16) << L2_mg << setw(16) << diff_native << endl;

        if (diff_native < 1e-4) {
            cout << "\nMultigrid within 1e-4 of native solution at " << n_cycles << " cycles!" << endl;
            converged = true;
            break;
        }
    }

    if (!converged) {
        cerr << "Smoke FAILED: multigrid not within 1e-4 of native solution within " << cycles.back() << " cycles" << endl;
        return 1;
    }
    return 0;
}
