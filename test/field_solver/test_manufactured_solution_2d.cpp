// Standalone fast-tier CI target extracted from test_psolver_2d.cpp
// (test_manufactured_solution, unchanged): verifies 2D Poisson convergence
// against a manufactured solution. The full test suite in test_psolver_2d.cpp
// runs as its own (slow-tier) target.
#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_2d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"
#include <iostream>
#include <cmath>
#include <vector>

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace boundary_creator;

int main()
{
    cout << "\n=== Manufactured solution test (2D) ===" << endl;

    vector<double> errL2;

    for(int k=1;k<=6;k++)
    {
        int N = 40*k;

        Poisson_solver_2d solver;
        grid2D g({0,0},{1,1},{N,N});

        solver.init(
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

        host_node_field2D<double> Phi(g);
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
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(),Phi.size());

        solver.solve(phi_vec,rho_vec);

        double L2=0;

        Phi.for_each(
            [&](size_t i,double& v,auto pos){
                double exact=sin(M_PI*pos.x())*sin(M_PI*pos.y());
                double e=abs(v-exact);
                L2+=e*e;
            }
        );

        L2=sqrt(L2/Phi.size());

        errL2.push_back(L2);

        cout<<"N="<<N<<"\tL2="<<L2<<endl;
    }

    double order = log(errL2.front() / errL2.back()) / log(6);
    cout << "Estimated order ~ " << order << endl;

    // Convergence acceptance criteria: the scheme is 2nd order
    // (central differences / linear FEM with lumping), so on this uniform
    // refinement path the estimated order should be near 2 and the error
    // must stay finite, decrease and remain small.
    for (double e : errL2) {
        if (!isfinite(e)) { cerr << "FAIL: non-finite L2 error" << endl; return 1; }
    }
    if (!(errL2.front() > errL2.back())) { cerr << "FAIL: L2 error did not decrease" << endl; return 1; }
    if (errL2.back() >= 1e-3) { cerr << "FAIL: final L2 error too large: " << errL2.back() << endl; return 1; }
    if (!(order > 1.5 && order < 2.5)) { cerr << "FAIL: convergence order " << order << " not ~2" << endl; return 1; }
    cout << "2D convergence test PASSED" << endl;
    return 0;
}
