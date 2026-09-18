#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_3d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include "../../src/field_solver/boundary_creator/Neumann.hpp"
#include "../../src/field_solver/boundary_creator/Robin.hpp"

using namespace std;
using namespace psum;
using namespace psum::field;
using namespace psum::field_solver;
using namespace boundary_creator;

void test_manufactured_solution()
{
    cout << "\n=== Manufactured solution test (3D) - Multigrid backend ===" << endl;

    vector<double> errL2;

    for(int k=1;k<=5;k++)
    {
        int N = 20*k;

        Poisson_solver_3d solver;
        grid3D g({0,0,0},{1,1,1},{N,N,N});

        int I = g.numCells<0>() + 1;
        int J = g.numCells<1>() + 1;
        int K = g.numCells<2>() + 1;
        int depth = 0;
        int n = I;
        if (J > n) n = J;
        if (K > n) n = K;
        while ((1 << depth) < n) depth++;
        if (depth > 4) depth = 4;

        char options[256];
        snprintf(options, sizeof(options), "I %d J %d K %d depth %d", I, J, K, depth);
        solver.set_options(options);

        solver.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_3d::Cartesian,
            {},
            {
                Dirichlet_plane(g,boundary_direction_3d::Ypos)=0,
                Dirichlet_plane(g,boundary_direction_3d::Yneg)=0,
                Dirichlet_plane(g,boundary_direction_3d::Xpos)=0,
                Dirichlet_plane(g,boundary_direction_3d::Xneg)=0,
                Dirichlet_plane(g,boundary_direction_3d::Zpos)=0,
                Dirichlet_plane(g,boundary_direction_3d::Zneg)=0
            }
        );

        host_node_field3D<double> Phi(g);
        host_node_field3D<double> rho(g);

        rho.for_each(
            [](size_t i,double& v,host_node_field3D<double>::Position pos){
                double x=pos.x();
                double y=pos.y();
                double z=pos.z();
                double lap = -3*M_PI*M_PI*sin(M_PI*x)*sin(M_PI*y)*sin(M_PI*z);
                v = -8.854e-12 * lap;
            }
        );

        Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(),rho.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(),Phi.size());

        solver.solve(phi_vec,rho_vec);

        double L2=0;

        Phi.for_each(
            [&](size_t i,double& v,host_node_field3D<double>::Position pos){
                double exact=sin(M_PI*pos.x())*sin(M_PI*pos.y())*sin(M_PI*pos.z());
                double e=abs(v-exact);
                L2+=e*e;
            }
        );

        L2=sqrt(L2/Phi.size());

        errL2.push_back(L2);

        cout<<"N="<<N<<"\tL2="<<L2<<endl;
    }

    double order = log(errL2.front() / errL2.back()) / log(errL2.size());
    cout << "Estimated order ~ " << order << endl;
}

void test_fixedZone_func()
{
    cout << "\n=== Fixed zone function test (3D) - Multigrid backend ===" << endl;

    vector<double> err;
    double radius = 0.5;
    double q_0 = 0.01;
    double theoryAns = radius * radius * q_0 / (6 * 8.854e-12);

    cout << "theory ans:" << theoryAns << endl;

    for (int k = 1; k <= 4; k++) {
        Poisson_solver_3d solver;
        grid3D g({0.0, 0.0, 0.0}, {2.0, 2.0, 2.0}, {40*k, 40*k, 40*k});

        int I = g.numCells<0>() + 1;
        int J = g.numCells<1>() + 1;
        int K = g.numCells<2>() + 1;
        int depth = 0;
        int n = I;
        if (J > n) n = J;
        if (K > n) n = K;
        while ((1 << depth) < n) depth++;
        if (depth > 4) depth = 4;

        char options[256];
        snprintf(options, sizeof(options), "I %d J %d K %d depth %d", I, J, K, depth);
        solver.set_options(options);

        solver.init(
            "eigen_multigrid_cpu",
            g,
            Poisson_solver_3d::Cartesian,
            {
                Dirichlet_func(g, [=](double x, double y, double z) { 
                    return ((x - 1.0) * (x - 1.0) + (y - 1.0) * (y - 1.0) + (z - 1.0) * (z - 1.0)) > radius * radius; 
                }) = 0
            }
        );
        host_node_field3D<double> Phi(g);
        host_node_field3D<double> q_dens(g);
        q_dens.setConstant(q_0);
        Eigen::Map<Eigen::VectorXd> q_dens_vec(q_dens.data(), q_dens.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
        solver.solve(phi_vec, q_dens_vec);

        Eigen::RowVector3d pos{1.0, 1.0, 1.0};
        double solvedAns = interp(pos, Phi);
        cout << "N=" << g.numCells<0>() << "\terr(phi@(1,1,1))=" << abs(solvedAns - theoryAns) / theoryAns << endl;
        err.push_back(abs(solvedAns - theoryAns) / theoryAns);
        if (k==2) Phi.plot("ans_fvz_3d_multigrid.plt", "phi");
    }
    double order = log(err.front() / err.back()) / log(err.size());
    cout << "Estimated order ~ " << order << endl;
    cout << "A sphere." << endl;
    cout << "See in output file." << endl;
}

int main() {
    test_manufactured_solution();
    cout << endl;
    test_fixedZone_func();
    cout << endl;
    return 0;
}
