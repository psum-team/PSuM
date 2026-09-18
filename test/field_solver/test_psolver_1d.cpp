#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_1d.hpp"
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

void test_manufactured_solution()
{
    cout << "\n=== Manufactured solution test (1D) ===" << endl;

    vector<double> errL2;

    for(int k=1;k<=6;k++)
    {
        int N = 40*k;

        Poisson_solver_1d solver;
        grid1D g({0},{1},{N});

        solver.init(
            g,
            {},
            {
                Dirichlet_point(g,boundary_direction_1d::L)=0,
                Dirichlet_point(g,boundary_direction_1d::R)=0
            }
        );

        host_node_field1D<double> Phi(g);
        host_node_field1D<double> rho(g);

        rho.for_each(
            [](size_t i,double& v,auto pos){
                double x=pos.x();
                double lap = -M_PI*M_PI*sin(M_PI*x);
                v = -8.854e-12 * lap;
            }
        );

        Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(),rho.size());
        Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(),Phi.size());

        solver.solve(phi_vec,rho_vec);

        double L2=0;

        Phi.for_each(
            [&](size_t i,double& v,auto pos){
                double exact=sin(M_PI*pos.x());
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
}

void test_simple_case()
{
    cout << "\n=== Simple case test (1D) ===" << endl;
    
    Poisson_solver_1d solver;
    grid1D g({0}, {1}, {100});
    
    solver.init(
        g,
        {},
        {
            Dirichlet_point(g, boundary_direction_1d::L) = 0,
            Dirichlet_point(g, boundary_direction_1d::R) = 1
        }
    );
    
    host_node_field1D<double> Phi(g);
    host_node_field1D<double> rho(g);
    rho.setConstant(0);
    
    Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(), rho.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
    
    solver.solve(phi_vec, rho_vec);
    
    double max_err = 0;
    Phi.for_each(
        [&](size_t i, double& v, auto pos) {
            double exact = pos.x();
            double err = abs(v - exact);
            max_err = max(max_err, err);
        }
    );
    
    cout << "Max error: " << max_err << endl;
    
    if (max_err < 1e-10) {
        cout << "✅ PASSED" << endl;
    } else {
        cout << "❌ FAILED" << endl;
    }
}

void test_neumann_boundary()
{
    cout << "\n=== Neumann boundary test (1D) ===" << endl;
    cout << "Theory: φ(x) = x²" << endl;
    cout << "∇²φ = 2, so ρ = -2ε₀" << endl;
    cout << "Boundary: φ'(0) = 0 (Neumann), φ(1) = 1 (Dirichlet)" << endl;
    
    Poisson_solver_1d solver;
    grid1D g({0}, {1}, {100});
    
    solver.init(
        g,
        {
            Neumann_point(g, boundary_direction_1d::L) = 0
        },
        {
            Dirichlet_point(g, boundary_direction_1d::R) = 1
        }
    );
    
    host_node_field1D<double> Phi(g);
    host_node_field1D<double> rho(g);
    rho.setConstant(-8.854e-12 * 2.0);
    
    Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(), rho.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
    
    solver.solve(phi_vec, rho_vec);
    
    double max_err = 0;
    Phi.for_each(
        [&](size_t i, double& v, auto pos) {
            double x = pos.x();
            double exact = x*x;
            double err = abs(v - exact);
            max_err = max(max_err, err);
        }
    );
    
    cout << "Max error: " << max_err << endl;
    
    if (max_err < 1e-8) {
        cout << "✅ PASSED" << endl;
    } else {
        cout << "❌ FAILED" << endl;
    }
}

void test_robin_boundary()
{
    cout << "\n=== Robin boundary test (1D) ===" << endl;
    
    cout << "\n--- Test A: Fixed coefficients (φ(x) = x²) ---" << endl;
    cout << "Robin BC: φ + φ' = value" << endl;
    cout << "At x=0: 0 + 0 = 0" << endl;
    cout << "At x=1: 1 + 2 = 3" << endl;
    cout << "Poisson eq: ∇²φ = 2, so ρ = -2ε₀" << endl;
    
    Poisson_solver_1d solver1;
    grid1D g1({0}, {1}, {100});
    
    solver1.init(
        g1,
        {
            Robin_point(g1, boundary_direction_1d::L, {1.0, 1.0}) = 0.0,
            Robin_point(g1, boundary_direction_1d::R, {1.0, 1.0}) = 3.0
        },
        {}
    );
    
    host_node_field1D<double> Phi1(g1);
    host_node_field1D<double> rho1(g1);
    rho1.setConstant(-8.854e-12 * 2.0);
    
    Eigen::Map<Eigen::VectorXd> rho1_vec(rho1.data(), rho1.size());
    Eigen::Map<Eigen::VectorXd> phi1_vec(Phi1.data(), Phi1.size());
    
    solver1.solve(phi1_vec, rho1_vec);
    
    double max_err1 = 0;
    Phi1.for_each(
        [&](size_t i, double& v, auto pos) {
            double x = pos.x();
            double exact = x*x;
            double err = abs(v - exact);
            max_err1 = max(max_err1, err);
        }
    );
    
    cout << "Max error (Test A): " << max_err1 << endl;
    if (max_err1 < 1e-8) {
        cout << "✅ Test A PASSED" << endl;
    } else {
        cout << "❌ Test A FAILED" << endl;
    }
    
    cout << "\n--- Test B: Function coefficients (φ(x) = x²) ---" << endl;
    cout << "Robin BC: a(x)*φ + b(x)*φ' = value" << endl;
    cout << "At x=0: 1*0 + 1*0 = 0" << endl;
    cout << "At x=1: 1*1 + 2*2 = 5" << endl;
    cout << "Poisson eq: ∇²φ = 2, so ρ = -2ε₀" << endl;
    
    Poisson_solver_1d solver2;
    grid1D g2({0}, {1}, {100});
    
    auto coef_func_L = [](double x) -> std::pair<double, double> {
        return {1.0, 1.0};
    };
    
    auto coef_func_R = [](double x) -> std::pair<double, double> {
        return {1.0, 2.0};
    };
    
    solver2.init(
        g2,
        {
            Robin_point(g2, boundary_direction_1d::L, coef_func_L) = 0.0,
            Robin_point(g2, boundary_direction_1d::R, coef_func_R) = 5.0
        },
        {}
    );
    
    host_node_field1D<double> Phi2(g2);
    host_node_field1D<double> rho2(g2);
    rho2.setConstant(-8.854e-12 * 2.0);
    
    Eigen::Map<Eigen::VectorXd> rho2_vec(rho2.data(), rho2.size());
    Eigen::Map<Eigen::VectorXd> phi2_vec(Phi2.data(), Phi2.size());
    
    solver2.solve(phi2_vec, rho2_vec);
    
    double max_err2 = 0;
    Phi2.for_each(
        [&](size_t i, double& v, auto pos) {
            double x = pos.x();
            double exact = x*x;
            double err = abs(v - exact);
            max_err2 = max(max_err2, err);
        }
    );
    
    cout << "Max error (Test B): " << max_err2 << endl;
    if (max_err2 < 1e-8) {
        cout << "✅ Test B PASSED" << endl;
    } else {
        cout << "❌ Test B FAILED" << endl;
    }
}

void test_variable_epsilon()
{
    cout << "\n=== Variable epsilon test (1D) ===" << endl;
    cout << "Theory: φ(x) = x²" << endl;
    cout << "ε(x) = ε₀(1+x)" << endl;
    cout << "Poisson eq: ∇·(ε∇φ) = -ρ" << endl;
    cout << "∇·(ε∇φ) = d/dx[ε(x)·dφ/dx] = d/dx[ε₀(1+x)·2x] = d/dx[ε₀(2x+2x²)]" << endl;
    cout << "         = ε₀(2+4x)" << endl;
    cout << "So: ρ = -ε₀(2+4x)" << endl;
    
    Poisson_solver_1d solver;
    grid1D g({0}, {1}, {100});
    
    auto epsilon_func = [](double x) {
        return 8.854e-12 * (1.0 + x);
    };
    
    solver.init(
        g,
        {},
        {
            Dirichlet_point(g, boundary_direction_1d::L) = 0,
            Dirichlet_point(g, boundary_direction_1d::R) = 1
        },
        epsilon_func
    );
    
    host_node_field1D<double> Phi(g);
    host_node_field1D<double> rho(g);
    
    rho.for_each(
        [](size_t i, double& v, auto pos) {
            double x = pos.x();
            v = -8.854e-12 * (2.0 + 4.0*x);
        }
    );
    
    Eigen::Map<Eigen::VectorXd> rho_vec(rho.data(), rho.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());
    
    solver.solve(phi_vec, rho_vec);
    
    double max_err = 0;
    Phi.for_each(
        [&](size_t i, double& v, auto pos) {
            double x = pos.x();
            double exact = x*x;
            double err = abs(v - exact);
            max_err = max(max_err, err);
        }
    );
    
    cout << "Max error: " << max_err << endl;
    
    if (max_err < 1e-8) {
        cout << "✅ PASSED" << endl;
    } else {
        cout << "❌ FAILED" << endl;
    }
}

int main() {
    test_manufactured_solution();
    cout << endl;
    test_simple_case();
    cout << endl;
    test_neumann_boundary();
    cout << endl;
    test_robin_boundary();
    cout << endl;
    test_variable_epsilon();
    cout << endl;
    return 0;
}
