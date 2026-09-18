#include <psum/field.hpp>
#include "../../src/field_solver/Poisson_solver_3d.hpp"
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

const double domain_range_z = 0.080;
const double domain_range_r = 0.070;
const double thruster_body_up_length = 0.025;
const double thruster_body_down_length = 0.025;
const double thruster_body_up_thickness = 0.016;
const double thruster_body_down_thickness = 0.031;
const double ceramic_thickness = 0.004;
const double metal_thickness = 0.003;

std::vector<Eigen::VectorXd> solve_3d_case(int nz, int ny, int nx, std::string backend, std::vector<int> cycles) {

    grid3D g(
        {-domain_range_r, -domain_range_r, 0}, 
        {domain_range_r, domain_range_r, domain_range_z}, 
        {nx, ny, nz}
    );

    Poisson_solver_3d poisson_solver;
    double reflect_threshold = -10;

    char options[256];
    snprintf(options, sizeof(options), "I %d J %d K %d depth %d", nx + 1, ny + 1, nz + 1, 2);

    if (backend != "native") poisson_solver.set_options(options);
    poisson_solver.init(
        backend,
        g,
        Poisson_solver_3d::Cartesian,
        {
            Robin_plane(g, boundary_direction_3d::Xpos,
                [](double pos_x, double pos_y, double pos_z)
                {
                    double value_coeff = 1.0;
                    double grad_coeff = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) / pos_x;
                    return make_pair(value_coeff, grad_coeff);
                }
            ) = reflect_threshold,
            Robin_plane(g, boundary_direction_3d::Xneg,
                [](double pos_x, double pos_y, double pos_z)
                {
                    double value_coeff = 1.0;
                    double grad_coeff = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) / -pos_x;
                    return make_pair(value_coeff, grad_coeff);
                }
            ) = reflect_threshold,
            Robin_plane(g, boundary_direction_3d::Ypos,
                [](double pos_x, double pos_y, double pos_z)
                {
                    double value_coeff = 1.0;
                    double grad_coeff = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) / pos_y;
                    return make_pair(value_coeff, grad_coeff);
                }
            ) = reflect_threshold,
            Robin_plane(g, boundary_direction_3d::Yneg,
                [](double pos_x, double pos_y, double pos_z)
                {
                    double value_coeff = 1.0;
                    double grad_coeff = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) / -pos_y;
                    return make_pair(value_coeff, grad_coeff);
                }
            ) = reflect_threshold,
            Robin_plane(g, boundary_direction_3d::Zpos,
                [](double pos_x, double pos_y, double pos_z)
                {
                    double value_coeff = 1.0;
                    double grad_coeff = (pos_x * pos_x + pos_y * pos_y + pos_z * pos_z) / pos_z;
                    return make_pair(value_coeff, grad_coeff);
                }
            ) = reflect_threshold,
            Neumann_plane(g, boundary_direction_3d::Zneg) = 0,
        },
        {
            Dirichlet_plane(g, boundary_direction_3d::Zneg,
                [&](double x, double y)
                {
                    double r = sqrt(x * x + y * y);
                    return (r > thruster_body_down_thickness && r < domain_range_r - thruster_body_up_thickness);
                }
            ) = 300,
            Dirichlet_box_rz(
                g, 
                thruster_body_down_length - metal_thickness, 0, 
                thruster_body_down_length, thruster_body_down_thickness
            ) = -10,
            Dirichlet_box_rz(
                g, 
                thruster_body_up_length - metal_thickness, domain_range_r - thruster_body_up_thickness, 
                thruster_body_up_length, domain_range_r
            ) = -10
        },
        [](double pos_x, double pos_y, double pos_z)
        {
            double pos_r = sqrt(pos_x * pos_x + pos_y * pos_y);
            if ((pos_z > 0.0 && pos_z < thruster_body_down_length && pos_r > thruster_body_down_thickness && pos_r < thruster_body_down_thickness + ceramic_thickness) ||
                (pos_z > 0.0 && pos_z < thruster_body_up_length && pos_r > domain_range_r - thruster_body_up_thickness - ceramic_thickness && pos_r < domain_range_r - thruster_body_up_thickness))
                return 4 * 8.854e-12;
            else
                return 8.854e-12;
        }
    );


    host_node_field3D<double> Phi(g);
    host_node_field3D<double> qdens(g);

    qdens.setZero();

    for (size_t i = 0; i < qdens.size(); i++)
    {
        auto node_idx = qdens.getGrid().i2n(i);
        auto pos = qdens.getGrid().nodePosition(node_idx);
        double nx_val = pos.x();
        double ny_val = pos.y();
        double nz_val = pos.z();
        double r = sqrt(nx_val * nx_val + ny_val * ny_val);
        if (nz_val > 0.005 && nz_val < 0.02 && r > 0.037 && r < 0.048)
            qdens(i) = 1e-5;
    }


    Eigen::Map<Eigen::VectorXd> qdens_vec(qdens.data(), qdens.size());
    Eigen::Map<Eigen::VectorXd> phi_vec(Phi.data(), Phi.size());

    std::vector<Eigen::VectorXd> result;
    if (backend != "native") {
        for (int cycle : cycles) {
            std::string cycles_opt = "max_cycles " + std::to_string(cycle);
            poisson_solver.set_options(cycles_opt);
            poisson_solver.solve(phi_vec, qdens_vec);
            Phi.plot("test_hall_thruster_phi_mg_" + to_string(cycle) + ".plt", "phi");
            result.push_back(phi_vec);
        }
    } else {
        poisson_solver.solve(phi_vec, qdens_vec);
        // Phi.plot("test_hall_thruster_phi_native.plt", "phi");
        result.push_back(phi_vec);
    }
    return result;
}

void test_hall_thruster_case() {
    const int nx = 192 / 2;
    const int ny = 192 / 2;
    const int nz = 112 / 2;
    cout << "Grid size: " << nx << " x " << ny << " x " << nz << endl;

    cout << setw(12) << "Cycles" << setw(16) << "vs Native Err" << endl;
    cout << string(44, '-') << endl;


    Eigen::VectorXd phi_native_vec = solve_3d_case(nz, ny, nx, "native", {1})[0];

    // Test multigrid with increasing cycles
    vector<int> cycles = {1, 5, 15, 20};
    auto phi_mg_vec_vec = solve_3d_case(nz, ny, nx, "eigen_multigrid_cpu", cycles);

    for (int i = 0; i < cycles.size(); i++) {
        // Calculate L2 difference from native solution over entire domain
        double diff_native = (phi_native_vec - phi_mg_vec_vec[i]).norm() / phi_native_vec.norm();

        cout << setw(12) << cycles[i] << setw(16) << diff_native << endl;
    }
}

int main()
{
    cout << "\n========================================" << std::endl;
    cout << "Native vs Multigrid Convergence Tests" << std::endl;
    cout << "========================================" << std::endl;

    test_hall_thruster_case();

    cout << "\n========================================" << std::endl;
    cout << "All convergence tests completed!" << std::endl;
    cout << "========================================" << std::endl;

    return 0;
}
