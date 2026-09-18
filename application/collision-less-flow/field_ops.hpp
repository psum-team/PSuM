#ifndef PSUM_MAGNET_NOZZLE_2D_FIELD_OPS_HPP
#define PSUM_MAGNET_NOZZLE_2D_FIELD_OPS_HPP

#include "types.hpp"
#include "species.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <functional>

using namespace psum::field::interp_tools;
using namespace psum::field_solver::boundary_creator;

namespace magnet_nozzle_2d {

namespace detail {

inline double comp_ellint_1(double m) {
    if (m < 1e-15) return M_PI / 2.0;
    const int N = 64;
    const double h = M_PI / (2.0 * N);
    double sum = 0.0;
    for (int i = 0; i <= N; i++) {
        double theta = i * h;
        double sin_theta = sin(theta);
        double val = 1.0 / sqrt(1.0 - m * sin_theta * sin_theta);
        double w;
        if (i == 0 || i == N) w = 1.0;
        else if (i % 2 == 1) w = 4.0;
        else w = 2.0;
        sum += w * val;
    }
    return sum * h / 3.0;
}

inline double comp_ellint_2(double m) {
    if (m < 1e-15) return M_PI / 2.0;
    const int N = 64;
    const double h = M_PI / (2.0 * N);
    double sum = 0.0;
    for (int i = 0; i <= N; i++) {
        double theta = i * h;
        double sin_theta = sin(theta);
        double val = sqrt(1.0 - m * sin_theta * sin_theta);
        double w;
        if (i == 0 || i == N) w = 1.0;
        else if (i % 2 == 1) w = 4.0;
        else w = 2.0;
        sum += w * val;
    }
    return sum * h / 3.0;
}

}

inline double create_coil_field(DNf<2>& B, HNf<2>& B_host, double B_max, double x_max, double L) {
    const double a = L;
    const double mu0_I_over_2pi = a * B_max / M_PI;

    B_host.for_each([&](size_t idx, Eigen::RowVector<double, 2>& B_val, const auto& pos) {
        double x = pos.x();
        double r = pos.y();
        double dz = x - x_max;

        if (r < 1e-10 * a) {
            double a2_dz2 = a * a + dz * dz;
            double denom = a2_dz2 * sqrt(a2_dz2);
            B_val(0) = B_max * a * a * a / denom;
            B_val(1) = 0.0;
        } else {
            double beta2 = (a + r) * (a + r) + dz * dz;
            double beta = sqrt(beta2);
            double alpha2 = (a - r) * (a - r) + dz * dz;
            double m = 4.0 * a * r / beta2;

            double K = detail::comp_ellint_1(m);
            double E = detail::comp_ellint_2(m);

            double coeff = mu0_I_over_2pi / beta;

            B_val(0) = coeff * (K + (a * a - r * r - dz * dz) / alpha2 * E);
            B_val(1) = coeff * dz / r * (-K + (a * a + r * r + dz * dz) / alpha2 * E);
        }
    });

    B.copy(B_host.getContent());
    B.plot("output/coil_field.plt");
    std::cout << "Coil field created: B_max=" << B_max << ", x_max=" << x_max << ", L=" << L << std::endl;
    return B_max;
}

inline void init_poisson_solver(
    Poisson_solver_2d& solver,
    const Grid& grid,
    double r_length,
    double epsilon,
    std::function<double(const std::array<double, 2>&)> reflect_threshold_east,
    std::function<double(const std::array<double, 2>&)> reflect_threshold_north
) {
    solver.init(
        "cuda_sparselu_gpu",
        grid,
        Poisson_solver_2d::Cylindrical,
        {
            Robin_Line(
                grid, boundary_direction_2d::E,
                [](double x_pos, double y_pos){
                    double grad_coeff = (x_pos * x_pos + y_pos * y_pos) / x_pos;
                    return std::make_pair(1.0, grad_coeff);
                }
            )=reflect_threshold_east,
            Robin_Line(
                grid, boundary_direction_2d::N,
                [](double x_pos, double y_pos){
                    double grad_coeff = (x_pos * x_pos + y_pos * y_pos) / y_pos;
                    return std::make_pair(1.0, grad_coeff);
                }
            )=reflect_threshold_north,
            Neumann_line(grid, boundary_direction_2d::S) = 0.0,
        },
        {
            Dirichlet_line(grid, boundary_direction_2d::W, std::make_pair(0.0, r_length)) = 0.0
        },
        epsilon
    );
}

}

#endif
