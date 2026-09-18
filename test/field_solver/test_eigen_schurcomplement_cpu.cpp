#include "../../src/field_solver/backend.hpp"
#include "../../src/field_solver/Poisson_solver_2d.hpp"
#include "../../src/field_solver/boundary_creator/Dirichlet.hpp"
#include <Eigen/Sparse>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace psum::field_solver;
using namespace psum::field_solver::boundary_creator;

void build_five_point_stencil(int nx, int ny,
                              std::vector<unsigned long long>& rows,
                              std::vector<unsigned long long>& cols,
                              std::vector<double>& vals,
                              std::vector<double>& b) {
    const int n = nx * ny;
    rows.clear();
    cols.clear();
    vals.clear();
    b.assign(n, 1.0);

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const int idx = i * ny + j;
            rows.push_back(idx);
            cols.push_back(idx);
            vals.push_back(4.0);

            if (i > 0) {
                rows.push_back(idx);
                cols.push_back((i - 1) * ny + j);
                vals.push_back(-1.0);
            }
            if (i + 1 < nx) {
                rows.push_back(idx);
                cols.push_back((i + 1) * ny + j);
                vals.push_back(-1.0);
            }
            if (j > 0) {
                rows.push_back(idx);
                cols.push_back(i * ny + (j - 1));
                vals.push_back(-1.0);
            }
            if (j + 1 < ny) {
                rows.push_back(idx);
                cols.push_back(i * ny + (j + 1));
                vals.push_back(-1.0);
            }
        }
    }
}

double residual_norm(int n,
                     const std::vector<unsigned long long>& rows,
                     const std::vector<unsigned long long>& cols,
                     const std::vector<double>& vals,
                     const std::vector<double>& b,
                     const std::vector<double>& x) {
    Eigen::SparseMatrix<double> A(n, n);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(vals.size());
    for (size_t k = 0; k < vals.size(); ++k) {
        triplets.emplace_back(rows[k], cols[k], vals[k]);
    }
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::Map<const Eigen::VectorXd> x_vec(x.data(), n);
    Eigen::Map<const Eigen::VectorXd> b_vec(b.data(), n);
    return (A * x_vec - b_vec).norm() / b_vec.norm();
}

int main() {
    {
        const int nx = 12;
        const int ny = 10;
        const int n = nx * ny;

        std::vector<unsigned long long> rows;
        std::vector<unsigned long long> cols;
        std::vector<double> vals;
        std::vector<double> b;
        std::vector<double> x(n, 0.0);
        build_five_point_stencil(nx, ny, rows, cols, vals, b);

        solver_backend solver("eigen_schurcomplement_cpu");
        solver.set_options("I 12 J 10 blocks_x 3 blocks_y 2 num_threads 4");
        solver.set_matrix(n, n, rows.size(), rows.data(), cols.data(), vals.data());
        solver.solve(b.data(), x.data());

        const double rel_residual = residual_norm(n, rows, cols, vals, b, x);
        if (rel_residual > 1e-10) {
            std::cerr << "eigen_schurcomplement_cpu residual too large: " << rel_residual << std::endl;
            return 1;
        }
        std::cout << "eigen_schurcomplement_cpu backend residual: " << rel_residual << std::endl;
    }

    {
        psum::field::grid2D grid({0.0, 0.0}, {1.0, 1.0}, {8, 6});
        const int node_x = grid.numCells<0>() + 1;
        const int node_y = grid.numCells<1>() + 1;

        Poisson_solver_2d native_solver;
        native_solver.init(
            "native",
            grid,
            Poisson_solver_2d::Cartesian,
            {
                Dirichlet_line(grid, psum::field::boundary_direction_2d::N) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::S) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::E) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::W) = 0.0
            }
        );

        Poisson_solver_2d schur_solver;
        schur_solver.set_options("I " + std::to_string(node_x) +
                                 " J " + std::to_string(node_y) +
                                 " blocks_x 2 blocks_y 3 num_threads 4");
        schur_solver.init(
            "eigen_schurcomplement_cpu",
            grid,
            Poisson_solver_2d::Cartesian,
            {
                Dirichlet_line(grid, psum::field::boundary_direction_2d::N) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::S) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::E) = 0.0,
                Dirichlet_line(grid, psum::field::boundary_direction_2d::W) = 0.0
            }
        );

        Eigen::VectorXd source = Eigen::VectorXd::Ones(node_x * node_y);
        Eigen::VectorXd native_phi(node_x * node_y);
        Eigen::VectorXd schur_phi(node_x * node_y);
        native_solver.solve(native_phi, source);
        schur_solver.solve(schur_phi, source);

        const double relative_diff = (native_phi - schur_phi).norm() / native_phi.norm();
        if (relative_diff > 1e-10) {
            std::cerr << "eigen_schurcomplement_cpu Poisson diff too large: " << relative_diff << std::endl;
            return 1;
        }
        std::cout << "eigen_schurcomplement_cpu Poisson diff: " << relative_diff << std::endl;
    }

    return 0;
}
