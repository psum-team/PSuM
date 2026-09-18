#include <Eigen/Sparse>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <psum/field.hpp>
#include "../../src/field_solver/backend.hpp"

using namespace psum::field_solver;

using clock_type = std::chrono::steady_clock;

double elapsed_seconds(clock_type::time_point begin, clock_type::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

void build_five_point_stencil(int n,
                              std::vector<unsigned long long>& rows,
                              std::vector<unsigned long long>& cols,
                              std::vector<double>& vals,
                              std::vector<double>& b) {
    const int N = n * n;
    const double h = 1.0 / (n + 1);

    rows.clear();
    cols.clear();
    vals.clear();
    b.assign(N, 1.0);
    rows.reserve(5 * N);
    cols.reserve(5 * N);
    vals.reserve(5 * N);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const int idx = j * n + i;
            rows.push_back(idx);
            cols.push_back(idx);
            vals.push_back(4.0 / (h * h));

            if (i > 0) {
                rows.push_back(idx);
                cols.push_back(idx - 1);
                vals.push_back(-1.0 / (h * h));
            }
            if (i < n - 1) {
                rows.push_back(idx);
                cols.push_back(idx + 1);
                vals.push_back(-1.0 / (h * h));
            }
            if (j > 0) {
                rows.push_back(idx);
                cols.push_back(idx - n);
                vals.push_back(-1.0 / (h * h));
            }
            if (j < n - 1) {
                rows.push_back(idx);
                cols.push_back(idx + n);
                vals.push_back(-1.0 / (h * h));
            }
        }
    }
}

double compute_residual(const std::vector<unsigned long long>& rows,
                        const std::vector<unsigned long long>& cols,
                        const std::vector<double>& vals,
                        const std::vector<double>& b,
                        const std::vector<double>& x) {
    const int N = static_cast<int>(b.size());
    Eigen::SparseMatrix<double> matrix(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());
    for (std::size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(static_cast<int>(rows[k]), static_cast<int>(cols[k]), vals[k]);
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    const Eigen::Map<const Eigen::VectorXd> x_eigen(x.data(), N);
    const Eigen::Map<const Eigen::VectorXd> b_eigen(b.data(), N);
    return (matrix * x_eigen - b_eigen).cwiseAbs().maxCoeff();
}

void test_cpu_save_load(int n) {
    std::vector<unsigned long long> rows;
    std::vector<unsigned long long> cols;
    std::vector<double> vals;
    std::vector<double> b;
    build_five_point_stencil(n, rows, cols, vals, b);

    const int N = n * n;
    const std::string state_path = "./psum_eigen_schur_state_test.bin";
    const std::string options = "I " + std::to_string(n) + " J " + std::to_string(n) +
                                " blocks_x 5 blocks_y 5 num_threads 6";

    solver_backend setup_solver("eigen_schurcomplement_cpu");
    setup_solver.set_options(options.c_str());
    const auto setup_begin = clock_type::now();
    setup_solver.set_matrix(N, N, rows.size(), rows.data(), cols.data(), vals.data());
    setup_solver.set_options(("save_state " + state_path).c_str());
    const auto setup_end = clock_type::now();

    solver_backend loaded_solver("eigen_schurcomplement_cpu");
    const auto load_begin = clock_type::now();
    loaded_solver.set_options(("load_state " + state_path).c_str());
    const auto load_end = clock_type::now();

    std::vector<double> x(N, 0.0);
    const auto solve_begin = clock_type::now();
    loaded_solver.solve(b.data(), x.data());
    const auto solve_end = clock_type::now();

    const double residual = compute_residual(rows, cols, vals, b, x);
    if (residual >= 1e-8) {
        throw std::runtime_error("CPU Schur save/load residual too large: " + std::to_string(residual));
    }
    std::cout << "CPU Schur setup+save: " << elapsed_seconds(setup_begin, setup_end)
              << " s, load: " << elapsed_seconds(load_begin, load_end)
              << " s, solve: " << elapsed_seconds(solve_begin, solve_end)
              << " s, residual: " << residual << std::endl;
}

void test_cuda_save_load(int n) {
    std::vector<unsigned long long> rows;
    std::vector<unsigned long long> cols;
    std::vector<double> vals;
    std::vector<double> b;
    build_five_point_stencil(n, rows, cols, vals, b);

    const int N = n * n;
    const std::string state_path = "./psum_cuda_schur_state_test.bin";
    const std::string options = "I " + std::to_string(n) + " J " + std::to_string(n) +
                                " blocks_x 5 blocks_y 5";

    solver_backend setup_solver("cuda_schurcomplement_gpu");
    setup_solver.set_options(options.c_str());
    const auto setup_begin = clock_type::now();
    setup_solver.set_matrix(N, N, rows.size(), rows.data(), cols.data(), vals.data());
    setup_solver.set_options(("save_state " + state_path).c_str());
    const auto setup_end = clock_type::now();

    solver_backend loaded_solver("cuda_schurcomplement_gpu");
    const auto load_begin = clock_type::now();
    loaded_solver.set_options(("load_state " + state_path).c_str());
    const auto load_end = clock_type::now();

    sycl::queue q{sycl::gpu_selector()};
    psum::field::device_array<double> b_d(q, b);
    psum::field::device_array<double> x_d(q, std::vector<double>(N, 0.0));
    q.wait();
    const auto solve_begin = clock_type::now();
    loaded_solver.solve(b_d.data(), x_d.data());
    q.wait();
    const auto solve_end = clock_type::now();

    const std::vector<double> x = x_d.to_host();
    const double residual = compute_residual(rows, cols, vals, b, x);
    if (residual >= 1e-7) {
        throw std::runtime_error("CUDA Schur save/load residual too large: " + std::to_string(residual));
    }
    std::cout << "CUDA Schur setup+save: " << elapsed_seconds(setup_begin, setup_end)
              << " s, load: " << elapsed_seconds(load_begin, load_end)
              << " s, solve: " << elapsed_seconds(solve_begin, solve_end)
              << " s, residual: " << residual << std::endl;
}

int main() {
    try {
        constexpr int n = 1000;
        std::cout << "Schur state save/load test grid: " << n << "x" << n << std::endl;
        test_cpu_save_load(n);
        test_cuda_save_load(n);
        std::cout << "Schur state save/load tests passed." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Schur state save/load test failed: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
