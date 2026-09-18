#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <psum/field.hpp>
#include "../../src/field_solver/backend.hpp"
#include "../../src/timer.hpp"
#include "../../src/random/rander.hpp"

using namespace psum::field_solver;
using namespace psum::random;

void build_five_point_stencil(int n, std::vector<unsigned long long>& rows,
                               std::vector<unsigned long long>& cols,
                               std::vector<double>& vals,
                               std::vector<double>& b);

double compute_residual(const std::vector<unsigned long long>& rows,
                        const std::vector<unsigned long long>& cols,
                        const std::vector<double>& vals,
                        const std::vector<double>& b,
                        const std::vector<double>& x);

void benchmark_direct_eigen(int n, int repeats = 10);
void benchmark_backend(const char* solver_name, int n, int repeats = 10);
void benchmark_backend_gpu(const char* solver_name, int n, int repeats = 10, const std::string& div_n="none");

int main() {
    std::vector<int> grid_sizes = {200, 300, 400, 500, 600};

    std::cout << "Checking solvers..." << std::endl;
    std::cout << "Problem: 5-point Poisson" << std::endl;
    std::cout << "Testing solver correctness" << std::endl;
    std::cout << std::endl;

    for (int n : grid_sizes) {
        std::cout << "Grid " << n << "x" << n << " (N=" << n*n << "): ";
        std::cout.flush();

        std::cout << "CPU ";
        benchmark_backend("eigen_sparselu_cpu", n, 10);
        std::cout << "MG ";
        benchmark_backend("eigen_multigrid_cpu", n, 10);
        std::cout << "Schur ";
        benchmark_backend("eigen_schurcomplement_cpu", n, 10);
        std::cout << "GPU ";
        benchmark_backend_gpu("cuda_sparselu_gpu", n, 10);
        std::cout << "SchurGPU ";
        benchmark_backend_gpu("cuda_schurcomplement_gpu", n, 10, "5");
        std::cout << std::endl;
    }

    std::cout << std::endl;
    PrintTimer;
    return 0;
}

void build_five_point_stencil(int n, std::vector<unsigned long long>& rows,
                               std::vector<unsigned long long>& cols,
                               std::vector<double>& vals,
                               std::vector<double>& b) {
    int N = n * n;
    double h = 1.0 / (n + 1);

    rows.clear();
    cols.clear();
    vals.clear();
    b.resize(N);

    int estimated_nnz = 5 * N;
    rows.reserve(estimated_nnz);
    cols.reserve(estimated_nnz);
    vals.reserve(estimated_nnz);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            int idx = j * n + i;
            b[idx] = 1.0;

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
    int N = b.size();
    Eigen::SparseMatrix<double> matrix(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());

    for (size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(rows[k], cols[k], vals[k]);
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd x_eigen = Eigen::Map<const Eigen::VectorXd>(x.data(), N);
    Eigen::VectorXd b_eigen = Eigen::Map<const Eigen::VectorXd>(b.data(), N);
    Eigen::VectorXd residual = matrix * x_eigen - b_eigen;

    return residual.cwiseAbs().maxCoeff();
}

void benchmark_direct_eigen(int n, int repeats) {
    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals, b;
    int N = n * n;

    build_five_point_stencil(n, rows, cols, vals, b);

    Eigen::SparseMatrix<double> matrix(N, N);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(rows.size());

    for (size_t k = 0; k < rows.size(); ++k) {
        triplets.emplace_back(rows[k], cols[k], vals[k]);
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd b_eigen = Eigen::Map<Eigen::VectorXd>(b.data(), N);
    Eigen::VectorXd x_eigen(N);

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(matrix);
    solver.factorize(matrix);

    for (int r = 0; r < repeats; ++r) {
        Tic("eigen_direct");
        x_eigen = solver.solve(b_eigen);
        Toc;
    }

    std::vector<double> x_std(x_eigen.data(), x_eigen.data() + x_eigen.size());
    double max_residual = compute_residual(rows, cols, vals, b, x_std);
    if (max_residual < 1e-8) {
        std::cout << "✅ ";
    } else {
        std::cout << "❌ (res=" << max_residual << ") ";
    }
}

void benchmark_backend(const char* solver_name, int n, int repeats) {
    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals, b, x;
    int N = n * n;
    build_five_point_stencil(n, rows, cols, vals, b);
    x.resize(N);

    solver_backend solver;
    try {
        solver_backend solver_temp(solver_name);
        std::swap(solver, solver_temp);
    }
    catch (const std::exception& e) {
        std::cout << "❌ (exception: " << e.what() << ") ";
        return;
    }

    std::string solver_name_str(solver_name);
    if (solver_name_str == "eigen_multigrid_cpu") {
        int depth = 3;
        while ((n >> depth) > 16) depth++;
        int max_cycles = 20;
        std::string options = "I " + std::to_string(n) 
            + " J " + std::to_string(n) + " K 1 depth " 
            + std::to_string(depth) + " max_cycles " 
            + std::to_string(max_cycles) + " pre_relax 4 post_relax 2";
        solver.set_options(options.c_str());
    } else if (solver_name_str == "eigen_schurcomplement_cpu") {
        std::string options = "I " + std::to_string(n)
            + " J " + std::to_string(n)
            + " blocks_x 5 blocks_y 5 num_threads 8";
        solver.set_options(options.c_str());
    }

    solver.set_matrix(N, N, rows.size(), rows.data(), cols.data(), vals.data());
    for (int r = 0; r < repeats; ++r) {
        solver.solve(b.data(), x.data());
    }
    double max_residual = compute_residual(rows, cols, vals, b, x);
    if (max_residual < 1e-8) {
        std::cout << "✅ ";
    } else {
        std::cout << "❌ (res=" << max_residual << ") ";
    }
}

void benchmark_backend_gpu(const char* solver_name, int n, int repeats, const std::string& div_n) {
    std::vector<unsigned long long> rows, cols;
    std::vector<double> vals, b, x;
    int N = n * n;
    build_five_point_stencil(n, rows, cols, vals, b);
    x.resize(N);

    solver_backend solver;
    try {
        solver_backend solver_temp(solver_name);
        std::swap(solver, solver_temp);
    }
    catch (const std::exception& e) {
        std::cout << "❌ (exception: " << e.what() << ") ";
        return;
    }

    std::string solver_name_str(solver_name);
    if (solver_name_str == "cuda_schurcomplement_gpu") {
        std::string options = "I " + std::to_string(n)
            + " J " + std::to_string(n)
            + " blocks_x " + div_n + " blocks_y " + div_n;
        solver.set_options(options.c_str());
    }

    solver.set_matrix(N, N, rows.size(), rows.data(), cols.data(), vals.data());
    sycl::queue q{sycl::gpu_selector()};
    psum::field::device_array<double> b_d(q, b);
    psum::field::device_array<double> x_d(q, b);
    for (int r = 0; r < repeats; ++r) {
        solver.solve(b_d.data(), x_d.data());
    }
    x = x_d.to_host();
    double max_residual = compute_residual(rows, cols, vals, b, x);
    if (max_residual < 1e-7) {
        std::cout << "✅ ";
    } else {
        std::cout << "❌ (res=" << max_residual << ") ";
    }
}
