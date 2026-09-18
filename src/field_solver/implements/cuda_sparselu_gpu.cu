#include "../register/interface.h"
#include "../../timer.hpp"
#include "cuda_sparse_lu_common.cuh"
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace psum {

namespace field_solver {

namespace implements {

namespace cuda_sparselu_gpu {

__global__ void apply_source_clear_kernel(unsigned long long size, const unsigned long long* idxs, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idxs[idx]] = 0.0;
    }
}

__global__ void apply_source_accumulate_kernel(unsigned long long size, const unsigned long long* idxs, const double* new_values, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&b[idxs[idx]], new_values[idx]);
    }
}

struct CudaLUSolver {
    cuda_sparse_lu_common::SparseLUCuda solver;
    bool initialized = false;
    int N;
    std::vector<unsigned long long> source_replace_idxs;
    std::vector<double> source_replace_values;
    unsigned long long* d_source_replace_idxs = nullptr;
    double* d_source_replace_values = nullptr;
    unsigned long long source_replace_size = 0;
    std::vector<unsigned long long> source_addback_idxs;
    std::vector<double> source_addback_values;
    unsigned long long* d_source_addback_idxs = nullptr;
    double* d_source_addback_values = nullptr;
    unsigned long long source_addback_size = 0;
};

psum_field_solver_handle init() {
    return new CudaLUSolver();
}

void set_matrix(psum_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);

    solver->N = n_row;

    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix(n_row, n_col);
    matrix.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    for (unsigned long long i = 0; i < nnz; ++i) {
        triplets.emplace_back(rows[i], cols[i], vals[i]);
    }
    matrix.setFromTriplets(triplets.begin(), triplets.end());

    solver->solver.set(matrix);
    solver->initialized = true;
}

void solve(psum_field_solver_handle h, double* b, double* x) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (!solver->initialized) {
        throw std::runtime_error("Solver not initialized");
    }

    if (solver->source_addback_size > 0) {
        const int block_size = 256;
        const int grid_size = (solver->source_addback_size + block_size - 1) / block_size;
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(solver->source_addback_size, solver->d_source_addback_idxs, solver->d_source_addback_values, b);
    }

    if (solver->source_replace_size > 0) {
        const int block_size = 256;
        const int grid_size = (solver->source_replace_size + block_size - 1) / block_size;
        apply_source_clear_kernel<<<grid_size, block_size>>>(solver->source_replace_size, solver->d_source_replace_idxs, b);
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(solver->source_replace_size, solver->d_source_replace_idxs, solver->d_source_replace_values, b);
    }

    cudaDeviceSynchronize();
    Tic("cuda_sparselu_gpu.sparse_lu");
    solver->solver.solve(b, x);
    cudaDeviceSynchronize();
    Toc_("cuda_sparselu_gpu.sparse_lu");
}

void set_options(psum_field_solver_handle, const char*) {
}

void set_source_replace(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* new_values) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    solver->source_replace_idxs.clear();
    solver->source_replace_values.clear();
    solver->source_replace_idxs.reserve(size);
    solver->source_replace_values.reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        solver->source_replace_idxs.push_back(idxs[i]);
        solver->source_replace_values.push_back(new_values[i]);
    }

    if (solver->d_source_replace_idxs) cudaFree(solver->d_source_replace_idxs);
    if (solver->d_source_replace_values) cudaFree(solver->d_source_replace_values);

    solver->source_replace_size = size;
    cudaMalloc(&solver->d_source_replace_idxs, size * sizeof(unsigned long long));
    cudaMalloc(&solver->d_source_replace_values, size * sizeof(double));
    cudaMemcpy(solver->d_source_replace_idxs, idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(solver->d_source_replace_values, new_values, size * sizeof(double), cudaMemcpyHostToDevice);
}

void set_source_addback(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* add_values) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    solver->source_addback_idxs.clear();
    solver->source_addback_values.clear();
    solver->source_addback_idxs.reserve(size);
    solver->source_addback_values.reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        solver->source_addback_idxs.push_back(idxs[i]);
        solver->source_addback_values.push_back(add_values[i]);
    }

    if (solver->d_source_addback_idxs) cudaFree(solver->d_source_addback_idxs);
    if (solver->d_source_addback_values) cudaFree(solver->d_source_addback_values);

    solver->source_addback_size = size;
    cudaMalloc(&solver->d_source_addback_idxs, size * sizeof(unsigned long long));
    cudaMalloc(&solver->d_source_addback_values, size * sizeof(double));
    cudaMemcpy(solver->d_source_addback_idxs, idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(solver->d_source_addback_values, add_values, size * sizeof(double), cudaMemcpyHostToDevice);
}

void solver_free(psum_field_solver_handle h) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (solver->d_source_replace_idxs) cudaFree(solver->d_source_replace_idxs);
    if (solver->d_source_replace_values) cudaFree(solver->d_source_replace_values);
    if (solver->d_source_addback_idxs) cudaFree(solver->d_source_addback_idxs);
    if (solver->d_source_addback_values) cudaFree(solver->d_source_addback_values);
    delete solver;
}

static int _ = []() {
    func_table table;
    table.init = init;
    table.set_matrix = set_matrix;
    table.solve = solve;
    table.set_options = set_options;
    table.set_source_replace = set_source_replace;
    table.set_source_addback = set_source_addback;
    table.free = solver_free;
    register_solver("cuda_sparselu_gpu", table);
    return 0;
}();

}

}

}

}
