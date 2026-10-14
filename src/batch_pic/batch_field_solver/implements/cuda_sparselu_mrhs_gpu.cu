#include "../register/interface.h"
#include "../../../timer.hpp"
#include "cuda_sparse_lu_mrhs_common.cuh"
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace psum {

namespace batch_pic {

namespace batch_field_solver {

namespace implements {

namespace cuda_sparselu_mrhs_gpu {

__global__ void apply_source_clear_kernel(unsigned long long size, int batch_size, unsigned long long case_id, const unsigned long long* idxs, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idxs[idx] * batch_size + case_id] = 0.0;
    }
}

__global__ void apply_source_accumulate_kernel(unsigned long long size, int batch_size, unsigned long long case_id, const unsigned long long* idxs, const double* new_values, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&b[idxs[idx] * batch_size + case_id], new_values[idx]);
    }
}

struct CudaLUSolver {
    cuda_sparse_lu_mrhs_common::SparseLUCudaMRHS solver;
    bool initialized = false;
    int N;
    int batch_size = 0;
    std::vector<std::vector<unsigned long long>> source_replace_idxs;
    std::vector<std::vector<double>> source_replace_values;
    std::vector<unsigned long long*> d_source_replace_idxs;
    std::vector<double*> d_source_replace_values;
    std::vector<std::vector<unsigned long long>> source_addback_idxs;
    std::vector<std::vector<double>> source_addback_values;
    std::vector<unsigned long long*> d_source_addback_idxs;
    std::vector<double*> d_source_addback_values;

    void resize_sources(unsigned long long cases) {
        for (size_t c = cases; c < source_replace_idxs.size(); ++c) {
            if (d_source_replace_idxs[c]) cudaFree(d_source_replace_idxs[c]);
            if (d_source_replace_values[c]) cudaFree(d_source_replace_values[c]);
            if (d_source_addback_idxs[c]) cudaFree(d_source_addback_idxs[c]);
            if (d_source_addback_values[c]) cudaFree(d_source_addback_values[c]);
        }
        source_replace_idxs.resize(cases);
        source_replace_values.resize(cases);
        d_source_replace_idxs.resize(cases, nullptr);
        d_source_replace_values.resize(cases, nullptr);
        source_addback_idxs.resize(cases);
        source_addback_values.resize(cases);
        d_source_addback_idxs.resize(cases, nullptr);
        d_source_addback_values.resize(cases, nullptr);
        batch_size = (int)cases;
    }
};

psum_batch_field_solver_handle init() {
    return new CudaLUSolver();
}

void set_matrix(psum_batch_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
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

void solve(psum_batch_field_solver_handle h, double* b, double* x) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (!solver->initialized) {
        throw std::runtime_error("Solver not initialized");
    }

    for (unsigned long long c = 0; c < solver->source_addback_idxs.size(); ++c) {
        if (solver->source_addback_idxs[c].empty()) continue;
        const int block_size = 256;
        const int grid_size = (int)((solver->source_addback_idxs[c].size() + block_size - 1) / block_size);
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(solver->source_addback_idxs[c].size(), solver->batch_size, c, solver->d_source_addback_idxs[c], solver->d_source_addback_values[c], b);
    }

    for (unsigned long long c = 0; c < solver->source_replace_idxs.size(); ++c) {
        if (solver->source_replace_idxs[c].empty()) continue;
        const int block_size = 256;
        const int grid_size = (int)((solver->source_replace_idxs[c].size() + block_size - 1) / block_size);
        apply_source_clear_kernel<<<grid_size, block_size>>>(solver->source_replace_idxs[c].size(), solver->batch_size, c, solver->d_source_replace_idxs[c], b);
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(solver->source_replace_idxs[c].size(), solver->batch_size, c, solver->d_source_replace_idxs[c], solver->d_source_replace_values[c], b);
    }

    cudaDeviceSynchronize();
    Tic("cuda_sparselu_mrhs_gpu.sparse_lu");
    solver->solver.solve(b, x);
    cudaDeviceSynchronize();
    Toc_("cuda_sparselu_mrhs_gpu.sparse_lu");
}

// options are "key=value" pairs joined by ';' or ',': batch_size is the
// only key this backend knows, unknown keys are ignored
void set_options(psum_batch_field_solver_handle h, const char* options) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (options == nullptr) return;
    std::string opt(options);
    size_t pos = 0;
    while (pos < opt.size()) {
        size_t sep = opt.find_first_of(";,", pos);
        std::string pair = opt.substr(pos, sep == std::string::npos ? std::string::npos : sep - pos);
        pos = (sep == std::string::npos) ? opt.size() : sep + 1;
        size_t eq = pair.find('=');
        if (eq == std::string::npos) continue;
        std::string key = pair.substr(0, eq);
        std::string value = pair.substr(eq + 1);
        auto trim = [](std::string& v) {
            while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
            while (!v.empty() && (v.back() == ' ' || v.back() == '\t')) v.pop_back();
        };
        trim(key);
        trim(value);
        if (key == "batch_size") {
            solver->resize_sources(std::max(1, std::stoi(value)));
            solver->solver.set_batch_size(solver->batch_size);
        }
    }
}

void set_source_replace(psum_batch_field_solver_handle h, unsigned long long case_id, unsigned long long size, unsigned long long* idxs, double* new_values) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (case_id >= solver->source_replace_idxs.size()) {
        solver->resize_sources(case_id + 1);
        solver->solver.set_batch_size(solver->batch_size);
    }
    solver->source_replace_idxs[case_id].clear();
    solver->source_replace_values[case_id].clear();
    solver->source_replace_idxs[case_id].reserve(size);
    solver->source_replace_values[case_id].reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        solver->source_replace_idxs[case_id].push_back(idxs[i]);
        solver->source_replace_values[case_id].push_back(new_values[i]);
    }

    if (solver->d_source_replace_idxs[case_id]) cudaFree(solver->d_source_replace_idxs[case_id]);
    if (solver->d_source_replace_values[case_id]) cudaFree(solver->d_source_replace_values[case_id]);

    cudaMalloc(&solver->d_source_replace_idxs[case_id], size * sizeof(unsigned long long));
    cudaMalloc(&solver->d_source_replace_values[case_id], size * sizeof(double));
    cudaMemcpy(solver->d_source_replace_idxs[case_id], idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(solver->d_source_replace_values[case_id], new_values, size * sizeof(double), cudaMemcpyHostToDevice);
}

void set_source_addback(psum_batch_field_solver_handle h, unsigned long long case_id, unsigned long long size, unsigned long long* idxs, double* add_values) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    if (case_id >= solver->source_addback_idxs.size()) {
        solver->resize_sources(case_id + 1);
        solver->solver.set_batch_size(solver->batch_size);
    }
    solver->source_addback_idxs[case_id].clear();
    solver->source_addback_values[case_id].clear();
    solver->source_addback_idxs[case_id].reserve(size);
    solver->source_addback_values[case_id].reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        solver->source_addback_idxs[case_id].push_back(idxs[i]);
        solver->source_addback_values[case_id].push_back(add_values[i]);
    }

    if (solver->d_source_addback_idxs[case_id]) cudaFree(solver->d_source_addback_idxs[case_id]);
    if (solver->d_source_addback_values[case_id]) cudaFree(solver->d_source_addback_values[case_id]);

    cudaMalloc(&solver->d_source_addback_idxs[case_id], size * sizeof(unsigned long long));
    cudaMalloc(&solver->d_source_addback_values[case_id], size * sizeof(double));
    cudaMemcpy(solver->d_source_addback_idxs[case_id], idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(solver->d_source_addback_values[case_id], add_values, size * sizeof(double), cudaMemcpyHostToDevice);
}

void solver_free(psum_batch_field_solver_handle h) {
    CudaLUSolver* solver = static_cast<CudaLUSolver*>(h);
    for (size_t c = 0; c < solver->d_source_replace_idxs.size(); ++c) {
        if (solver->d_source_replace_idxs[c]) cudaFree(solver->d_source_replace_idxs[c]);
        if (solver->d_source_replace_values[c]) cudaFree(solver->d_source_replace_values[c]);
        if (solver->d_source_addback_idxs[c]) cudaFree(solver->d_source_addback_idxs[c]);
        if (solver->d_source_addback_values[c]) cudaFree(solver->d_source_addback_values[c]);
    }
    delete solver;
}

static int _ = []() {
    batch_func_table table;
    table.init = init;
    table.set_matrix = set_matrix;
    table.solve = solve;
    table.set_options = set_options;
    table.set_source_replace = set_source_replace;
    table.set_source_addback = set_source_addback;
    table.free = solver_free;
    register_batch_solver("cuda_sparselu_mrhs_gpu", table);
    return 0;
}();

}

}

}

}

}
