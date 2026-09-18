#include "../register/interface.h"
#include "../../timer.hpp"
#include "cuda_sparse_lu_common.cuh"
#include "schur_complement/schur_complement_core.hpp"
#include <Eigen/Sparse>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace psum {

namespace field_solver {

namespace implements {

namespace cuda_schurcomplement_gpu {

__global__ void apply_source_clear_kernel(unsigned long long size, const unsigned long long* idxs, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idxs[idx]] = 0.0;
    }
}

__global__ void apply_source_accumulate_kernel(unsigned long long size, const unsigned long long* idxs, const double* values, double* b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(&b[idxs[idx]], values[idx]);
    }
}

__global__ void gather_by_global_index_kernel(int n, const int* globals, const double* src, double* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[globals[idx]];
    }
}

__global__ void scatter_by_global_index_kernel(int n, const int* globals, const double* src, double* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[globals[idx]] = src[idx];
    }
}

__global__ void subtract_matvec_kernel(int nnz, const int* rows, const int* cols, const double* vals,
                                       const double* src, double* dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        atomicAdd(&dst[rows[idx]], -vals[idx] * src[cols[idx]]);
    }
}

__global__ void dense_matvec_colmajor_kernel(int rows, int cols, const double* mat, const double* vec, double* result) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows) {
        double sum = 0.0;
        for (int j = 0; j < cols; ++j) {
            sum += mat[idx + j * rows] * vec[j];
        }
        result[idx] = sum;
    }
}

__global__ void load_sparse_column_kernel(int col, const int* outer, const int* inner, const double* vals, double* dst) {
    const int begin = outer[col];
    const int end = outer[col + 1];
    for (int k = begin + blockIdx.x * blockDim.x + threadIdx.x; k < end; k += blockDim.x * gridDim.x) {
        dst[inner[k]] = vals[k];
    }
}

using cuda_sparse_lu_common::SparseLUCuda;

struct SolverContext {
    unsigned long long n = 0;
    int I = 0;
    int J = 0;
    int blocks_x = 1;
    int blocks_y = 1;
    int num_threads = 0;
    bool initialized = false;
    bool loaded_state = false;

    Eigen::SparseMatrix<double> A;
    schur_complement::schur_complement_core core;
    std::vector<int> interface_nodes;

    SparseLUCuda interior_lu;
    bool has_interior_lu = false;

    double* d_schur_rhs = nullptr;
    double* d_schur_x = nullptr;
    int schur_buffer_size = 0;

    int* d_interior_globals = nullptr;
    int* d_interface_globals = nullptr;
    double* d_interior_rhs = nullptr;
    double* d_interior_y = nullptr;
    double* d_interior_x = nullptr;
    int interior_buffer_size = 0;

    int mc_nnz = 0;
    int* d_mc_rows = nullptr;
    int* d_mc_cols = nullptr;
    double* d_mc_vals = nullptr;

    int mb_nnz = 0;
    int* d_mb_rows = nullptr;
    int* d_mb_cols = nullptr;
    double* d_mb_vals = nullptr;

    double* d_s_inv = nullptr;

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

    ~SolverContext() {
        if (d_schur_rhs) cudaFree(d_schur_rhs);
        if (d_schur_x) cudaFree(d_schur_x);
        if (d_interior_globals) cudaFree(d_interior_globals);
        if (d_interface_globals) cudaFree(d_interface_globals);
        if (d_interior_rhs) cudaFree(d_interior_rhs);
        if (d_interior_y) cudaFree(d_interior_y);
        if (d_interior_x) cudaFree(d_interior_x);
        if (d_mc_rows) cudaFree(d_mc_rows);
        if (d_mc_cols) cudaFree(d_mc_cols);
        if (d_mc_vals) cudaFree(d_mc_vals);
        if (d_mb_rows) cudaFree(d_mb_rows);
        if (d_mb_cols) cudaFree(d_mb_cols);
        if (d_mb_vals) cudaFree(d_mb_vals);
        if (d_s_inv) cudaFree(d_s_inv);
        if (d_source_replace_idxs) cudaFree(d_source_replace_idxs);
        if (d_source_replace_values) cudaFree(d_source_replace_values);
        if (d_source_addback_idxs) cudaFree(d_source_addback_idxs);
        if (d_source_addback_values) cudaFree(d_source_addback_values);
    }

    void ensure_schur_buffers(int size) {
        if (schur_buffer_size >= size) return;
        if (d_schur_rhs) cudaFree(d_schur_rhs);
        if (d_schur_x) cudaFree(d_schur_x);
        cudaMalloc(&d_schur_rhs, size * sizeof(double));
        cudaMalloc(&d_schur_x, size * sizeof(double));
        schur_buffer_size = size;
    }

    void ensure_interior_buffers(int n) {
        if (interior_buffer_size >= n) return;
        if (d_interior_rhs) cudaFree(d_interior_rhs);
        if (d_interior_y) cudaFree(d_interior_y);
        if (d_interior_x) cudaFree(d_interior_x);
        if (n > 0) {
            cudaMalloc(&d_interior_rhs, n * sizeof(double));
            cudaMalloc(&d_interior_y, n * sizeof(double));
            cudaMalloc(&d_interior_x, n * sizeof(double));
        } else {
            d_interior_rhs = nullptr;
            d_interior_y = nullptr;
            d_interior_x = nullptr;
        }
        interior_buffer_size = n;
    }
};

psum_field_solver_handle init() {
    return new SolverContext();
}

template<typename T>
void write_binary(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!os) {
        throw std::runtime_error("cuda_schurcomplement_gpu: failed writing state file");
    }
}

template<typename T>
void read_binary(std::istream& is, T& value) {
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!is) {
        throw std::runtime_error("cuda_schurcomplement_gpu: failed reading state file");
    }
}

void write_int_vector(std::ostream& os, const std::vector<int>& values) {
    std::uint64_t size = values.size();
    write_binary(os, size);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(values.data()), sizeof(int) * size);
    }
}

std::vector<int> read_int_vector(std::istream& is) {
    std::uint64_t size = 0;
    read_binary(is, size);
    std::vector<int> values(size);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(values.data()), sizeof(int) * size);
        if (!is) {
            throw std::runtime_error("cuda_schurcomplement_gpu: failed reading state vector");
        }
    }
    return values;
}

void write_pair_vector(std::ostream& os, const std::vector<std::pair<int, int>>& values) {
    std::uint64_t size = values.size();
    write_binary(os, size);
    for (const auto& [first, second] : values) {
        write_binary(os, first);
        write_binary(os, second);
    }
}

std::vector<std::pair<int, int>> read_pair_vector(std::istream& is) {
    std::uint64_t size = 0;
    read_binary(is, size);
    std::vector<std::pair<int, int>> values(size);
    for (auto& [first, second] : values) {
        read_binary(is, first);
        read_binary(is, second);
    }
    return values;
}

void write_sparse_matrix(std::ostream& os, const Eigen::SparseMatrix<double>& matrix) {
    Eigen::SparseMatrix<double> compressed(matrix);
    compressed.makeCompressed();
    const int rows = static_cast<int>(compressed.rows());
    const int cols = static_cast<int>(compressed.cols());
    const int nnz = static_cast<int>(compressed.nonZeros());
    write_binary(os, rows);
    write_binary(os, cols);
    write_binary(os, nnz);
    os.write(reinterpret_cast<const char*>(compressed.outerIndexPtr()), sizeof(int) * (cols + 1));
    os.write(reinterpret_cast<const char*>(compressed.innerIndexPtr()), sizeof(int) * nnz);
    os.write(reinterpret_cast<const char*>(compressed.valuePtr()), sizeof(double) * nnz);
    if (!os) {
        throw std::runtime_error("cuda_schurcomplement_gpu: failed writing sparse matrix state");
    }
}

Eigen::SparseMatrix<double> read_sparse_matrix(std::istream& is) {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    read_binary(is, rows);
    read_binary(is, cols);
    read_binary(is, nnz);
    std::vector<int> outer(cols + 1);
    std::vector<int> inner(nnz);
    std::vector<double> values(nnz);
    is.read(reinterpret_cast<char*>(outer.data()), sizeof(int) * (cols + 1));
    is.read(reinterpret_cast<char*>(inner.data()), sizeof(int) * nnz);
    is.read(reinterpret_cast<char*>(values.data()), sizeof(double) * nnz);
    if (!is) {
        throw std::runtime_error("cuda_schurcomplement_gpu: failed reading sparse matrix state");
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    for (int col = 0; col < cols; ++col) {
        for (int k = outer[col]; k < outer[col + 1]; ++k) {
            triplets.emplace_back(inner[k], col, values[k]);
        }
    }
    Eigen::SparseMatrix<double> matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

void write_dense_matrix(std::ostream& os, const Eigen::MatrixXd& matrix) {
    const int rows = static_cast<int>(matrix.rows());
    const int cols = static_cast<int>(matrix.cols());
    write_binary(os, rows);
    write_binary(os, cols);
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    if (size > 0) {
        os.write(reinterpret_cast<const char*>(matrix.data()), sizeof(double) * size);
        if (!os) {
            throw std::runtime_error("cuda_schurcomplement_gpu: failed writing dense matrix state");
        }
    }
}

Eigen::MatrixXd read_dense_matrix(std::istream& is) {
    int rows = 0;
    int cols = 0;
    read_binary(is, rows);
    read_binary(is, cols);
    Eigen::MatrixXd matrix(rows, cols);
    const std::uint64_t size = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    if (size > 0) {
        is.read(reinterpret_cast<char*>(matrix.data()), sizeof(double) * size);
        if (!is) {
            throw std::runtime_error("cuda_schurcomplement_gpu: failed reading dense matrix state");
        }
    }
    return matrix;
}

void save_state_file(const SolverContext* ctx, const std::string& path) {
    if (!ctx->initialized) {
        throw std::runtime_error("cuda_schurcomplement_gpu: cannot save state before setup");
    }
    std::ofstream os(path, std::ios::binary);
    if (!os) {
        throw std::runtime_error("cuda_schurcomplement_gpu: cannot open state file for write");
    }

    const char magic[16] = {'P','S','U','M','S','C','H','U','R','S','T','A','T','E','1','\0'};
    os.write(magic, sizeof(magic));
    const std::uint32_t version = 1;
    write_binary(os, version);
    write_binary(os, ctx->n);
    write_binary(os, ctx->I);
    write_binary(os, ctx->J);
    write_binary(os, ctx->blocks_x);
    write_binary(os, ctx->blocks_y);
    write_binary(os, ctx->core.n);
    write_binary(os, ctx->core.n_interface);
    write_binary(os, ctx->core.n_interior);
    write_int_vector(os, ctx->interface_nodes);
    write_pair_vector(os, ctx->core.interface_map);
    write_pair_vector(os, ctx->core.interior_map);
    write_sparse_matrix(os, ctx->core.mat_A);
    write_sparse_matrix(os, ctx->core.mat_B);
    write_sparse_matrix(os, ctx->core.mat_C);
    write_dense_matrix(os, ctx->core.S);
    write_dense_matrix(os, ctx->core.S_inv);
}

void load_state_file(SolverContext* ctx, const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("cuda_schurcomplement_gpu: cannot open state file for read");
    }

    const char expected_magic[16] = {'P','S','U','M','S','C','H','U','R','S','T','A','T','E','1','\0'};
    char magic[16] = {};
    is.read(magic, sizeof(magic));
    if (!is || std::memcmp(magic, expected_magic, sizeof(magic)) != 0) {
        throw std::runtime_error("cuda_schurcomplement_gpu: invalid state file");
    }

    std::uint32_t version = 0;
    read_binary(is, version);
    if (version != 1) {
        throw std::runtime_error("cuda_schurcomplement_gpu: unsupported state file version");
    }

    read_binary(is, ctx->n);
    read_binary(is, ctx->I);
    read_binary(is, ctx->J);
    read_binary(is, ctx->blocks_x);
    read_binary(is, ctx->blocks_y);
    read_binary(is, ctx->core.n);
    read_binary(is, ctx->core.n_interface);
    read_binary(is, ctx->core.n_interior);
    ctx->interface_nodes = read_int_vector(is);
    ctx->core.interface_map = read_pair_vector(is);
    ctx->core.interior_map = read_pair_vector(is);
    ctx->core.mat_A = read_sparse_matrix(is);
    ctx->core.mat_B = read_sparse_matrix(is);
    ctx->core.mat_C = read_sparse_matrix(is);
    ctx->core.S = read_dense_matrix(is);
    ctx->core.S_inv = read_dense_matrix(is);
    if (!is) {
        throw std::runtime_error("cuda_schurcomplement_gpu: failed reading state file");
    }
    ctx->loaded_state = true;
}

int infer_square_size(unsigned long long n) {
    const auto root = static_cast<unsigned long long>(std::llround(std::sqrt(static_cast<double>(n))));
    if (root * root != n) {
        throw std::runtime_error("cuda_schurcomplement_gpu: I and J must be provided for non-square systems");
    }
    return static_cast<int>(root);
}

void ensure_grid_shape(SolverContext* ctx) {
    if (ctx->I <= 0 && ctx->J <= 0) {
        ctx->I = infer_square_size(ctx->n);
        ctx->J = ctx->I;
    } else if (ctx->I <= 0) {
        if (ctx->n % static_cast<unsigned long long>(ctx->J) != 0) {
            throw std::runtime_error("cuda_schurcomplement_gpu: invalid J option for matrix size");
        }
        ctx->I = static_cast<int>(ctx->n / static_cast<unsigned long long>(ctx->J));
    } else if (ctx->J <= 0) {
        if (ctx->n % static_cast<unsigned long long>(ctx->I) != 0) {
            throw std::runtime_error("cuda_schurcomplement_gpu: invalid I option for matrix size");
        }
        ctx->J = static_cast<int>(ctx->n / static_cast<unsigned long long>(ctx->I));
    }

    if (static_cast<unsigned long long>(ctx->I) * static_cast<unsigned long long>(ctx->J) != ctx->n) {
        throw std::runtime_error("cuda_schurcomplement_gpu: I * J must equal matrix dimension");
    }

    ctx->blocks_x = std::max(1, std::min(ctx->blocks_x, ctx->I));
    ctx->blocks_y = std::max(1, std::min(ctx->blocks_y, ctx->J));
}

int block_of_node(const SolverContext* ctx, int global) {
    const int i = global / ctx->J;
    const int j = global % ctx->J;
    const int bx = std::min(ctx->blocks_x - 1, (i * ctx->blocks_x) / ctx->I);
    const int by = std::min(ctx->blocks_y - 1, (j * ctx->blocks_y) / ctx->J);
    return bx * ctx->blocks_y + by;
}

void solve_interior(SolverContext* ctx, double* d_b, double* d_x) {
    if (!ctx->has_interior_lu) {
        return;
    }
    ctx->interior_lu.solve(d_b, d_x);
}

void discover_interface(SolverContext* ctx) {
    const int n = static_cast<int>(ctx->n);

    std::vector<int> node_block(n);
    for (int g = 0; g < n; ++g) {
        node_block[g] = block_of_node(ctx, g);
    }

    std::vector<char> is_interface(n, 0);
    for (int col = 0; col < ctx->A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(ctx->A, col); it; ++it) {
            const int row = it.row();
            if (row == col) {
                continue;
            }
            if (node_block[row] != node_block[col]) {
                const int interface_node = node_block[row] > node_block[col] ? row : col;
                is_interface[interface_node] = 1;
            }
        }
    }

    ctx->interface_nodes.clear();
    for (int g = 0; g < n; ++g) {
        if (is_interface[g]) {
            ctx->interface_nodes.push_back(g);
        }
    }
}

template<typename SparseMat>
void upload_sparse_coo(const SparseMat& mat, int& nnz_out, int*& d_rows, int*& d_cols, double*& d_vals) {
    if (d_rows) { cudaFree(d_rows); d_rows = nullptr; }
    if (d_cols) { cudaFree(d_cols); d_cols = nullptr; }
    if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }

    nnz_out = static_cast<int>(mat.nonZeros());
    if (nnz_out == 0) return;

    std::vector<int> h_rows(nnz_out), h_cols(nnz_out);
    std::vector<double> h_vals(nnz_out);
    int k = 0;
    for (int col = 0; col < mat.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat, col); it; ++it) {
            h_rows[k] = static_cast<int>(it.row());
            h_cols[k] = static_cast<int>(it.col());
            h_vals[k] = it.value();
            ++k;
        }
    }
    cudaMalloc(&d_rows, sizeof(int) * nnz_out);
    cudaMalloc(&d_cols, sizeof(int) * nnz_out);
    cudaMalloc(&d_vals, sizeof(double) * nnz_out);
    cudaMemcpy(d_rows, h_rows.data(), sizeof(int) * nnz_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, h_cols.data(), sizeof(int) * nnz_out, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, h_vals.data(), sizeof(double) * nnz_out, cudaMemcpyHostToDevice);
}

template<typename SparseMat>
void upload_sparse_csc(const SparseMat& mat, int*& d_outer, int*& d_inner, double*& d_vals) {
    if (d_outer) { cudaFree(d_outer); d_outer = nullptr; }
    if (d_inner) { cudaFree(d_inner); d_inner = nullptr; }
    if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }

    const int cols = static_cast<int>(mat.cols());
    const int nnz = static_cast<int>(mat.nonZeros());
    cudaMalloc(&d_outer, sizeof(int) * (cols + 1));
    cudaMemcpy(d_outer, mat.outerIndexPtr(), sizeof(int) * (cols + 1), cudaMemcpyHostToDevice);
    if (nnz == 0) {
        return;
    }

    cudaMalloc(&d_inner, sizeof(int) * nnz);
    cudaMalloc(&d_vals, sizeof(double) * nnz);
    cudaMemcpy(d_inner, mat.innerIndexPtr(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, mat.valuePtr(), sizeof(double) * nnz, cudaMemcpyHostToDevice);
}

void compute_schur_contribution_gpu(SolverContext* ctx) {
    const int n_interior = ctx->core.n_interior;
    const int n_interface = ctx->core.n_interface;
    if (n_interior == 0 || n_interface == 0) {
        return;
    }

    SparseLUCuda contribution_lu;
    contribution_lu.set(ctx->core.mat_A);

    int* d_b_outer = nullptr;
    int* d_b_inner = nullptr;
    double* d_b_vals = nullptr;
    upload_sparse_csc(ctx->core.mat_B, d_b_outer, d_b_inner, d_b_vals);

    int c_nnz = 0;
    int* d_c_rows = nullptr;
    int* d_c_cols = nullptr;
    double* d_c_vals = nullptr;
    upload_sparse_coo(ctx->core.mat_C, c_nnz, d_c_rows, d_c_cols, d_c_vals);

    double* d_rhs = nullptr;
    double* d_w = nullptr;
    double* d_s = nullptr;
    const std::size_t s_size = static_cast<std::size_t>(n_interface) * static_cast<std::size_t>(n_interface);
    cudaMalloc(&d_rhs, sizeof(double) * n_interior);
    cudaMalloc(&d_w, sizeof(double) * n_interior);
    cudaMalloc(&d_s, sizeof(double) * s_size);
    cudaMemcpy(d_s, ctx->core.S.data(), sizeof(double) * s_size, cudaMemcpyHostToDevice);

    const int block_size = 256;
    for (int col = 0; col < n_interface; ++col) {
        cudaMemset(d_rhs, 0, sizeof(double) * n_interior);
        load_sparse_column_kernel<<<32, block_size>>>(col, d_b_outer, d_b_inner, d_b_vals, d_rhs);
        contribution_lu.solve(d_rhs, d_w);
        if (c_nnz > 0) {
            const int grid_size = (c_nnz + block_size - 1) / block_size;
            subtract_matvec_kernel<<<grid_size, block_size>>>(c_nnz, d_c_rows, d_c_cols, d_c_vals,
                                                              d_w, d_s + static_cast<std::size_t>(col) * n_interface);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(ctx->core.S.data(), d_s, sizeof(double) * s_size, cudaMemcpyDeviceToHost);

    cudaFree(d_rhs);
    cudaFree(d_w);
    cudaFree(d_s);
    if (d_b_outer) cudaFree(d_b_outer);
    if (d_b_inner) cudaFree(d_b_inner);
    if (d_b_vals) cudaFree(d_b_vals);
    if (d_c_rows) cudaFree(d_c_rows);
    if (d_c_cols) cudaFree(d_c_cols);
    if (d_c_vals) cudaFree(d_c_vals);

    ctx->core.finalize_inverse();
}

void upload_device_data(SolverContext* ctx) {
    const int n_interior = ctx->core.n_interior;
    const int n_interface = ctx->core.n_interface;

    if (ctx->d_interior_globals) { cudaFree(ctx->d_interior_globals); ctx->d_interior_globals = nullptr; }
    if (n_interior > 0) {
        std::vector<int> ig(n_interior);
        for (int i = 0; i < n_interior; ++i) {
            ig[i] = ctx->core.interior_map[i].first;
        }
        cudaMalloc(&ctx->d_interior_globals, sizeof(int) * n_interior);
        cudaMemcpy(ctx->d_interior_globals, ig.data(), sizeof(int) * n_interior, cudaMemcpyHostToDevice);
    }

    if (ctx->d_interface_globals) { cudaFree(ctx->d_interface_globals); ctx->d_interface_globals = nullptr; }
    if (n_interface > 0) {
        std::vector<int> ig(n_interface);
        for (int i = 0; i < n_interface; ++i) {
            ig[i] = ctx->core.interface_map[i].first;
        }
        cudaMalloc(&ctx->d_interface_globals, sizeof(int) * n_interface);
        cudaMemcpy(ctx->d_interface_globals, ig.data(), sizeof(int) * n_interface, cudaMemcpyHostToDevice);
    }

    upload_sparse_coo(ctx->core.mat_C, ctx->mc_nnz, ctx->d_mc_rows, ctx->d_mc_cols, ctx->d_mc_vals);
    upload_sparse_coo(ctx->core.mat_B, ctx->mb_nnz, ctx->d_mb_rows, ctx->d_mb_cols, ctx->d_mb_vals);

    if (ctx->d_s_inv) { cudaFree(ctx->d_s_inv); ctx->d_s_inv = nullptr; }
    if (n_interface > 0) {
        const size_t s_size = static_cast<size_t>(n_interface) * static_cast<size_t>(n_interface);
        cudaMalloc(&ctx->d_s_inv, sizeof(double) * s_size);
        cudaMemcpy(ctx->d_s_inv, ctx->core.S_inv.data(), sizeof(double) * s_size, cudaMemcpyHostToDevice);
        ctx->ensure_schur_buffers(n_interface);
    }
}

void rebuild_runtime_state(SolverContext* ctx) {
    ctx->has_interior_lu = ctx->core.n_interior > 0;
    if (ctx->has_interior_lu) {
        ctx->interior_lu.set(ctx->core.mat_A);
        ctx->ensure_interior_buffers(ctx->core.n_interior);
    }

    upload_device_data(ctx);
    ctx->initialized = true;
}

void set_matrix(psum_field_solver_handle h, unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
    if (n_row != n_col) {
        throw std::runtime_error("cuda_schurcomplement_gpu: matrix must be square");
    }

    SolverContext* ctx = static_cast<SolverContext*>(h);
    if (ctx->loaded_state) {
        if (ctx->n != n_row || ctx->n != n_col) {
            throw std::runtime_error("cuda_schurcomplement_gpu: loaded state dimension does not match matrix");
        }
        return;
    }

    ctx->n = n_row;
    ensure_grid_shape(ctx);

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);
    for (unsigned long long i = 0; i < nnz; ++i) {
        triplets.emplace_back(static_cast<int>(rows[i]), static_cast<int>(cols[i]), vals[i]);
    }

    ctx->A.resize(static_cast<int>(n_row), static_cast<int>(n_col));
    ctx->A.setFromTriplets(triplets.begin(), triplets.end());
    ctx->A.makeCompressed();

    discover_interface(ctx);

    ctx->core.setup(ctx->A, ctx->interface_nodes, ctx->num_threads,
                    schur_complement::schur_contribution_backend::external);

    compute_schur_contribution_gpu(ctx);

    rebuild_runtime_state(ctx);
    ctx->loaded_state = false;
}

void solve(psum_field_solver_handle h, double* b, double* x) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    if (!ctx->initialized) {
        throw std::runtime_error("cuda_schurcomplement_gpu: solver not initialized");
    }

    const int block_size = 256;
    const int n_interior = ctx->core.n_interior;
    const int n_interface = ctx->core.n_interface;

    if (ctx->source_addback_size > 0) {
        const int grid_size = (ctx->source_addback_size + block_size - 1) / block_size;
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(ctx->source_addback_size, ctx->d_source_addback_idxs, ctx->d_source_addback_values, b);
    }

    if (ctx->source_replace_size > 0) {
        const int grid_size = (ctx->source_replace_size + block_size - 1) / block_size;
        apply_source_clear_kernel<<<grid_size, block_size>>>(ctx->source_replace_size, ctx->d_source_replace_idxs, b);
        apply_source_accumulate_kernel<<<grid_size, block_size>>>(ctx->source_replace_size, ctx->d_source_replace_idxs, ctx->d_source_replace_values, b);
    }

    if (n_interior > 0) {
        const int grid_size = (n_interior + block_size - 1) / block_size;
        gather_by_global_index_kernel<<<grid_size, block_size>>>(n_interior, ctx->d_interior_globals, b, ctx->d_interior_rhs);
    }
    if (n_interface > 0) {
        const int grid_size = (n_interface + block_size - 1) / block_size;
        gather_by_global_index_kernel<<<grid_size, block_size>>>(n_interface, ctx->d_interface_globals, b, ctx->d_schur_rhs);
    }

    cudaDeviceSynchronize();
    Tic("cuda_schurcomplement_gpu.interior1");
    if (ctx->has_interior_lu) {
        solve_interior(ctx, ctx->d_interior_rhs, ctx->d_interior_y);
    }
    cudaDeviceSynchronize();
    Toc_("cuda_schurcomplement_gpu.interior1");

    if (n_interface > 0 && ctx->mc_nnz > 0) {
        const int grid_size = (ctx->mc_nnz + block_size - 1) / block_size;
        subtract_matvec_kernel<<<grid_size, block_size>>>(ctx->mc_nnz, ctx->d_mc_rows, ctx->d_mc_cols,
                                                          ctx->d_mc_vals, ctx->d_interior_y, ctx->d_schur_rhs);
    }

    if (n_interface > 0) {
        const int grid_size = (n_interface + block_size - 1) / block_size;
        dense_matvec_colmajor_kernel<<<grid_size, block_size>>>(n_interface, n_interface, ctx->d_s_inv, ctx->d_schur_rhs, ctx->d_schur_x);
    }

    if (n_interior > 0) {
        cudaMemcpy(ctx->d_interior_x, ctx->d_interior_rhs, n_interior * sizeof(double), cudaMemcpyDeviceToDevice);

        if (n_interface > 0 && ctx->mb_nnz > 0) {
            const int grid_size = (ctx->mb_nnz + block_size - 1) / block_size;
            subtract_matvec_kernel<<<grid_size, block_size>>>(ctx->mb_nnz, ctx->d_mb_rows, ctx->d_mb_cols,
                                                              ctx->d_mb_vals, ctx->d_schur_x, ctx->d_interior_x);
        }

        cudaDeviceSynchronize();
        Tic("cuda_schurcomplement_gpu.interior2");
        solve_interior(ctx, ctx->d_interior_x, ctx->d_interior_x);
        cudaDeviceSynchronize();
        Toc_("cuda_schurcomplement_gpu.interior2");

        const int grid_size = (n_interior + block_size - 1) / block_size;
        scatter_by_global_index_kernel<<<grid_size, block_size>>>(n_interior, ctx->d_interior_globals, ctx->d_interior_x, x);
        if (n_interface > 0) {
            const int interface_grid_size = (n_interface + block_size - 1) / block_size;
            scatter_by_global_index_kernel<<<interface_grid_size, block_size>>>(n_interface, ctx->d_interface_globals, ctx->d_schur_x, x);
        }
    } else {
        if (n_interface > 0) {
            const int grid_size = (n_interface + block_size - 1) / block_size;
            scatter_by_global_index_kernel<<<grid_size, block_size>>>(n_interface, ctx->d_interface_globals, ctx->d_schur_x, x);
        }
    }

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cuda_schurcomplement_gpu: solve synchronization failed: ") +
                                 cudaGetErrorString(err));
    }
}

void set_options(psum_field_solver_handle h, const char* options) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    std::istringstream iss(options);
    std::string key;
    while (iss >> key) {
        if (key == "I") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("cuda_schurcomplement_gpu: I must be set before set_matrix");
            }
            ctx->I = value;
        } else if (key == "J") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("cuda_schurcomplement_gpu: J must be set before set_matrix");
            }
            ctx->J = value;
        } else if (key == "K") {
            int ignored;
            iss >> ignored;
        } else if (key == "blocks_x") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("cuda_schurcomplement_gpu: blocks_x must be set before set_matrix");
            }
            ctx->blocks_x = value;
        } else if (key == "blocks_y") {
            int value;
            iss >> value;
            if (ctx->initialized) {
                throw std::runtime_error("cuda_schurcomplement_gpu: blocks_y must be set before set_matrix");
            }
            ctx->blocks_y = value;
        } else if (key == "num_threads") {
            iss >> ctx->num_threads;
        } else if (key == "save_state") {
            std::string path;
            iss >> path;
            if (path.empty()) {
                throw std::runtime_error("cuda_schurcomplement_gpu: save_state requires a path");
            }
            save_state_file(ctx, path);
        } else if (key == "load_state") {
            std::string path;
            iss >> path;
            if (path.empty()) {
                throw std::runtime_error("cuda_schurcomplement_gpu: load_state requires a path");
            }
            load_state_file(ctx, path);
            rebuild_runtime_state(ctx);
        }
    }
}

void set_source_replace(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* new_values) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    ctx->source_replace_idxs.clear();
    ctx->source_replace_values.clear();
    ctx->source_replace_idxs.reserve(size);
    ctx->source_replace_values.reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        ctx->source_replace_idxs.push_back(idxs[i]);
        ctx->source_replace_values.push_back(new_values[i]);
    }

    if (ctx->d_source_replace_idxs) {
        cudaFree(ctx->d_source_replace_idxs);
        ctx->d_source_replace_idxs = nullptr;
    }
    if (ctx->d_source_replace_values) {
        cudaFree(ctx->d_source_replace_values);
        ctx->d_source_replace_values = nullptr;
    }

    ctx->source_replace_size = size;
    if (size > 0) {
        cudaMalloc(&ctx->d_source_replace_idxs, size * sizeof(unsigned long long));
        cudaMalloc(&ctx->d_source_replace_values, size * sizeof(double));
        cudaMemcpy(ctx->d_source_replace_idxs, idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_source_replace_values, new_values, size * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void set_source_addback(psum_field_solver_handle h, unsigned long long size, unsigned long long* idxs, double* add_values) {
    SolverContext* ctx = static_cast<SolverContext*>(h);
    ctx->source_addback_idxs.clear();
    ctx->source_addback_values.clear();
    ctx->source_addback_idxs.reserve(size);
    ctx->source_addback_values.reserve(size);
    for (unsigned long long i = 0; i < size; ++i) {
        ctx->source_addback_idxs.push_back(idxs[i]);
        ctx->source_addback_values.push_back(add_values[i]);
    }

    if (ctx->d_source_addback_idxs) {
        cudaFree(ctx->d_source_addback_idxs);
        ctx->d_source_addback_idxs = nullptr;
    }
    if (ctx->d_source_addback_values) {
        cudaFree(ctx->d_source_addback_values);
        ctx->d_source_addback_values = nullptr;
    }

    ctx->source_addback_size = size;
    if (size > 0) {
        cudaMalloc(&ctx->d_source_addback_idxs, size * sizeof(unsigned long long));
        cudaMalloc(&ctx->d_source_addback_values, size * sizeof(double));
        cudaMemcpy(ctx->d_source_addback_idxs, idxs, size * sizeof(unsigned long long), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_source_addback_values, add_values, size * sizeof(double), cudaMemcpyHostToDevice);
    }
}

void solver_free(psum_field_solver_handle h) {
    delete static_cast<SolverContext*>(h);
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
    register_solver("cuda_schurcomplement_gpu", table);
    return 0;
}();

}

}

}

}
