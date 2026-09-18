#include "../register/interface.h"
#include "multigrid/multigrid_utilities.hpp"
#include "multigrid/multigrid_core.hpp"
#include "multigrid/convSparseMatrix.hpp"
#include "multigrid/cuda_vector.cuh"
#include "multigrid/conv_cuda_sparse_matrix.cuh"
#include "multigrid/cuda_inner_solver.cuh"
#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <Eigen/SparseCore>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace psum {

namespace field_solver {

namespace implements {

namespace cuda_multigrid_gpu {

    __global__ void apply_source_replace_kernel(unsigned long long size,
                                                const unsigned long long* idxs,
                                                const double* values,
                                                double* b) {
        unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            b[idxs[i]] = values[i];
        }
    }

    __global__ void apply_source_addback_kernel(unsigned long long size,
                                                const unsigned long long* idxs,
                                                const double* values,
                                                double* b) {
        unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            atomicAdd(&b[idxs[i]], values[i]);
        }
    }

    struct SolverContext {
        cudaStream_t stream = nullptr;
        cublasHandle_t cublas_handle = nullptr;
        cusparseHandle_t cusparse_handle = nullptr;
        multigrid::multigrid_core<multigrid::ConvCudaSparseMatrix<double>,
                                  multigrid::cuda_inner_solver> core;
        unsigned long long n_row = 0;
        int I = 0, J = 0, K = 0;
        int depth = 3;
        int pre_relax = 2;
        int post_relax = 1;
        int max_cycles = 10;
        int num_conv_mask = 5;
        int num_conv_mask_relax = 5;
        int num_conv_mask_restrict = 5;
        int num_conv_mask_interpolate = 5;
        int num_conv_mask_a = 5;
        Eigen::SparseMatrix<double> matrix;
        std::vector<unsigned long long> source_replace_idxs;
        std::vector<double> source_replace_values;
        unsigned long long* d_source_replace_idxs = nullptr;
        double* d_source_replace_values = nullptr;
        unsigned long long source_replace_size = 0;
        unsigned long long source_replace_capacity = 0;
        std::vector<unsigned long long> source_addback_idxs;
        std::vector<double> source_addback_values;
        unsigned long long* d_source_addback_idxs = nullptr;
        double* d_source_addback_values = nullptr;
        unsigned long long source_addback_size = 0;
        unsigned long long source_addback_capacity = 0;
    };

    void set_active_context_(SolverContext* ctx) {
        multigrid::cuda_vector::default_stream_ = ctx->stream;
        multigrid::cuda_vector::default_cublas_handle_ = ctx->cublas_handle;
        multigrid::cuda_vector::default_cusparse_handle_ = ctx->cusparse_handle;
    }

    psum_field_solver_handle init() {
        int device_count = 0;
        multigrid::cuda_multigrid_check(cudaGetDeviceCount(&device_count),
                                        "cuda_multigrid_gpu: cudaGetDeviceCount failed");
        if (device_count <= 0) {
            throw std::runtime_error("cuda_multigrid_gpu: no CUDA device available");
        }

        SolverContext* ctx = new SolverContext();
        try {
            multigrid::cuda_multigrid_check(cublasCreate(&ctx->cublas_handle),
                                            "cuda_multigrid_gpu: cublasCreate failed");
            multigrid::cuda_multigrid_check(cusparseCreate(&ctx->cusparse_handle),
                                            "cuda_multigrid_gpu: cusparseCreate failed");
            set_active_context_(ctx);
        } catch (...) {
            if (ctx->cusparse_handle) cusparseDestroy(ctx->cusparse_handle);
            if (ctx->cublas_handle) cublasDestroy(ctx->cublas_handle);
            if (ctx->stream) cudaStreamDestroy(ctx->stream);
            delete ctx;
            throw;
        }
        return ctx;
    }

    void set_matrix(psum_field_solver_handle h, unsigned long long n_row,
                    unsigned long long n_col, unsigned long long nnz,
                    unsigned long long* rows, unsigned long long* cols, double* vals) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        ctx->n_row = n_row;

        Eigen::SparseMatrix<double> A(n_row, n_col);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nnz);
        for (unsigned long long i = 0; i < nnz; ++i)
            triplets.emplace_back(rows[i], cols[i], vals[i]);
        A.setFromTriplets(triplets.begin(), triplets.end());
        ctx->matrix = A;

        set_active_context_(ctx);

        std::vector<Eigen::SparseMatrix<double>> RMats;
        if (ctx->K <= 1)
            RMats = multigrid::get_restriction_matrices_2d(ctx->I, ctx->J, ctx->depth);
        else
            RMats = multigrid::get_restriction_matrices_3d(ctx->I, ctx->J, ctx->K, ctx->depth);

        double c_coeff = ctx->K <= 1 ? 2.0 : 8.0;
        ctx->core.setup(A, RMats, c_coeff);
        ctx->core.set_relax_counts(ctx->pre_relax, ctx->post_relax);

        for (size_t i = 0; i < ctx->core.relax_mats_.size(); i++)
            ctx->core.relax_mats_[i].set(ctx->core.relax_mats_[i].original_mat_, ctx->num_conv_mask_relax);
        for (size_t i = 0; i < ctx->core.restrict_mats_.size(); i++) {
            ctx->core.restrict_mats_[i].set(ctx->core.restrict_mats_[i].original_mat_, ctx->num_conv_mask_restrict);
            ctx->core.interpolate_mats_[i].set(ctx->core.interpolate_mats_[i].original_mat_, ctx->num_conv_mask_interpolate);
        }
        for (size_t i = 0; i < ctx->core.a_mats_.size(); i++)
            ctx->core.a_mats_[i].set(ctx->core.a_mats_[i].original_mat_, ctx->num_conv_mask_a);
    }

    void solve(psum_field_solver_handle h, double* b, double* x) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        set_active_context_(ctx);

        const int threads = 256;
        if (ctx->source_addback_size > 0) {
            unsigned int blocks = static_cast<unsigned int>((ctx->source_addback_size + threads - 1) / threads);
            apply_source_addback_kernel<<<blocks, threads, 0, ctx->stream>>>(
                ctx->source_addback_size, ctx->d_source_addback_idxs, ctx->d_source_addback_values, b);
            multigrid::cuda_multigrid_check(cudaGetLastError(),
                                            "cuda_multigrid_gpu: source addback launch failed");
        }

        if (ctx->source_replace_size > 0) {
            unsigned int blocks = static_cast<unsigned int>((ctx->source_replace_size + threads - 1) / threads);
            apply_source_replace_kernel<<<blocks, threads, 0, ctx->stream>>>(
                ctx->source_replace_size, ctx->d_source_replace_idxs, ctx->d_source_replace_values, b);
            multigrid::cuda_multigrid_check(cudaGetLastError(),
                                            "cuda_multigrid_gpu: source replace launch failed");
        }

        multigrid::cuda_vector d_b, d_x;
        d_b.data_ = b;
        d_b.size_ = static_cast<int>(ctx->n_row);
        d_x.data_ = x;
        d_x.size_ = static_cast<int>(ctx->n_row);

        try {
            ctx->core.set_relax_counts(ctx->pre_relax, ctx->post_relax);
            ctx->core.v_cycle_multi(d_b, d_x, ctx->max_cycles);
            multigrid::cuda_multigrid_check(cudaStreamSynchronize(ctx->stream),
                                            "cuda_multigrid_gpu: solve sync failed");

            d_b.data_ = nullptr;
            d_x.data_ = nullptr;
        } catch (...) {
            d_b.data_ = nullptr;
            d_x.data_ = nullptr;
            throw;
        }
    }

    void set_options(psum_field_solver_handle h, const char* options) {
        SolverContext* ctx = static_cast<SolverContext*>(h);

        std::istringstream iss(options);
        std::string key;
        bool k_provided = false;
        while (iss >> key) {
            if (key == "I") iss >> ctx->I;
            else if (key == "J") iss >> ctx->J;
            else if (key == "K") { iss >> ctx->K; k_provided = true; }
            else if (key == "depth") iss >> ctx->depth;
            else if (key == "pre_relax") iss >> ctx->pre_relax;
            else if (key == "post_relax") iss >> ctx->post_relax;
            else if (key == "max_cycles") iss >> ctx->max_cycles;
            else if (key == "num_conv_mask") {
                iss >> ctx->num_conv_mask;
                ctx->num_conv_mask_relax = ctx->num_conv_mask;
                ctx->num_conv_mask_restrict = ctx->num_conv_mask;
                ctx->num_conv_mask_interpolate = ctx->num_conv_mask;
                ctx->num_conv_mask_a = ctx->num_conv_mask;
            }
            else if (key == "num_conv_mask_relax") iss >> ctx->num_conv_mask_relax;
            else if (key == "num_conv_mask_restrict") iss >> ctx->num_conv_mask_restrict;
            else if (key == "num_conv_mask_interpolate") iss >> ctx->num_conv_mask_interpolate;
            else if (key == "num_conv_mask_a") iss >> ctx->num_conv_mask_a;
        }
        if (!k_provided)
            ctx->K = 1;
    }

    void free_source_device_storage_(SolverContext* ctx) {
        if (ctx->d_source_replace_idxs) cudaFree(ctx->d_source_replace_idxs);
        if (ctx->d_source_replace_values) cudaFree(ctx->d_source_replace_values);
        if (ctx->d_source_addback_idxs) cudaFree(ctx->d_source_addback_idxs);
        if (ctx->d_source_addback_values) cudaFree(ctx->d_source_addback_values);
        ctx->d_source_replace_idxs = nullptr;
        ctx->d_source_replace_values = nullptr;
        ctx->d_source_addback_idxs = nullptr;
        ctx->d_source_addback_values = nullptr;
        ctx->source_replace_capacity = 0;
        ctx->source_addback_capacity = 0;
    }

    void upload_source_replace_to_device_(SolverContext* ctx) {
        ctx->source_replace_size = ctx->source_replace_idxs.size();
        if (ctx->source_replace_size > 0) {
            if (ctx->source_replace_size > ctx->source_replace_capacity) {
                if (ctx->d_source_replace_idxs) cudaFree(ctx->d_source_replace_idxs);
                if (ctx->d_source_replace_values) cudaFree(ctx->d_source_replace_values);
                multigrid::cuda_multigrid_check(
                    cudaMalloc(reinterpret_cast<void**>(&ctx->d_source_replace_idxs),
                               sizeof(unsigned long long) * ctx->source_replace_size),
                    "cuda_multigrid_gpu: source replace index allocation failed");
                multigrid::cuda_multigrid_check(
                    cudaMalloc(reinterpret_cast<void**>(&ctx->d_source_replace_values),
                               sizeof(double) * ctx->source_replace_size),
                    "cuda_multigrid_gpu: source replace value allocation failed");
                ctx->source_replace_capacity = ctx->source_replace_size;
            }
            multigrid::cuda_multigrid_check(
                cudaMemcpyAsync(ctx->d_source_replace_idxs, ctx->source_replace_idxs.data(),
                                sizeof(unsigned long long) * ctx->source_replace_size,
                                cudaMemcpyHostToDevice, ctx->stream),
                "cuda_multigrid_gpu: source replace index upload failed");
            multigrid::cuda_multigrid_check(
                cudaMemcpyAsync(ctx->d_source_replace_values, ctx->source_replace_values.data(),
                                sizeof(double) * ctx->source_replace_size,
                                cudaMemcpyHostToDevice, ctx->stream),
                "cuda_multigrid_gpu: source replace value upload failed");
        }
    }

    void upload_source_addback_to_device_(SolverContext* ctx) {
        ctx->source_addback_size = ctx->source_addback_idxs.size();
        if (ctx->source_addback_size > 0) {
            if (ctx->source_addback_size > ctx->source_addback_capacity) {
                if (ctx->d_source_addback_idxs) cudaFree(ctx->d_source_addback_idxs);
                if (ctx->d_source_addback_values) cudaFree(ctx->d_source_addback_values);
                multigrid::cuda_multigrid_check(
                    cudaMalloc(reinterpret_cast<void**>(&ctx->d_source_addback_idxs),
                               sizeof(unsigned long long) * ctx->source_addback_size),
                    "cuda_multigrid_gpu: source addback index allocation failed");
                multigrid::cuda_multigrid_check(
                    cudaMalloc(reinterpret_cast<void**>(&ctx->d_source_addback_values),
                               sizeof(double) * ctx->source_addback_size),
                    "cuda_multigrid_gpu: source addback value allocation failed");
                ctx->source_addback_capacity = ctx->source_addback_size;
            }
            multigrid::cuda_multigrid_check(
                cudaMemcpyAsync(ctx->d_source_addback_idxs, ctx->source_addback_idxs.data(),
                                sizeof(unsigned long long) * ctx->source_addback_size,
                                cudaMemcpyHostToDevice, ctx->stream),
                "cuda_multigrid_gpu: source addback index upload failed");
            multigrid::cuda_multigrid_check(
                cudaMemcpyAsync(ctx->d_source_addback_values, ctx->source_addback_values.data(),
                                sizeof(double) * ctx->source_addback_size,
                                cudaMemcpyHostToDevice, ctx->stream),
                "cuda_multigrid_gpu: source addback value upload failed");
        }
    }

    void set_source_replace(psum_field_solver_handle h, unsigned long long size,
                            unsigned long long* idxs, double* new_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_replace_idxs.assign(idxs, idxs + size);
        ctx->source_replace_values.assign(new_values, new_values + size);
        upload_source_replace_to_device_(ctx);
    }

    void set_source_addback(psum_field_solver_handle h, unsigned long long size,
                            unsigned long long* idxs, double* add_values) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        ctx->source_addback_idxs.assign(idxs, idxs + size);
        ctx->source_addback_values.assign(add_values, add_values + size);
        upload_source_addback_to_device_(ctx);
    }

    void solver_free(psum_field_solver_handle h) {
        SolverContext* ctx = static_cast<SolverContext*>(h);
        free_source_device_storage_(ctx);
        cudaStream_t stream = ctx->stream;
        cublasHandle_t cublas_handle = ctx->cublas_handle;
        cusparseHandle_t cusparse_handle = ctx->cusparse_handle;
        if (multigrid::cuda_vector::default_stream_ == stream) {
            multigrid::cuda_vector::default_stream_ = nullptr;
        }
        if (multigrid::cuda_vector::default_cublas_handle_ == cublas_handle) {
            multigrid::cuda_vector::default_cublas_handle_ = nullptr;
        }
        if (multigrid::cuda_vector::default_cusparse_handle_ == cusparse_handle) {
            multigrid::cuda_vector::default_cusparse_handle_ = nullptr;
        }
        delete ctx;
        if (cusparse_handle != nullptr) {
            cusparseDestroy(cusparse_handle);
        }
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
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
        register_solver("cuda_multigrid_gpu", table);
        return 0;
    }();

}

}

}

}
