#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CUDA_INNER_SOLVER_CUH
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CUDA_INNER_SOLVER_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_vector.cuh"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

struct cuda_inner_solver {
    static constexpr int max_coarse_size_ = 8192;

    int size_ = 0;
    double* d_inv_ = nullptr;

    cuda_inner_solver() = default;

    cuda_inner_solver(const cuda_inner_solver&) = delete;
    cuda_inner_solver& operator=(const cuda_inner_solver&) = delete;

    cuda_inner_solver(cuda_inner_solver&& o) noexcept
        : size_(o.size_), d_inv_(o.d_inv_) {
        o.size_ = 0;
        o.d_inv_ = nullptr;
    }

    cuda_inner_solver& operator=(cuda_inner_solver&& o) noexcept {
        if (this != &o) {
            free_();
            size_ = o.size_;
            d_inv_ = o.d_inv_;
            o.size_ = 0;
            o.d_inv_ = nullptr;
        }
        return *this;
    }

    ~cuda_inner_solver() {
        free_();
    }

    void compute(const Eigen::SparseMatrix<double>& A) {
        free_();
        size_ = static_cast<int>(A.rows());
        if (size_ > max_coarse_size_) {
            throw std::runtime_error("cuda_inner_solver: coarse matrix too large (" +
                std::to_string(size_) + " > " + std::to_string(max_coarse_size_) + ")");
        }

        Eigen::MatrixXd inv = Eigen::MatrixXd(A).inverse();

        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&d_inv_), sizeof(double) * size_ * size_),
                             "cuda_inner_solver: device allocation failed");
        cuda_multigrid_check(cudaMemcpyAsync(d_inv_, inv.data(), sizeof(double) * size_ * size_,
                                             cudaMemcpyHostToDevice, stream),
                             "cuda_inner_solver: host-to-device copy failed");
        cuda_multigrid_check(cudaStreamSynchronize(stream), "cuda_inner_solver: upload sync failed");
    }

    cuda_vector solve(const cuda_vector& b) const {
        int n = size_;
        cuda_vector result;
        result.resize(n);
        if (n <= 0) return result;
        solve_to(b, result);
        return result;
    }

    void solve_to(const cuda_vector& b, cuda_vector& result) const {
        int n = size_;
        result.resize(n);
        if (n <= 0) return;
        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        cublasHandle_t handle = cuda_multigrid_cublas_handle(stream);
        const double alpha = 1.0;
        const double beta = 0.0;
        cuda_multigrid_check(cublasDgemv(handle, CUBLAS_OP_N, n, n,
                                         &alpha, d_inv_, n, b.data(), 1,
                                         &beta, result.data(), 1),
                             "cuda_inner_solver: cublasDgemv failed");
    }

private:
    void free_() {
        if (d_inv_ != nullptr) {
            cudaFree(d_inv_);
        }
        d_inv_ = nullptr;
        size_ = 0;
    }
};

}

}

}

}

#endif
