#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CUDA_VECTOR_CUH
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CUDA_VECTOR_CUH

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <Eigen/Core>
#include <stdexcept>
#include <string>
#include <vector>
#include "sparse_linear_trait.hpp"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

inline void cuda_multigrid_check(cudaError_t status, const char* where) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(status));
    }
}

inline void cuda_multigrid_check(cublasStatus_t status, const char* where) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": cuBLAS status " + std::to_string(status));
    }
}

inline void cuda_multigrid_check(cusparseStatus_t status, const char* where) {
    if (status != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(where) + ": " + cusparseGetErrorString(status));
    }
}

inline cudaStream_t cuda_multigrid_stream(cudaStream_t stream) {
    return stream == nullptr ? cudaStreamDefault : stream;
}

static __global__ void cuda_vector_add_kernel(const double* a, const double* b, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

static __global__ void cuda_vector_subtract_kernel(const double* a, const double* b, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

static __global__ void cuda_vector_multiply_kernel(const double* a, const double* b, double* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

struct cuda_vector {
    static inline cudaStream_t default_stream_ = nullptr;
    static inline cublasHandle_t default_cublas_handle_ = nullptr;
    static inline cusparseHandle_t default_cusparse_handle_ = nullptr;

    double* data_ = nullptr;
    int size_ = 0;

    cuda_vector() = default;

    cuda_vector(const cuda_vector&) = delete;
    cuda_vector& operator=(const cuda_vector&) = delete;

    cuda_vector(cuda_vector&& o) noexcept
        : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    cuda_vector& operator=(cuda_vector&& o) noexcept {
        if (this != &o) {
            free_();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    ~cuda_vector() {
        free_();
    }

    void resize(int n) {
        if (n == size_) return;
        free_();
        if (n > 0) {
            cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(double) * n),
                                 "cuda_vector: device allocation failed");
            size_ = n;
        }
    }

    int rows() const { return size_; }

    void setZero() {
        if (data_ && size_ > 0) {
            cudaStream_t stream = cuda_multigrid_stream(default_stream_);
            cuda_multigrid_check(cudaMemsetAsync(data_, 0, sizeof(double) * size_, stream),
                                 "cuda_vector: memset failed");
        }
    }

    double* data() { return data_; }
    const double* data() const { return data_; }

private:
    void free_() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
        data_ = nullptr;
        size_ = 0;
    }
};

inline cublasHandle_t cuda_multigrid_cublas_handle(cudaStream_t) {
    cublasHandle_t handle = cuda_vector::default_cublas_handle_;
    if (handle == nullptr) {
        throw std::runtime_error("cuda_multigrid: cuBLAS handle is not initialized");
    }
    return handle;
}

inline cusparseHandle_t cuda_multigrid_cusparse_handle(cudaStream_t) {
    cusparseHandle_t handle = cuda_vector::default_cusparse_handle_;
    if (handle == nullptr) {
        throw std::runtime_error("cuda_multigrid: cuSPARSE handle is not initialized");
    }
    return handle;
}

template <>
struct vector_traits<cuda_vector> {
    static cuda_vector add(const cuda_vector& a, const cuda_vector& b) {
        int n = a.rows();
        cuda_vector result;
        result.resize(n);
        if (n > 0) {
            cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
            cuda_vector_add_kernel<<<blocks_(n), threads_(), 0, stream>>>(a.data(), b.data(), result.data(), n);
            cuda_multigrid_check(cudaGetLastError(), "cuda_vector: add launch failed");
        }
        return result;
    }

    static cuda_vector subtract(const cuda_vector& a, const cuda_vector& b) {
        int n = a.rows();
        cuda_vector result;
        result.resize(n);
        if (n > 0) {
            cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
            cuda_vector_subtract_kernel<<<blocks_(n), threads_(), 0, stream>>>(a.data(), b.data(), result.data(), n);
            cuda_multigrid_check(cudaGetLastError(), "cuda_vector: subtract launch failed");
        }
        return result;
    }

    static cuda_vector multiply_with_array(const cuda_vector& vec, const cuda_vector& arr) {
        int n = vec.rows();
        cuda_vector result;
        result.resize(n);
        if (n > 0) {
            cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
            cuda_vector_multiply_kernel<<<blocks_(n), threads_(), 0, stream>>>(vec.data(), arr.data(), result.data(), n);
            cuda_multigrid_check(cudaGetLastError(), "cuda_vector: multiply launch failed");
        }
        return result;
    }

    static void set(std::vector<cuda_vector>& dest, const std::vector<Eigen::ArrayXd>& src) {
        dest.resize(src.size());
        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(static_cast<int>(src[i].size()));
            cuda_multigrid_check(cudaMemcpyAsync(dest[i].data(), src[i].data(),
                                                 sizeof(double) * src[i].size(),
                                                 cudaMemcpyHostToDevice, stream),
                                 "cuda_vector: host-to-device copy failed");
        }
        cuda_multigrid_check(cudaStreamSynchronize(stream), "cuda_vector: host-to-device sync failed");
    }

    static void copy_data(cuda_vector& dst, const cuda_vector& src) {
        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        cuda_multigrid_check(cudaMemcpyAsync(dst.data(), src.data(), sizeof(double) * dst.rows(),
                                             cudaMemcpyDeviceToDevice, stream),
                             "cuda_vector: device-to-device copy failed");
    }

    static void hadamard_inplace(cuda_vector& vec, const cuda_vector& arr) {
        int n = vec.rows();
        if (n <= 0) return;
        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        cublasHandle_t handle = cuda_multigrid_cublas_handle(stream);
        cuda_multigrid_check(cublasDdgmm(handle, CUBLAS_SIDE_LEFT, n, 1,
                                         vec.data(), n, arr.data(), 1, vec.data(), n),
                             "cuda_vector: cublasDdgmm inplace failed");
    }

    static void hadamard_to(const cuda_vector& in, const cuda_vector& arr, cuda_vector& out) {
        int n = in.rows();
        if (n <= 0) return;
        cudaStream_t stream = cuda_multigrid_stream(cuda_vector::default_stream_);
        cublasHandle_t handle = cuda_multigrid_cublas_handle(stream);
        cuda_multigrid_check(cublasDdgmm(handle, CUBLAS_SIDE_LEFT, n, 1,
                                         in.data(), n, arr.data(), 1, out.data(), n),
                             "cuda_vector: cublasDdgmm failed");
    }

private:
    static int threads_() { return 256; }
    static int blocks_(int n) { return (n + threads_() - 1) / threads_(); }
};

}

}

}

}

#endif
