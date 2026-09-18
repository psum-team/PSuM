#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_CUDA_SPARSE_MATRIX_CUH
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_CUDA_SPARSE_MATRIX_CUH

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "convSparseMatrix.hpp"
#include "cuda_vector.cuh"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template<typename T>
__global__ void cuda_scatter_rows_kernel(int count, const int* rows, const T* compact, T* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        out[rows[idx]] = compact[idx];
    }
}

template<typename T>
__global__ void cuda_conv_sparse_kernel(int total_rows, const int* rows, const int* anchors,
                                        const int* group_ids, const int* offsets,
                                        const T* coeffs, const int* group_coeff_offsets,
                                        const int* group_coeff_counts, bool anchor_at_max,
                                        const T* in, T* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rows) return;

    int row = rows[idx];
    int group = group_ids[idx];
    int coeff_begin = group_coeff_offsets[group];
    int coeff_count = group_coeff_counts[group];
    int base = anchor_at_max ? anchors[idx] : row;
    T value = T(0);
    for (int c = 0; c < coeff_count; ++c) {
        int pos = coeff_begin + c;
        value += coeffs[pos] * in[base + offsets[pos]];
    }
    out[row] = value;
}

template<typename T>
struct CudaCsrMatrix {
    int rows_ = 0;
    int cols_ = 0;
    int nnz_ = 0;
    int* d_row_offsets_ = nullptr;
    int* d_col_indices_ = nullptr;
    T* d_values_ = nullptr;
    cusparseSpMatDescr_t mat_descr_ = nullptr;
    cusparseDnVecDescr_t in_vec_ = nullptr;
    cusparseDnVecDescr_t out_vec_ = nullptr;
    const T* in_vec_ptr_ = nullptr;
    T* out_vec_ptr_ = nullptr;
    void* d_buffer_ = nullptr;
    size_t buffer_size_ = 0;
    cudaStream_t stream_ = nullptr;

    CudaCsrMatrix() = default;

    explicit CudaCsrMatrix(cudaStream_t stream) : stream_(stream) {}

    CudaCsrMatrix(const CudaCsrMatrix&) = delete;
    CudaCsrMatrix& operator=(const CudaCsrMatrix&) = delete;

    CudaCsrMatrix(CudaCsrMatrix&& other) noexcept
        : rows_(other.rows_),
          cols_(other.cols_),
          nnz_(other.nnz_),
          d_row_offsets_(other.d_row_offsets_),
          d_col_indices_(other.d_col_indices_),
          d_values_(other.d_values_),
          mat_descr_(other.mat_descr_),
          in_vec_(other.in_vec_),
          out_vec_(other.out_vec_),
          in_vec_ptr_(other.in_vec_ptr_),
          out_vec_ptr_(other.out_vec_ptr_),
          d_buffer_(other.d_buffer_),
          buffer_size_(other.buffer_size_),
          stream_(other.stream_) {
        other.zero_pointers_();
    }

    CudaCsrMatrix& operator=(CudaCsrMatrix&& other) noexcept {
        if (this != &other) {
            free_device_storage_();
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            d_row_offsets_ = other.d_row_offsets_;
            d_col_indices_ = other.d_col_indices_;
            d_values_ = other.d_values_;
            mat_descr_ = other.mat_descr_;
            in_vec_ = other.in_vec_;
            out_vec_ = other.out_vec_;
            in_vec_ptr_ = other.in_vec_ptr_;
            out_vec_ptr_ = other.out_vec_ptr_;
            d_buffer_ = other.d_buffer_;
            buffer_size_ = other.buffer_size_;
            stream_ = other.stream_;
            other.zero_pointers_();
        }
        return *this;
    }

    ~CudaCsrMatrix() {
        free_device_storage_();
    }

    template<int EigenOptions, typename EigenIndex>
    void set(const Eigen::SparseMatrix<T, EigenOptions, EigenIndex>& mat, cudaStream_t stream) {
        free_device_storage_();
        stream_ = stream;

        Eigen::SparseMatrix<T, Eigen::RowMajor> row_major = mat;
        row_major.makeCompressed();

        rows_ = static_cast<int>(row_major.rows());
        cols_ = static_cast<int>(row_major.cols());
        nnz_ = static_cast<int>(row_major.nonZeros());

        std::vector<int> h_row_offsets(rows_ + 1);
        for (int i = 0; i <= rows_; ++i) {
            h_row_offsets[i] = static_cast<int>(row_major.outerIndexPtr()[i]);
        }

        std::vector<int> h_col_indices(nnz_);
        std::vector<T> h_values(nnz_);
        for (int i = 0; i < nnz_; ++i) {
            h_col_indices[i] = static_cast<int>(row_major.innerIndexPtr()[i]);
            h_values[i] = row_major.valuePtr()[i];
        }

        d_row_offsets_ = alloc_copy_(h_row_offsets);
        d_col_indices_ = alloc_copy_(h_col_indices);
        d_values_ = alloc_copy_(h_values);

        cuda_multigrid_check(cusparseCreateCsr(&mat_descr_,
                                               rows_, cols_, nnz_,
                                               d_row_offsets_, d_col_indices_, d_values_,
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_32I,
                                               CUSPARSE_INDEX_BASE_ZERO,
                                               data_type_()),
                             "CudaCsrMatrix: cusparseCreateCsr failed");
    }

    void multiply(const T* d_in, T* d_out) {
        if (rows_ <= 0) return;
        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cusparseHandle_t handle = cuda_multigrid_cusparse_handle(stream);

        bind_dense_vectors_(d_in, d_out);

        const T alpha = T(1);
        const T beta = T(0);
        if (d_buffer_ == nullptr) {
            cuda_multigrid_check(cusparseSpMV_bufferSize(handle,
                                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                         &alpha,
                                                         mat_descr_,
                                                         in_vec_,
                                                         &beta,
                                                         out_vec_,
                                                         data_type_(),
                                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                                         &buffer_size_),
                                 "CudaCsrMatrix: cusparseSpMV_bufferSize failed");
            cuda_multigrid_check(cudaMalloc(&d_buffer_, buffer_size_),
                                 "CudaCsrMatrix: SpMV buffer allocation failed");
        }

        cuda_multigrid_check(cusparseSpMV(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          mat_descr_,
                                          in_vec_,
                                          &beta,
                                          out_vec_,
                                          data_type_(),
                                          CUSPARSE_SPMV_ALG_DEFAULT,
                                          d_buffer_),
                             "CudaCsrMatrix: cusparseSpMV failed");
    }

private:
    static cudaDataType data_type_() {
        if constexpr (std::is_same_v<T, double>) {
            return CUDA_R_64F;
        } else if constexpr (std::is_same_v<T, float>) {
            return CUDA_R_32F;
        } else {
            throw std::runtime_error("CudaCsrMatrix: unsupported scalar type");
        }
    }

    void zero_pointers_() {
        rows_ = 0;
        cols_ = 0;
        nnz_ = 0;
        d_row_offsets_ = nullptr;
        d_col_indices_ = nullptr;
        d_values_ = nullptr;
        mat_descr_ = nullptr;
        in_vec_ = nullptr;
        out_vec_ = nullptr;
        in_vec_ptr_ = nullptr;
        out_vec_ptr_ = nullptr;
        d_buffer_ = nullptr;
        buffer_size_ = 0;
    }

    void bind_dense_vectors_(const T* d_in, T* d_out) {
        if (in_vec_ == nullptr) {
            cuda_multigrid_check(cusparseCreateDnVec(&in_vec_, cols_, const_cast<T*>(d_in), data_type_()),
                                 "CudaCsrMatrix: cusparseCreateDnVec input failed");
            in_vec_ptr_ = d_in;
        } else if (in_vec_ptr_ != d_in) {
            cuda_multigrid_check(cusparseDnVecSetValues(in_vec_, const_cast<T*>(d_in)),
                                 "CudaCsrMatrix: cusparseDnVecSetValues input failed");
            in_vec_ptr_ = d_in;
        }

        if (out_vec_ == nullptr) {
            cuda_multigrid_check(cusparseCreateDnVec(&out_vec_, rows_, d_out, data_type_()),
                                 "CudaCsrMatrix: cusparseCreateDnVec output failed");
            out_vec_ptr_ = d_out;
        } else if (out_vec_ptr_ != d_out) {
            cuda_multigrid_check(cusparseDnVecSetValues(out_vec_, d_out),
                                 "CudaCsrMatrix: cusparseDnVecSetValues output failed");
            out_vec_ptr_ = d_out;
        }
    }

    template<typename U>
    U* alloc_copy_(const std::vector<U>& host) {
        if (host.empty()) return nullptr;
        U* d = nullptr;
        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(U) * host.size()),
                             "CudaCsrMatrix: device allocation failed");
        cuda_multigrid_check(cudaMemcpyAsync(d, host.data(), sizeof(U) * host.size(),
                                             cudaMemcpyHostToDevice, stream),
                             "CudaCsrMatrix: host-to-device copy failed");
        cuda_multigrid_check(cudaStreamSynchronize(stream), "CudaCsrMatrix: upload sync failed");
        return d;
    }

    template<typename U>
    void free_ptr_(U*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    void free_device_storage_() {
        if (in_vec_ != nullptr) {
            cusparseDestroyDnVec(in_vec_);
            in_vec_ = nullptr;
        }
        if (out_vec_ != nullptr) {
            cusparseDestroyDnVec(out_vec_);
            out_vec_ = nullptr;
        }
        if (mat_descr_ != nullptr) {
            cusparseDestroySpMat(mat_descr_);
            mat_descr_ = nullptr;
        }
        free_ptr_(d_buffer_);
        free_ptr_(d_row_offsets_);
        free_ptr_(d_col_indices_);
        free_ptr_(d_values_);
        zero_pointers_();
    }
};

template<typename T>
struct ConvCudaSparseMatrix {
    cudaStream_t stream_ = nullptr;
    int MatRows_ = 0;
    bool Anchor_at_max_ = false;

    int num_groups_ = 0;
    int total_conv_rows_ = 0;
    int total_stencil_entries_ = 0;
    bool covers_all_rows_ = false;

    int* d_group_coeff_offsets_ = nullptr;
    int* d_group_coeff_counts_ = nullptr;
    int* d_conv_rows_ = nullptr;
    int* d_conv_anchors_ = nullptr;
    int* d_conv_group_id_ = nullptr;
    int* d_stencil_col_offsets_ = nullptr;
    T* d_stencil_vals_ = nullptr;

    CudaCsrMatrix<T> residual_spmv_;
    int num_residual_rows_ = 0;
    int* d_non_all_zero_rows_ = nullptr;
    T* d_compact_buf_ = nullptr;

    T* multiply_scratch_ = nullptr;
    Eigen::SparseMatrix<T, Eigen::RowMajor> original_mat_;

    ConvCudaSparseMatrix() = default;

    explicit ConvCudaSparseMatrix(cudaStream_t stream) : stream_(stream), residual_spmv_(stream) {}

    ConvCudaSparseMatrix(const ConvCudaSparseMatrix&) = delete;
    ConvCudaSparseMatrix& operator=(const ConvCudaSparseMatrix&) = delete;

    ConvCudaSparseMatrix(ConvCudaSparseMatrix&& other) noexcept
        : stream_(other.stream_),
          MatRows_(other.MatRows_),
          Anchor_at_max_(other.Anchor_at_max_),
          num_groups_(other.num_groups_),
          total_conv_rows_(other.total_conv_rows_),
          total_stencil_entries_(other.total_stencil_entries_),
          covers_all_rows_(other.covers_all_rows_),
          d_group_coeff_offsets_(other.d_group_coeff_offsets_),
          d_group_coeff_counts_(other.d_group_coeff_counts_),
          d_conv_rows_(other.d_conv_rows_),
          d_conv_anchors_(other.d_conv_anchors_),
          d_conv_group_id_(other.d_conv_group_id_),
          d_stencil_col_offsets_(other.d_stencil_col_offsets_),
          d_stencil_vals_(other.d_stencil_vals_),
          residual_spmv_(std::move(other.residual_spmv_)),
          num_residual_rows_(other.num_residual_rows_),
          d_non_all_zero_rows_(other.d_non_all_zero_rows_),
          d_compact_buf_(other.d_compact_buf_),
          multiply_scratch_(other.multiply_scratch_),
          original_mat_(std::move(other.original_mat_)) {
        other.zero_pointers_();
    }

    ConvCudaSparseMatrix& operator=(ConvCudaSparseMatrix&& other) noexcept {
        if (this != &other) {
            free_device_storage_();
            stream_ = other.stream_;
            MatRows_ = other.MatRows_;
            Anchor_at_max_ = other.Anchor_at_max_;
            num_groups_ = other.num_groups_;
            total_conv_rows_ = other.total_conv_rows_;
            total_stencil_entries_ = other.total_stencil_entries_;
            covers_all_rows_ = other.covers_all_rows_;
            d_group_coeff_offsets_ = other.d_group_coeff_offsets_;
            d_group_coeff_counts_ = other.d_group_coeff_counts_;
            d_conv_rows_ = other.d_conv_rows_;
            d_conv_anchors_ = other.d_conv_anchors_;
            d_conv_group_id_ = other.d_conv_group_id_;
            d_stencil_col_offsets_ = other.d_stencil_col_offsets_;
            d_stencil_vals_ = other.d_stencil_vals_;
            residual_spmv_ = std::move(other.residual_spmv_);
            num_residual_rows_ = other.num_residual_rows_;
            d_non_all_zero_rows_ = other.d_non_all_zero_rows_;
            d_compact_buf_ = other.d_compact_buf_;
            multiply_scratch_ = other.multiply_scratch_;
            original_mat_ = std::move(other.original_mat_);
            other.zero_pointers_();
        }
        return *this;
    }

    ~ConvCudaSparseMatrix() {
        free_device_storage_();
    }

    template<int EigenOptions, typename EigenIndex>
    void set(const Eigen::SparseMatrix<T, EigenOptions, EigenIndex>& mat) {
        free_device_storage_();
        stream_ = cuda_vector::default_stream_;
        MatRows_ = static_cast<int>(mat.rows());
        original_mat_ = mat;

        ConvSparseMatrix<T> cpu_analyzer;
        cpu_analyzer.set(mat, 0);
        upload_residual_data_(cpu_analyzer);
        update_coverage_();
        alloc_multiply_scratch_();
    }

    template<int EigenOptions, typename EigenIndex>
    void set(const Eigen::SparseMatrix<T, EigenOptions, EigenIndex>& mat, int num_conv_mask) {
        free_device_storage_();
        stream_ = cuda_vector::default_stream_;

        ConvSparseMatrix<T> cpu_analyzer;
        cpu_analyzer.set(mat, num_conv_mask);

        MatRows_ = cpu_analyzer.MatRows;
        Anchor_at_max_ = cpu_analyzer.Anchor_at_max;
        original_mat_ = cpu_analyzer.original_mat;

        upload_conv_data_(cpu_analyzer);
        upload_residual_data_(cpu_analyzer);
        update_coverage_();
        alloc_multiply_scratch_();
    }

    void multiply(const T* d_in, T* d_out) {
        if (MatRows_ == 0) return;

        cudaStream_t stream = cuda_multigrid_stream(stream_);
        if (!covers_all_rows_) {
            cuda_multigrid_check(cudaMemsetAsync(d_out, 0, sizeof(T) * MatRows_, stream),
                                 "ConvCudaSparseMatrix: output memset failed");
        }

        if (num_residual_rows_ > 0) {
            residual_spmv_.multiply(d_in, d_compact_buf_);
            scatter_kernel_(d_out);
        }

        if (total_conv_rows_ > 0) {
            conv_kernel_(d_in, d_out);
        }
    }

    void multiply_add(const T* d_in, const T* d_b, T* d_out, T b_coeff, T ans_coeff) {
        multiply(d_in, multiply_scratch_);

        int n = MatRows_;
        if (n <= 0) return;

        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cublasHandle_t handle = cuda_multigrid_cublas_handle(stream);
        T alpha = ans_coeff;
        T beta = b_coeff * ans_coeff;
        if constexpr (std::is_same_v<T, double>) {
            cuda_multigrid_check(cublasDgeam(handle,
                                             CUBLAS_OP_N, CUBLAS_OP_N,
                                             n, 1,
                                             &alpha, multiply_scratch_, n,
                                             &beta, d_b, n,
                                             d_out, n),
                                 "ConvCudaSparseMatrix: cublasDgeam failed");
        } else if constexpr (std::is_same_v<T, float>) {
            cuda_multigrid_check(cublasSgeam(handle,
                                             CUBLAS_OP_N, CUBLAS_OP_N,
                                             n, 1,
                                             &alpha, multiply_scratch_, n,
                                             &beta, d_b, n,
                                             d_out, n),
                                 "ConvCudaSparseMatrix: cublasSgeam failed");
        } else {
            throw std::runtime_error("ConvCudaSparseMatrix: unsupported scalar type");
        }
    }

    int rows() const { return MatRows_; }
    int cols() const { return MatRows_; }

    T coeff(int row, int col) const {
        return original_mat_.coeff(row, col);
    }

private:
    void zero_pointers_() {
        d_group_coeff_offsets_ = nullptr;
        d_group_coeff_counts_ = nullptr;
        d_conv_rows_ = nullptr;
        d_conv_anchors_ = nullptr;
        d_conv_group_id_ = nullptr;
        d_stencil_col_offsets_ = nullptr;
        d_stencil_vals_ = nullptr;
        d_non_all_zero_rows_ = nullptr;
        d_compact_buf_ = nullptr;
        multiply_scratch_ = nullptr;
        num_groups_ = 0;
        total_conv_rows_ = 0;
        total_stencil_entries_ = 0;
        covers_all_rows_ = false;
        num_residual_rows_ = 0;
        MatRows_ = 0;
    }

    void update_coverage_() {
        covers_all_rows_ = (MatRows_ > 0 && total_conv_rows_ + num_residual_rows_ == MatRows_);
    }

    template<typename U>
    U* alloc_copy_(const std::vector<U>& host) {
        if (host.empty()) return nullptr;
        U* d = nullptr;
        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&d), sizeof(U) * host.size()),
                             "ConvCudaSparseMatrix: device allocation failed");
        cuda_multigrid_check(cudaMemcpyAsync(d, host.data(), sizeof(U) * host.size(),
                                             cudaMemcpyHostToDevice, stream),
                             "ConvCudaSparseMatrix: host-to-device copy failed");
        cuda_multigrid_check(cudaStreamSynchronize(stream), "ConvCudaSparseMatrix: upload sync failed");
        return d;
    }

    template<typename U>
    void free_ptr_(U*& ptr) {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
    }

    void alloc_multiply_scratch_() {
        if (MatRows_ > 0) {
            cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&multiply_scratch_), sizeof(T) * MatRows_),
                                 "ConvCudaSparseMatrix: scratch allocation failed");
        }
    }

    void upload_conv_data_(const ConvSparseMatrix<T>& cpu) {
        const auto& convCmpt = cpu.convCmpt;
        const auto& convs_Anchor = cpu.convs_Anchor;

        num_groups_ = static_cast<int>(convCmpt.size());
        if (num_groups_ == 0) {
            total_conv_rows_ = 0;
            total_stencil_entries_ = 0;
            return;
        }

        std::vector<int> h_group_coeff_offsets;
        std::vector<int> h_group_coeff_counts;
        std::vector<int> h_conv_rows;
        std::vector<int> h_conv_anchors;
        std::vector<int> h_conv_group_id;
        std::vector<int> h_stencil_col_offsets;
        std::vector<T> h_stencil_vals;

        h_group_coeff_offsets.reserve(num_groups_);
        h_group_coeff_counts.reserve(num_groups_);
        total_conv_rows_ = 0;
        total_stencil_entries_ = 0;
        for (int g = 0; g < num_groups_; ++g) {
            const auto& stencil = convCmpt[g].first;
            const auto& rows = convCmpt[g].second;
            int stencil_size = static_cast<int>(stencil.size());
            int row_count = static_cast<int>(rows.size());
            h_group_coeff_offsets.push_back(static_cast<int>(h_stencil_vals.size()));
            h_group_coeff_counts.push_back(stencil_size);
            for (int c = 0; c < stencil_size; ++c) {
                h_stencil_vals.push_back(stencil[c].first);
                h_stencil_col_offsets.push_back(stencil[c].second);
            }
            for (int r = 0; r < row_count; ++r) {
                h_conv_rows.push_back(rows[r]);
                h_conv_group_id.push_back(g);
                if (Anchor_at_max_) h_conv_anchors.push_back(convs_Anchor[g][r]);
            }
            total_conv_rows_ += row_count;
            total_stencil_entries_ += stencil_size;
        }

        d_group_coeff_offsets_ = alloc_copy_(h_group_coeff_offsets);
        d_group_coeff_counts_ = alloc_copy_(h_group_coeff_counts);
        d_conv_rows_ = alloc_copy_(h_conv_rows);
        d_conv_group_id_ = alloc_copy_(h_conv_group_id);
        d_stencil_col_offsets_ = alloc_copy_(h_stencil_col_offsets);
        d_stencil_vals_ = alloc_copy_(h_stencil_vals);
        if (Anchor_at_max_) d_conv_anchors_ = alloc_copy_(h_conv_anchors);
    }

    void upload_residual_data_(const ConvSparseMatrix<T>& cpu) {
        num_residual_rows_ = static_cast<int>(cpu.non_all_zero_rows.size());

        if (num_residual_rows_ > 0) {
            d_non_all_zero_rows_ = alloc_copy_(cpu.non_all_zero_rows);
            residual_spmv_.set(cpu.restMat, stream_);

            cuda_multigrid_check(cudaMalloc(reinterpret_cast<void**>(&d_compact_buf_), sizeof(T) * num_residual_rows_),
                                 "ConvCudaSparseMatrix: compact buffer allocation failed");
        }
    }

    void scatter_kernel_(T* d_out) {
        int count = num_residual_rows_;
        if (count <= 0) return;
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cuda_scatter_rows_kernel<T><<<blocks, threads, 0, stream>>>(
            count, d_non_all_zero_rows_, d_compact_buf_, d_out);
        cuda_multigrid_check(cudaGetLastError(), "ConvCudaSparseMatrix: scatter launch failed");
    }

    void conv_kernel_(const T* d_in, T* d_out) {
        if (total_conv_rows_ <= 0) return;
        int threads = 256;
        int blocks = (total_conv_rows_ + threads - 1) / threads;
        cudaStream_t stream = cuda_multigrid_stream(stream_);
        cuda_conv_sparse_kernel<T><<<blocks, threads, 0, stream>>>(
            total_conv_rows_, d_conv_rows_, d_conv_anchors_, d_conv_group_id_,
            d_stencil_col_offsets_, d_stencil_vals_, d_group_coeff_offsets_,
            d_group_coeff_counts_, Anchor_at_max_, d_in, d_out);
        cuda_multigrid_check(cudaGetLastError(), "ConvCudaSparseMatrix: conv launch failed");
    }

    void free_device_storage_() {
        free_ptr_(d_group_coeff_offsets_);
        free_ptr_(d_group_coeff_counts_);
        free_ptr_(d_conv_rows_);
        free_ptr_(d_conv_anchors_);
        free_ptr_(d_conv_group_id_);
        free_ptr_(d_stencil_col_offsets_);
        free_ptr_(d_stencil_vals_);
        free_ptr_(d_non_all_zero_rows_);
        free_ptr_(d_compact_buf_);
        free_ptr_(multiply_scratch_);
        residual_spmv_ = CudaCsrMatrix<T>(stream_);
        num_groups_ = 0;
        total_conv_rows_ = 0;
        total_stencil_entries_ = 0;
        covers_all_rows_ = false;
        num_residual_rows_ = 0;
        MatRows_ = 0;
    }
};

template <>
struct matrix_traits<ConvCudaSparseMatrix<double>> {
    using Scalar = double;
    using Vector = cuda_vector;

    static void multiply(ConvCudaSparseMatrix<double>& mat, const Vector& in, Vector& out) {
        mat.multiply(in.data(), out.data());
    }

    static void multiply_add(ConvCudaSparseMatrix<double>& mat, const Vector& x,
                             const Vector& b, Vector& out, double b_coeff = 1.0, double ans_coeff = 1.0) {
        mat.multiply_add(x.data(), b.data(), out.data(), b_coeff, ans_coeff);
    }

    static void set(ConvCudaSparseMatrix<double>& dest, const Eigen::SparseMatrix<double>& src) {
        dest.set(src);
    }
};

}

}

}

}

#endif
