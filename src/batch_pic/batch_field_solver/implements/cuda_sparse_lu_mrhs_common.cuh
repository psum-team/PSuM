#ifndef PSUM_BATCH_PIC_BATCH_FIELD_SOLVER_CUDA_SPARSE_LU_MRHS_COMMON_CUH
#define PSUM_BATCH_PIC_BATCH_FIELD_SOLVER_CUDA_SPARSE_LU_MRHS_COMMON_CUH


#include "../../../field_solver/implements/umfpack_lu_decompose.hpp"
#include <Eigen/Sparse>
#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <vector>

namespace psum {
namespace batch_pic {
namespace batch_field_solver {
namespace implements {
namespace cuda_sparse_lu_mrhs_common {

using psum::field_solver::implements::umfpack_lu_decompose;

struct CuSparseHandle {
    cusparseHandle_t handle = nullptr;
    CuSparseHandle() { cusparseCreate(&handle); }
    ~CuSparseHandle() { cusparseDestroy(handle); }
};

class SparseTriangularSolverMRHSCuda {
public:
    enum TriangularType {
        Lower,
        Upper,
        Invalid
    };

    void set(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& A, cudaStream_t stream = 0) {
        release();

        rows_ = A.rows();
        cols_ = A.cols();
        if (rows_ != cols_) {
            throw std::runtime_error("cuda sparse LU mrhs: triangular matrix must be square");
        }

        tri_type_ = analyze_triangular_type(A);
        if (tri_type_ == TriangularType::Invalid) {
            throw std::runtime_error("cuda sparse LU mrhs: matrix is not triangular");
        }

        nnz_ = A.nonZeros();
        std::vector<int> h_csr_row_ptr(rows_ + 1);
        std::copy(A.outerIndexPtr(), A.outerIndexPtr() + rows_ + 1, h_csr_row_ptr.begin());

        cudaMalloc(&d_csr_row_ptr_, sizeof(int) * (rows_ + 1));
        cudaMalloc(&d_csr_col_ind_, sizeof(int) * nnz_);
        cudaMalloc(&d_csr_val_, sizeof(double) * nnz_);
        cudaMemcpy(d_csr_row_ptr_, h_csr_row_ptr.data(), sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_col_ind_, A.innerIndexPtr(), sizeof(int) * nnz_, cudaMemcpyHostToDevice);
        cudaMemcpy(d_csr_val_, A.valuePtr(), sizeof(double) * nnz_, cudaMemcpyHostToDevice);

        cusparseCreateCsr(&mat_a_, rows_, cols_, nnz_,
                          d_csr_row_ptr_, d_csr_col_ind_, d_csr_val_,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseFillMode_t fill_mode = (tri_type_ == TriangularType::Upper)
            ? CUSPARSE_FILL_MODE_UPPER
            : CUSPARSE_FILL_MODE_LOWER;
        cusparseSpMatSetAttribute(mat_a_, CUSPARSE_SPMAT_FILL_MODE, &fill_mode, sizeof(cusparseFillMode_t));

        associated_stream_ = stream;
        cusparseCreate(&handle_);
        cusparseSetStream(handle_, associated_stream_);
        initialized_ = true;
        if (batch_size_ > 0) {
            build_buffer();
        }
    }

    void set_batch_size(int batch) {
        batch_size_ = batch;
        if (mat_a_ && batch_size_ > 0) {
            build_buffer();
        }
    }

    ~SparseTriangularSolverMRHSCuda() {
        release();
    }

    void solve(double* d_b, double* d_x) {
        if (!initialized_ || batch_size_ <= 0) {
            throw std::runtime_error("cuda sparse LU mrhs: triangular solver not initialized");
        }

        cusparseDnMatSetValues(mat_x_, d_x);
        cusparseDnMatSetValues(mat_b_, d_b);

        cusparseSpSM_solve(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &alpha_, mat_a_, mat_b_, mat_x_, CUDA_R_64F,
                           CUSPARSE_SPSM_ALG_DEFAULT, descr_);
    }

private:
    int rows_ = 0;
    int cols_ = 0;
    int nnz_ = 0;
    int batch_size_ = 0;
    double* d_csr_val_ = nullptr;
    int* d_csr_row_ptr_ = nullptr;
    int* d_csr_col_ind_ = nullptr;
    cusparseSpMatDescr_t mat_a_ = nullptr;
    cusparseHandle_t handle_ = nullptr;
    cusparseDnMatDescr_t mat_x_ = nullptr;
    cusparseDnMatDescr_t mat_b_ = nullptr;
    TriangularType tri_type_ = TriangularType::Invalid;
    cusparseSpSMDescr_t descr_ = nullptr;
    void* d_buffer_ = nullptr;
    double* d_analysis_b_ = nullptr;
    double* d_analysis_x_ = nullptr;
    size_t buffer_size_ = 0;
    double alpha_ = 1.0;
    cudaStream_t associated_stream_ = 0;
    bool initialized_ = false;

    void release() {
        if (descr_) {
            cusparseSpSM_destroyDescr(descr_);
            descr_ = nullptr;
        }
        if (mat_x_) {
            cusparseDestroyDnMat(mat_x_);
            mat_x_ = nullptr;
        }
        if (mat_b_) {
            cusparseDestroyDnMat(mat_b_);
            mat_b_ = nullptr;
        }
        if (d_buffer_) {
            cudaFree(d_buffer_);
            d_buffer_ = nullptr;
        }
        if (d_analysis_b_) {
            cudaFree(d_analysis_b_);
            d_analysis_b_ = nullptr;
        }
        if (d_analysis_x_) {
            cudaFree(d_analysis_x_);
            d_analysis_x_ = nullptr;
        }
        if (mat_a_) {
            cusparseDestroySpMat(mat_a_);
            mat_a_ = nullptr;
        }
        if (handle_) {
            cusparseDestroy(handle_);
            handle_ = nullptr;
        }
        if (d_csr_row_ptr_) {
            cudaFree(d_csr_row_ptr_);
            d_csr_row_ptr_ = nullptr;
        }
        if (d_csr_col_ind_) {
            cudaFree(d_csr_col_ind_);
            d_csr_col_ind_ = nullptr;
        }
        if (d_csr_val_) {
            cudaFree(d_csr_val_);
            d_csr_val_ = nullptr;
        }
        rows_ = 0;
        cols_ = 0;
        nnz_ = 0;
        tri_type_ = TriangularType::Invalid;
        buffer_size_ = 0;
        associated_stream_ = 0;
        initialized_ = false;
    }

    void build_buffer() {
        cudaMalloc(&d_analysis_b_, (size_t)rows_ * batch_size_ * sizeof(double));
        cudaMalloc(&d_analysis_x_, (size_t)rows_ * batch_size_ * sizeof(double));

        cusparseCreateDnMat(&mat_x_, rows_, batch_size_, batch_size_, d_analysis_x_, CUDA_R_64F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&mat_b_, rows_, batch_size_, batch_size_, d_analysis_b_, CUDA_R_64F, CUSPARSE_ORDER_ROW);
        cusparseSpSM_createDescr(&descr_);

        cusparseSpSM_bufferSize(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha_, mat_a_, mat_b_, mat_x_, CUDA_R_64F,
                                CUSPARSE_SPSM_ALG_DEFAULT, descr_, &buffer_size_);

        cudaMalloc(&d_buffer_, buffer_size_);

        cusparseSpSM_analysis(handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha_, mat_a_, mat_b_, mat_x_, CUDA_R_64F,
                              CUSPARSE_SPSM_ALG_DEFAULT, descr_, d_buffer_);
    }

    template<typename MatrixType>
    TriangularType analyze_triangular_type(const MatrixType& A) {
        bool lower = true;
        bool upper = true;
        for (int k = 0; k < A.outerSize(); ++k) {
            for (typename MatrixType::InnerIterator it(A, k); it; ++it) {
                if (it.row() < it.col()) {
                    lower = false;
                }
                if (it.row() > it.col()) {
                    upper = false;
                }
            }
        }
        if (lower && !upper) {
            return TriangularType::Lower;
        }
        if (upper && !lower) {
            return TriangularType::Upper;
        }
        if (lower && upper) {
            return TriangularType::Lower;
        }
        return TriangularType::Invalid;
    }
};

static __global__ void apply_permutation_mrhs_kernel(int n, int batch, const int* inv_perm, const double* b, double* x) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const double* src = b + (size_t)idx * batch;
        double* dst = x + (size_t)inv_perm[idx] * batch;
        for (int c = 0; c < batch; ++c) {
            dst[c] = src[c];
        }
    }
}

class PermutationSolverMRHSCuda {
public:
    void set(const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>& P, cudaStream_t stream = 0) {
        release();

        rows_ = P.rows();

        std::vector<int> inv_perm(rows_);
        for (int i = 0; i < rows_; ++i) {
            inv_perm[P.indices()[i]] = i;
        }
        identity_ = true;
        for (int i = 0; i < rows_; ++i) {
            if (inv_perm[i] != i) {
                identity_ = false;
                break;
            }
        }

        if (!identity_) {
            cudaMalloc(&d_inv_perm_, sizeof(int) * rows_);
            cudaMemcpy(d_inv_perm_, inv_perm.data(), sizeof(int) * rows_, cudaMemcpyHostToDevice);
        }
        associated_stream_ = stream;
        initialized_ = true;
    }

    ~PermutationSolverMRHSCuda() {
        release();
    }

    void solve(double* d_b, double* d_x) {
        if (!initialized_) {
            throw std::runtime_error("cuda sparse LU mrhs: permutation solver not initialized");
        }
        if (identity_) {
            if (d_b != d_x) {
                cudaMemcpyAsync(d_x, d_b, sizeof(double) * rows_ * batch_size_, cudaMemcpyDeviceToDevice, associated_stream_);
            }
            return;
        }

        const int block_size = 256;
        const int grid_size = (rows_ + block_size - 1) / block_size;
        apply_permutation_mrhs_kernel<<<grid_size, block_size, 0, associated_stream_>>>(rows_, batch_size_, d_inv_perm_, d_b, d_x);
    }

    bool is_identity() const {
        return identity_;
    }

    void set_batch_size(int batch) {
        batch_size_ = batch;
    }

private:
    int rows_ = 0;
    int batch_size_ = 0;
    int* d_inv_perm_ = nullptr;
    bool initialized_ = false;
    bool identity_ = false;
    cudaStream_t associated_stream_ = 0;

    void release() {
        if (d_inv_perm_) {
            cudaFree(d_inv_perm_);
            d_inv_perm_ = nullptr;
        }
        rows_ = 0;
        batch_size_ = 0;
        initialized_ = false;
        identity_ = false;
        associated_stream_ = 0;
    }
};

class SparseLUCudaMRHS {
public:
    ~SparseLUCudaMRHS() {
        if (d_temp1_) {
            cudaFree(d_temp1_);
        }
        if (d_temp2_) {
            cudaFree(d_temp2_);
        }
        if (d_temp3_) {
            cudaFree(d_temp3_);
        }
        if (stream_created_) {
            cudaStreamDestroy(associated_stream_);
        }
    }

    template<typename MatrixType>
    void set(const MatrixType& A, cudaStream_t stream = 0) {
        if (stream != 0) {
            if (stream_created_) {
                cudaStreamDestroy(associated_stream_);
            }
            associated_stream_ = stream;
            stream_created_ = false;
        } else if (!stream_created_) {
            cudaStreamCreate(&associated_stream_);
            stream_created_ = true;
        }
        release_temp_buffers();

        n_ = A.rows();
        if (n_ != A.cols()) {
            throw std::runtime_error("cuda sparse LU mrhs: matrix must be square");
        }

        Eigen::SparseMatrix<double> col_major_a(A);
        auto decomp = umfpack_lu_decompose(col_major_a);
        L_solver_.set(decomp.L, associated_stream_);
        U_solver_.set(decomp.U, associated_stream_);
        PT_solver_.set(decomp.P.transpose(), associated_stream_);
        QT_solver_.set(decomp.Q.transpose(), associated_stream_);

        L_solver_.set_batch_size(batch_size_);
        U_solver_.set_batch_size(batch_size_);
        PT_solver_.set_batch_size(batch_size_);
        QT_solver_.set_batch_size(batch_size_);

        allocate_temp_buffers();
        initialized_ = true;
    }

    void set_batch_size(int batch) {
        batch_size_ = batch;
        L_solver_.set_batch_size(batch);
        U_solver_.set_batch_size(batch);
        PT_solver_.set_batch_size(batch);
        QT_solver_.set_batch_size(batch);
        if (initialized_) {
            allocate_temp_buffers();
        }
    }

    void solve_async(double* d_b, double* d_x) {
        if (!initialized_) {
            throw std::runtime_error("cuda sparse LU mrhs: solver not initialized");
        }

        double* after_pt = d_temp1_;
        if (PT_solver_.is_identity()) {
            after_pt = d_b;
        } else {
            PT_solver_.solve(d_b, after_pt);
        }

        L_solver_.solve(after_pt, d_temp2_);

        if (QT_solver_.is_identity()) {
            U_solver_.solve(d_temp2_, d_x);
        } else {
            U_solver_.solve(d_temp2_, d_temp3_);
            QT_solver_.solve(d_temp3_, d_x);
        }
    }

    void solve(double* d_b, double* d_x) {
        solve_async(d_b, d_x);
        cudaStreamSynchronize(associated_stream_);
    }

    int size() const {
        return n_;
    }

    cudaStream_t stream() const {
        return associated_stream_;
    }

private:
    int n_ = 0;
    int batch_size_ = 0;
    SparseTriangularSolverMRHSCuda L_solver_;
    SparseTriangularSolverMRHSCuda U_solver_;
    PermutationSolverMRHSCuda PT_solver_;
    PermutationSolverMRHSCuda QT_solver_;
    cudaStream_t associated_stream_ = 0;
    bool stream_created_ = false;
    bool initialized_ = false;

    double* d_temp1_ = nullptr;
    double* d_temp2_ = nullptr;
    double* d_temp3_ = nullptr;

    void release_temp_buffers() {
        if (d_temp1_) {
            cudaFree(d_temp1_);
            d_temp1_ = nullptr;
        }
        if (d_temp2_) {
            cudaFree(d_temp2_);
            d_temp2_ = nullptr;
        }
        if (d_temp3_) {
            cudaFree(d_temp3_);
            d_temp3_ = nullptr;
        }
        initialized_ = false;
    }

    void allocate_temp_buffers() {
        release_temp_buffers();
        const size_t bytes = (size_t)n_ * batch_size_ * sizeof(double);
        if (batch_size_ <= 0) {
            return;
        }
        if (!PT_solver_.is_identity()) {
            cudaMalloc(&d_temp1_, bytes);
        }
        cudaMalloc(&d_temp2_, bytes);
        if (!QT_solver_.is_identity()) {
            cudaMalloc(&d_temp3_, bytes);
        }
        initialized_ = true;
    }
};

}
}
}
}
}

#endif
