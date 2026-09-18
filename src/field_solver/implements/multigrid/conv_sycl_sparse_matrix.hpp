#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_SYCL_SPARSE_MATRIX_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_SYCL_SPARSE_MATRIX_HPP

#include <sycl/sycl.hpp>
#include <Eigen/SparseCore>
#include <vector>
#include <stdexcept>
#include "convSparseMatrix.hpp"
#include "sycl_vector.hpp"
#include <type_traits>

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template <typename T = double>
class sycl_coo_spmv {
public:
    sycl_coo_spmv() = default;
    explicit sycl_coo_spmv(sycl::queue q) : queue_(q) {}

    sycl_coo_spmv(const Eigen::SparseMatrix<T>& matrix, sycl::queue q)
        : queue_(q) {
        set_matrix(matrix);
    }

    sycl_coo_spmv(const sycl_coo_spmv&) = delete;
    sycl_coo_spmv& operator=(const sycl_coo_spmv&) = delete;

    sycl_coo_spmv(sycl_coo_spmv&& other) noexcept
        : queue_(other.queue_),
          rows_(other.rows_), cols_(other.cols_), nnz_(other.nnz_),
          row_indices_(other.row_indices_), col_indices_(other.col_indices_),
          values_(other.values_), scratch_(other.scratch_) {
        other.zero_out_();
    }

    sycl_coo_spmv& operator=(sycl_coo_spmv&& other) noexcept {
        if (this != &other) {
            free_all_();
            queue_ = other.queue_;
            rows_ = other.rows_; cols_ = other.cols_; nnz_ = other.nnz_;
            row_indices_ = other.row_indices_; col_indices_ = other.col_indices_;
            values_ = other.values_; scratch_ = other.scratch_;
            other.zero_out_();
        }
        return *this;
    }

    ~sycl_coo_spmv() { free_all_(); }

    void set_matrix(const Eigen::SparseMatrix<T>& matrix) {
        free_all_();
        rows_ = static_cast<unsigned long long>(matrix.rows());
        cols_ = static_cast<unsigned long long>(matrix.cols());
        nnz_ = static_cast<unsigned long long>(matrix.nonZeros());

        std::vector<unsigned long long> host_rows, host_cols;
        std::vector<T> host_vals;
        host_rows.reserve(nnz_);
        host_cols.reserve(nnz_);
        host_vals.reserve(nnz_);

        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (typename Eigen::SparseMatrix<T>::InnerIterator it(matrix, k); it; ++it) {
                host_rows.push_back(static_cast<unsigned long long>(it.row()));
                host_cols.push_back(static_cast<unsigned long long>(it.col()));
                host_vals.push_back(it.value());
            }
        }

        row_indices_ = copy_dev_(host_rows);
        col_indices_ = copy_dev_(host_cols);
        values_ = copy_dev_(host_vals);
        scratch_ = rows_ == 0 ? nullptr : sycl::malloc_device<T>(rows_, queue_);
    }

    inline void apply(const T* x, T* y) {
        if (rows_ > 0) {
            queue_.memset(y, 0, sizeof(T) * rows_);
        }
        if (nnz_ == 0) return;

        unsigned long long* rows = row_indices_;
        unsigned long long* cols = col_indices_;
        T* vals = values_;
        unsigned long long nnz = nnz_;

        queue_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nnz), [=](sycl::id<1> item) {
                unsigned long long i = item[0];
                sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ref(y[rows[i]]);
                ref.fetch_add(vals[i] * x[cols[i]]);
            });
        });
    }

    unsigned long long rows() const { return rows_; }

private:
    sycl::queue queue_{sycl::default_selector_v};
    unsigned long long rows_ = 0, cols_ = 0, nnz_ = 0;
    unsigned long long* row_indices_ = nullptr;
    unsigned long long* col_indices_ = nullptr;
    T* values_ = nullptr;
    T* scratch_ = nullptr;

    template <typename U>
    U* copy_dev_(const std::vector<U>& host) {
        if (host.empty()) return nullptr;
        U* d = sycl::malloc_device<U>(host.size(), queue_);
        if (!d) throw std::runtime_error("sycl_coo_spmv: alloc failed");
        queue_.memcpy(d, host.data(), sizeof(U) * host.size()).wait();
        return d;
    }

    void zero_out_() {
        row_indices_ = col_indices_ = nullptr;
        values_ = scratch_ = nullptr;
        rows_ = cols_ = nnz_ = 0;
    }

    template <typename U>
    void free_ptr_(U*& p) {
        if (p) { sycl::free(p, queue_); p = nullptr; }
    }

    void free_all_() {
        free_ptr_(row_indices_);
        free_ptr_(col_indices_);
        free_ptr_(values_);
        free_ptr_(scratch_);
        rows_ = cols_ = nnz_ = 0;
    }
};

template<typename T>
struct ConvSyclSparseMatrix {
    struct conv_group_desc {
        int coeff_offset;
        int coeff_count;
    };

    sycl::queue queue_;
    int MatRows_ = 0;
    bool Anchor_at_max_ = false;

    int num_groups_ = 0;
    int total_conv_rows_ = 0;
    int total_stencil_entries_ = 0;

    conv_group_desc* d_groups_ = nullptr;
    int* d_conv_rows_ = nullptr;
    int* d_conv_anchors_ = nullptr;
    int* d_conv_group_id_ = nullptr;
    T* d_stencil_vals_ = nullptr;
    int* d_stencil_col_offsets_ = nullptr;

    sycl_coo_spmv<T> residual_spmv_;
    int num_residual_rows_ = 0;
    int* d_non_all_zero_rows_ = nullptr;
    T* d_compact_buf_ = nullptr;

    T* multiply_scratch_ = nullptr;

    Eigen::SparseMatrix<T, Eigen::RowMajor> original_mat_;

    ConvSyclSparseMatrix() = default;

    explicit ConvSyclSparseMatrix(sycl::queue q) : queue_(q) {}

    ConvSyclSparseMatrix(const ConvSyclSparseMatrix&) = delete;
    ConvSyclSparseMatrix& operator=(const ConvSyclSparseMatrix&) = delete;

    ConvSyclSparseMatrix(ConvSyclSparseMatrix&& other) noexcept
        : queue_(other.queue_),
          MatRows_(other.MatRows_),
          Anchor_at_max_(other.Anchor_at_max_),
          num_groups_(other.num_groups_),
          total_conv_rows_(other.total_conv_rows_),
          total_stencil_entries_(other.total_stencil_entries_),
          d_groups_(other.d_groups_),
          d_conv_rows_(other.d_conv_rows_),
          d_conv_anchors_(other.d_conv_anchors_),
          d_conv_group_id_(other.d_conv_group_id_),
          d_stencil_vals_(other.d_stencil_vals_),
          d_stencil_col_offsets_(other.d_stencil_col_offsets_),
          residual_spmv_(std::move(other.residual_spmv_)),
          num_residual_rows_(other.num_residual_rows_),
          d_non_all_zero_rows_(other.d_non_all_zero_rows_),
          d_compact_buf_(other.d_compact_buf_),
          multiply_scratch_(other.multiply_scratch_),
          original_mat_(std::move(other.original_mat_)) {
        other.zero_pointers_();
    }

    ConvSyclSparseMatrix& operator=(ConvSyclSparseMatrix&& other) noexcept {
        if (this != &other) {
            free_device_storage_();
            queue_ = other.queue_;
            MatRows_ = other.MatRows_;
            Anchor_at_max_ = other.Anchor_at_max_;
            num_groups_ = other.num_groups_;
            total_conv_rows_ = other.total_conv_rows_;
            total_stencil_entries_ = other.total_stencil_entries_;
            d_groups_ = other.d_groups_;
            d_conv_rows_ = other.d_conv_rows_;
            d_conv_anchors_ = other.d_conv_anchors_;
            d_conv_group_id_ = other.d_conv_group_id_;
            d_stencil_vals_ = other.d_stencil_vals_;
            d_stencil_col_offsets_ = other.d_stencil_col_offsets_;
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

    ~ConvSyclSparseMatrix() {
        free_device_storage_();
    }

    template<int Eigen_T2, typename Eigen_T3>
    void set(const Eigen::SparseMatrix<T, Eigen_T2, Eigen_T3>& mat) {
        free_device_storage_();
        if (sycl_vector<T>::default_queue_ != nullptr) {
            queue_ = *sycl_vector<T>::default_queue_;
        }
        MatRows_ = mat.rows();
        original_mat_ = mat;

        ConvSparseMatrix<T> cpu_analyzer;
        cpu_analyzer.set(mat, 0);
        upload_residual_data_(cpu_analyzer);
        alloc_multiply_scratch_();
    }

    template<int Eigen_T2, typename Eigen_T3>
    void set(const Eigen::SparseMatrix<T, Eigen_T2, Eigen_T3>& mat, int num_conv_mask) {
        free_device_storage_();

        ConvSparseMatrix<T> cpu_analyzer;
        cpu_analyzer.set(mat, num_conv_mask);

        MatRows_ = cpu_analyzer.MatRows;
        Anchor_at_max_ = cpu_analyzer.Anchor_at_max;
        original_mat_ = cpu_analyzer.original_mat;

        upload_conv_data_(cpu_analyzer);
        upload_residual_data_(cpu_analyzer);
        alloc_multiply_scratch_();
    }

    void multiply(const T* d_in, T* d_out) {
        if (MatRows_ == 0) return;

        queue_.memset(d_out, 0, sizeof(T) * MatRows_);

        if (num_residual_rows_ > 0) {
            residual_spmv_.apply(d_in, d_compact_buf_);
            scatter_kernel_(d_out);
        }

        if (total_conv_rows_ > 0) {
            conv_kernel_(d_in, d_out);
        }
    }

    void multiply_add(const T* d_in, const T* d_b, T* d_out, T b_coeff, T ans_coeff) {
        multiply(d_in, multiply_scratch_);

        int N = MatRows_;
        T* scratch = multiply_scratch_;
        queue_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                d_out[idx] = scratch[idx] * ans_coeff + d_b[idx] * b_coeff * ans_coeff;
            });
        });
    }

    int rows() const { return MatRows_; }
    int cols() const { return MatRows_; }

    T coeff(int row, int col) const {
        return original_mat_.coeff(row, col);
    }

private:
    void zero_pointers_() {
        d_groups_ = nullptr;
        d_conv_rows_ = nullptr;
        d_conv_anchors_ = nullptr;
        d_conv_group_id_ = nullptr;
        d_stencil_vals_ = nullptr;
        d_stencil_col_offsets_ = nullptr;
        d_non_all_zero_rows_ = nullptr;
        d_compact_buf_ = nullptr;
        multiply_scratch_ = nullptr;
        num_groups_ = 0;
        total_conv_rows_ = 0;
        total_stencil_entries_ = 0;
        num_residual_rows_ = 0;
        MatRows_ = 0;
    }

    template<typename U>
    U* alloc_copy_(const std::vector<U>& host) {
        if (host.empty()) return nullptr;
        U* d = sycl::malloc_device<U>(host.size(), queue_);
        if (d == nullptr) {
            throw std::runtime_error("ConvSyclSparseMatrix: device allocation failed");
        }
        queue_.memcpy(d, host.data(), sizeof(U) * host.size()).wait();
        return d;
    }

    template<typename U>
    void free_ptr_(U*& ptr) {
        if (ptr != nullptr) {
            sycl::free(ptr, queue_);
            ptr = nullptr;
        }
    }

    void alloc_multiply_scratch_() {
        if (MatRows_ > 0) {
            multiply_scratch_ = sycl::malloc_device<T>(MatRows_, queue_);
            if (multiply_scratch_ == nullptr) {
                throw std::runtime_error("ConvSyclSparseMatrix: scratch allocation failed");
            }
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

        std::vector<conv_group_desc> h_groups(num_groups_);
        std::vector<int> h_conv_rows;
        std::vector<int> h_conv_anchors;
        std::vector<int> h_conv_group_id;
        std::vector<T> h_stencil_vals;
        std::vector<int> h_stencil_col_offsets;

        int row_offset = 0;
        int coeff_offset = 0;
        for (int g = 0; g < num_groups_; g++) {
            const auto& stencil = convCmpt[g].first;
            const auto& rows = convCmpt[g].second;
            int stencil_size = static_cast<int>(stencil.size());
            int row_count = static_cast<int>(rows.size());

            h_groups[g].coeff_offset = coeff_offset;
            h_groups[g].coeff_count = stencil_size;

            for (int c = 0; c < stencil_size; c++) {
                h_stencil_vals.push_back(stencil[c].first);
                h_stencil_col_offsets.push_back(stencil[c].second);
            }

            for (int r = 0; r < row_count; r++) {
                h_conv_rows.push_back(rows[r]);
                h_conv_group_id.push_back(g);
                if (Anchor_at_max_) {
                    h_conv_anchors.push_back(convs_Anchor[g][r]);
                }
            }

            coeff_offset += stencil_size;
            row_offset += row_count;
        }

        total_conv_rows_ = static_cast<int>(h_conv_rows.size());
        total_stencil_entries_ = static_cast<int>(h_stencil_vals.size());

        d_groups_ = alloc_copy_(h_groups);
        d_conv_rows_ = alloc_copy_(h_conv_rows);
        d_conv_group_id_ = alloc_copy_(h_conv_group_id);
        d_stencil_vals_ = alloc_copy_(h_stencil_vals);
        d_stencil_col_offsets_ = alloc_copy_(h_stencil_col_offsets);
        if (Anchor_at_max_) {
            d_conv_anchors_ = alloc_copy_(h_conv_anchors);
        }
    }

    void upload_residual_data_(const ConvSparseMatrix<T>& cpu) {
        num_residual_rows_ = static_cast<int>(cpu.non_all_zero_rows.size());

        if (num_residual_rows_ > 0) {
            d_non_all_zero_rows_ = alloc_copy_(cpu.non_all_zero_rows);

            Eigen::SparseMatrix<T> col_rest = cpu.restMat;
            residual_spmv_ = sycl_coo_spmv<T>(col_rest, queue_);

            d_compact_buf_ = sycl::malloc_device<T>(num_residual_rows_, queue_);
            if (d_compact_buf_ == nullptr) {
                throw std::runtime_error("ConvSyclSparseMatrix: compact buffer allocation failed");
            }
        }
    }

    void scatter_kernel_(T* d_out) {
        int count = num_residual_rows_;
        int* rows = d_non_all_zero_rows_;
        T* compact = d_compact_buf_;

        queue_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
                d_out[rows[idx]] = compact[idx];
            });
        });
    }

    void conv_kernel_(const T* d_in, T* d_out) {
        conv_group_desc* groups = d_groups_;
        int* rows = d_conv_rows_;
        int* anchors = d_conv_anchors_;
        int* group_ids = d_conv_group_id_;
        T* s_vals = d_stencil_vals_;
        int* s_offsets = d_stencil_col_offsets_;
        bool anchor_mode = Anchor_at_max_;
        int total = total_conv_rows_;
        const T* in = d_in;
        T* out = d_out;

        queue_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(total), [=](sycl::id<1> idx) {
                int row = rows[idx];
                int anchor = anchor_mode ? anchors[idx] : row;
                int g = group_ids[idx];
                int c_off = groups[g].coeff_offset;
                int c_cnt = groups[g].coeff_count;
                T v = T(0);
                for (int c = 0; c < c_cnt; c++) {
                    v += s_vals[c_off + c] * in[s_offsets[c_off + c] + anchor];
                }
                out[row] = v;
            });
        });
    }

    void free_device_storage_() {
        free_ptr_(d_groups_);
        free_ptr_(d_conv_rows_);
        free_ptr_(d_conv_anchors_);
        free_ptr_(d_conv_group_id_);
        free_ptr_(d_stencil_vals_);
        free_ptr_(d_stencil_col_offsets_);
        free_ptr_(d_non_all_zero_rows_);
        free_ptr_(d_compact_buf_);
        free_ptr_(multiply_scratch_);
        residual_spmv_ = sycl_coo_spmv<T>();
        num_groups_ = 0;
        total_conv_rows_ = 0;
        total_stencil_entries_ = 0;
        num_residual_rows_ = 0;
        MatRows_ = 0;
    }
};

template <typename T>
struct matrix_traits<ConvSyclSparseMatrix<T>> {
    using Scalar = T;
    using Vector = sycl_vector<T>;

    static void multiply(ConvSyclSparseMatrix<T>& mat, const Vector& in, Vector& out) {
        mat.multiply(in.data(), out.data());
    }

    static void multiply_add(ConvSyclSparseMatrix<T>& mat, const Vector& x,
                             const Vector& b, Vector& out, T b_coeff = T(1), T ans_coeff = T(1)) {
        mat.multiply_add(x.data(), b.data(), out.data(), b_coeff, ans_coeff);
    }

    static void set(ConvSyclSparseMatrix<T>& dest, const Eigen::SparseMatrix<double>& src) {
        if constexpr (std::is_same_v<T, double>) {
            dest.set(src);
        } else {
            Eigen::SparseMatrix<T> src_t = src.cast<T>();
            dest.set(src_t);
        }
    }
};

}

}

}

}

#endif
