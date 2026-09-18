#ifndef PSUM_FIELD_SOLVER_SPMV_SYCL_SPMV_CALCULATOR_HPP
#define PSUM_FIELD_SOLVER_SPMV_SYCL_SPMV_CALCULATOR_HPP

#include <Eigen/SparseCore>
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <vector>
#include "spmv_calculator.hpp"

namespace psum {

namespace field_solver {

namespace spmv {

class sycl_spmv_calculator : public spmv_calculator {
public:
    inline sycl_spmv_calculator() = default;

    explicit inline sycl_spmv_calculator(sycl::queue queue) : queue_(queue) {}

    explicit inline sycl_spmv_calculator(const Eigen::SparseMatrix<double>& matrix) {
        set_matrix(matrix);
    }

    inline sycl_spmv_calculator(const Eigen::SparseMatrix<double>& matrix, sycl::queue queue)
        : queue_(queue) {
        set_matrix(matrix);
    }

    sycl_spmv_calculator(const sycl_spmv_calculator&) = delete;
    sycl_spmv_calculator& operator=(const sycl_spmv_calculator&) = delete;

    inline sycl_spmv_calculator(sycl_spmv_calculator&& other) noexcept
        : queue_(other.queue_),
          rows_(other.rows_),
          cols_(other.cols_),
          nnz_(other.nnz_),
          row_indices_(other.row_indices_),
          col_indices_(other.col_indices_),
          values_(other.values_),
          scratch_(other.scratch_) {
        other.rows_ = 0;
        other.cols_ = 0;
        other.nnz_ = 0;
        other.row_indices_ = nullptr;
        other.col_indices_ = nullptr;
        other.values_ = nullptr;
        other.scratch_ = nullptr;
    }

    inline sycl_spmv_calculator& operator=(sycl_spmv_calculator&& other) noexcept {
        if (this != &other) {
            free_device_storage_();
            queue_ = other.queue_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            nnz_ = other.nnz_;
            row_indices_ = other.row_indices_;
            col_indices_ = other.col_indices_;
            values_ = other.values_;
            scratch_ = other.scratch_;
            other.rows_ = 0;
            other.cols_ = 0;
            other.nnz_ = 0;
            other.row_indices_ = nullptr;
            other.col_indices_ = nullptr;
            other.values_ = nullptr;
            other.scratch_ = nullptr;
        }
        return *this;
    }

    inline ~sycl_spmv_calculator() override {
        free_device_storage_();
    }

    inline void set_matrix(const Eigen::SparseMatrix<double>& matrix) override {
        free_device_storage_();

        rows_ = static_cast<unsigned long long>(matrix.rows());
        cols_ = static_cast<unsigned long long>(matrix.cols());
        nnz_ = static_cast<unsigned long long>(matrix.nonZeros());

        std::vector<unsigned long long> host_rows;
        std::vector<unsigned long long> host_cols;
        std::vector<double> host_values;
        host_rows.reserve(nnz_);
        host_cols.reserve(nnz_);
        host_values.reserve(nnz_);

        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, k); it; ++it) {
                host_rows.push_back(static_cast<unsigned long long>(it.row()));
                host_cols.push_back(static_cast<unsigned long long>(it.col()));
                host_values.push_back(it.value());
            }
        }

        row_indices_ = copy_to_device_(host_rows);
        col_indices_ = copy_to_device_(host_cols);
        values_ = copy_to_device_(host_values);
        scratch_ = rows_ == 0 ? nullptr : sycl::malloc_device<double>(rows_, queue_);
        if (rows_ > 0 && scratch_ == nullptr) {
            throw std::runtime_error("sycl_spmv_calculator: failed to allocate scratch buffer");
        }
    }

    inline void apply(const double* x, double* y) override {
        if (rows_ > 0) {
            queue_.memset(y, 0, sizeof(double) * rows_).wait();
        }
        if (nnz_ == 0) {
            return;
        }

        unsigned long long* rows = row_indices_;
        unsigned long long* cols = col_indices_;
        double* vals = values_;
        unsigned long long nnz = nnz_;

        queue_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nnz), [=](sycl::id<1> item) {
                unsigned long long i = item[0];
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space> ref(y[rows[i]]);
                ref.fetch_add(vals[i] * x[cols[i]]);
            });
        }).wait();
    }

    inline void apply_inplace(double* buf) override {
        if (rows_ != cols_) {
            throw std::runtime_error("sycl_spmv_calculator::apply_inplace requires a square matrix");
        }
        apply(buf, scratch_);
        queue_.memcpy(buf, scratch_, sizeof(double) * rows_).wait();
    }

    inline unsigned long long rows() const override {
        return rows_;
    }

    inline unsigned long long cols() const override {
        return cols_;
    }

private:
    sycl::queue queue_{sycl::default_selector_v};
    unsigned long long rows_ = 0;
    unsigned long long cols_ = 0;
    unsigned long long nnz_ = 0;
    unsigned long long* row_indices_ = nullptr;
    unsigned long long* col_indices_ = nullptr;
    double* values_ = nullptr;
    double* scratch_ = nullptr;

    template <typename T>
    inline T* copy_to_device_(const std::vector<T>& host) {
        if (host.empty()) return nullptr;
        T* device = sycl::malloc_device<T>(host.size(), queue_);
        if (device == nullptr) {
            throw std::runtime_error("sycl_spmv_calculator: failed to allocate device storage");
        }
        queue_.memcpy(device, host.data(), sizeof(T) * host.size()).wait();
        return device;
    }

    template <typename T>
    inline void free_device_ptr_(T*& ptr) {
        if (ptr != nullptr) {
            sycl::free(ptr, queue_);
            ptr = nullptr;
        }
    }

    inline void free_device_storage_() {
        free_device_ptr_(row_indices_);
        free_device_ptr_(col_indices_);
        free_device_ptr_(values_);
        free_device_ptr_(scratch_);
        rows_ = 0;
        cols_ = 0;
        nnz_ = 0;
    }
};

}

}

}

#endif
