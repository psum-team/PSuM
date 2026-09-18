#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SYCL_INNER_SOLVER_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SYCL_INNER_SOLVER_HPP

#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>
#include "sycl_vector.hpp"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template<typename T = double>
struct sycl_inner_solver {
    static constexpr int max_coarse_size_ = 8192;

    int size_ = 0;
    T* d_inv_ = nullptr;

    sycl_inner_solver() = default;

    sycl_inner_solver(const sycl_inner_solver&) = delete;
    sycl_inner_solver& operator=(const sycl_inner_solver&) = delete;

    sycl_inner_solver(sycl_inner_solver&& o) noexcept
        : size_(o.size_), d_inv_(o.d_inv_) {
        o.size_ = 0;
        o.d_inv_ = nullptr;
    }

    sycl_inner_solver& operator=(sycl_inner_solver&& o) noexcept {
        if (this != &o) {
            free_();
            size_ = o.size_;
            d_inv_ = o.d_inv_;
            o.size_ = 0;
            o.d_inv_ = nullptr;
        }
        return *this;
    }

    ~sycl_inner_solver() {
        free_();
    }

    void compute(const Eigen::SparseMatrix<double>& A) {
        free_();
        size_ = static_cast<int>(A.rows());
        if (size_ > max_coarse_size_) {
            throw std::runtime_error("sycl_inner_solver: coarse matrix too large (" +
                std::to_string(size_) + " > " + std::to_string(max_coarse_size_) + ")");
        }

        Eigen::MatrixXd inv = Eigen::MatrixXd(A).inverse();
        std::vector<T> h_inv(size_ * size_);
        for (int i = 0; i < size_; i++)
            for (int j = 0; j < size_; j++)
                h_inv[i * size_ + j] = static_cast<T>(inv(i, j));

        auto* q = sycl_vector<T>::default_queue_;
        d_inv_ = sycl::malloc_device<T>(size_ * size_, *q);
        if (d_inv_ == nullptr) {
            throw std::runtime_error("sycl_inner_solver: device allocation failed");
        }
        q->memcpy(d_inv_, h_inv.data(), sizeof(T) * size_ * size_).wait();
    }

    sycl_vector<T> solve(const sycl_vector<T>& b) const {
        int n = size_;
        sycl_vector<T> result;
        result.resize(n);
        solve_to(b, result);
        return result;
    }

    void solve_to(const sycl_vector<T>& b, sycl_vector<T>& result) const {
        int n = size_;
        result.resize(n);
        auto* q = sycl_vector<T>::default_queue_;
        T* d_out = result.data();
        const T* d_b = b.data();
        T* d_mat = d_inv_;
        int stride = n;

        q->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                T sum = T(0);
                for (int j = 0; j < stride; j++) {
                    sum += d_mat[idx * stride + j] * d_b[j];
                }
                d_out[idx] = sum;
            });
        });
    }

private:
    void free_() {
        if (d_inv_ != nullptr && sycl_vector<T>::default_queue_ != nullptr) {
            sycl::free(d_inv_, *sycl_vector<T>::default_queue_);
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
