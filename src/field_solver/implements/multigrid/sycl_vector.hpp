#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SYCL_VECTOR_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SYCL_VECTOR_HPP

#include <sycl/sycl.hpp>
#include <vector>
#include <stdexcept>
#include <type_traits>
#include "sparse_linear_trait.hpp"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template<typename T = double>
struct sycl_vector {
    static inline sycl::queue* default_queue_ = nullptr;

    T* data_ = nullptr;
    int size_ = 0;

    sycl_vector() = default;

    sycl_vector(const sycl_vector&) = delete;
    sycl_vector& operator=(const sycl_vector&) = delete;

    sycl_vector(sycl_vector&& o) noexcept
        : data_(o.data_), size_(o.size_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    sycl_vector& operator=(sycl_vector&& o) noexcept {
        if (this != &o) {
            free_();
            data_ = o.data_;
            size_ = o.size_;
            o.data_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    ~sycl_vector() {
        free_();
    }

    void resize(int n) {
        if (n == size_) return;
        free_();
        if (n > 0) {
            data_ = sycl::malloc_device<T>(n, *default_queue_);
            if (data_ == nullptr) {
                throw std::runtime_error("sycl_vector: device allocation failed");
            }
            size_ = n;
        }
    }

    int rows() const { return size_; }

    void setZero() {
        if (data_ && size_ > 0) {
            default_queue_->memset(data_, 0, sizeof(T) * size_);
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }

private:
    void free_() {
        if (data_ != nullptr && default_queue_ != nullptr) {
            sycl::free(data_, *default_queue_);
        }
        data_ = nullptr;
        size_ = 0;
    }
};

template <typename T>
struct vector_traits<sycl_vector<T>> {
    static sycl_vector<T> add(const sycl_vector<T>& a, const sycl_vector<T>& b) {
        int n = a.rows();
        sycl_vector<T> result;
        result.resize(n);
        T* d_out = result.data();
        const T* d_a = a.data();
        const T* d_b = b.data();
        sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                d_out[idx] = d_a[idx] + d_b[idx];
            });
        });
        return result;
    }

    static sycl_vector<T> subtract(const sycl_vector<T>& a, const sycl_vector<T>& b) {
        int n = a.rows();
        sycl_vector<T> result;
        result.resize(n);
        T* d_out = result.data();
        const T* d_a = a.data();
        const T* d_b = b.data();
        sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                d_out[idx] = d_a[idx] - d_b[idx];
            });
        });
        return result;
    }

    static sycl_vector<T> multiply_with_array(const sycl_vector<T>& vec, const sycl_vector<T>& arr) {
        int n = vec.rows();
        sycl_vector<T> result;
        result.resize(n);
        T* d_out = result.data();
        const T* d_v = vec.data();
        const T* d_a = arr.data();
        sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                d_out[idx] = d_v[idx] * d_a[idx];
            });
        });
        return result;
    }

    static void set(std::vector<sycl_vector<T>>& dest, const std::vector<Eigen::ArrayXd>& src) {
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i].resize(static_cast<int>(src[i].size()));
            if constexpr (std::is_same_v<T, double>) {
                sycl_vector<T>::default_queue_->memcpy(dest[i].data(), src[i].data(),
                                           sizeof(double) * src[i].size());
            } else {
                std::vector<T> tmp(src[i].data(), src[i].data() + src[i].size());
                auto ev = sycl_vector<T>::default_queue_->memcpy(dest[i].data(), tmp.data(),
                                                     sizeof(T) * tmp.size());
                ev.wait();
            }
        }
    }

    static void copy_data(sycl_vector<T>& dst, const sycl_vector<T>& src) {
        sycl_vector<T>::default_queue_->memcpy(dst.data(), src.data(),
                                   sizeof(T) * dst.rows());
    }

    static void hadamard_inplace(sycl_vector<T>& vec, const sycl_vector<T>& arr) {
        int n = vec.rows();
        T* d_v = vec.data();
        const T* d_a = arr.data();
        sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                d_v[idx] *= d_a[idx];
            });
        });
    }

    static void hadamard_to(const sycl_vector<T>& in, const sycl_vector<T>& arr, sycl_vector<T>& out) {
        int n = in.rows();
        const T* d_in = in.data();
        const T* d_a = arr.data();
        T* d_out = out.data();
        sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                d_out[idx] = d_in[idx] * d_a[idx];
            });
        });
    }
};

}

}

}

}

#endif
