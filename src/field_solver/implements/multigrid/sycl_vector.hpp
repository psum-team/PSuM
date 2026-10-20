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
    int size_ = 0; // size_ = rows() * batch_size()
    int batch_ = 1;

    sycl_vector() = default;

    sycl_vector(const sycl_vector&) = delete;
    sycl_vector& operator=(const sycl_vector&) = delete;

    sycl_vector(sycl_vector&& o) noexcept
        : data_(o.data_), size_(o.size_), batch_(o.batch_) {
        o.data_ = nullptr;
        o.size_ = 0;
        o.batch_ = 1;
    }

    sycl_vector& operator=(sycl_vector&& o) noexcept {
        if (this != &o) {
            free_();
            data_ = o.data_;
            size_ = o.size_;
            batch_ = o.batch_;
            o.data_ = nullptr;
            o.size_ = 0;
            o.batch_ = 1;
        }
        return *this;
    }

    ~sycl_vector() {
        free_();
    }

    void resize(int n, int batch = 1) {
        if (n * batch == size_ && batch == batch_) return;
        free_();
        if (n > 0 && batch > 0) {
            data_ = sycl::malloc_device<T>(n * batch, *default_queue_);
            if (data_ == nullptr) {
                throw std::runtime_error("sycl_vector: device allocation failed");
            }
            size_ = n * batch;
            batch_ = batch;
        }
    }

    int rows() const { return size_ / batch_; }

    int batch_size() const { return batch_; }

    void setZero() {
        if (data_ && size_ > 0) {
            T* d = data_;
            default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(size_), [=](sycl::id<1> idx) {
                    d[idx] = T(0);
                });
            });
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
        batch_ = 1;
    }
};

template <typename T>
struct vector_traits<sycl_vector<T>> {
    static int result_batch_(const sycl_vector<T>& a, const sycl_vector<T>& b) {
        if (a.rows() != b.rows()) {
            throw std::runtime_error("sycl_vector: incompatible row sizes");
        }
        int b_bigger = std::max(a.batch_size(), b.batch_size());
        int b_smaller = std::min(a.batch_size(), b.batch_size());
        if (b_smaller != 1 && b_smaller != b_bigger) {
            throw std::runtime_error("sycl_vector: incompatible batch sizes");
        }
        return b_bigger;
    }

    static sycl_vector<T> add(const sycl_vector<T>& a, const sycl_vector<T>& b) {
        int n = a.rows();
        sycl_vector<T> result;
        result.resize(n, result_batch_(a, b));
        int nb = result.batch_size();
        T* d_out = result.data();
        const T* d_a = a.data();
        const T* d_b = b.data();
        bool explicit_broadcast = (a.batch_size() != b.batch_size());
        if (explicit_broadcast) {
            const T* d_bigger = (a.batch_size() > b.batch_size()) ? d_a : d_b;
            const T* d_smaller = (a.batch_size() > b.batch_size()) ? d_b : d_a;
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                    int r = idx[0], c = idx[1];
                    d_out[r * nb + c] = d_bigger[r * nb + c] + d_smaller[r];
                });
            });
        } else {
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)n * nb), [=](sycl::id<1> idx) {
                    d_out[idx] = d_a[idx] + d_b[idx];
                });
            });
        }

        return result;
    }

    static sycl_vector<T> subtract(const sycl_vector<T>& a, const sycl_vector<T>& b) {
        int n = a.rows();
        sycl_vector<T> result;
        result.resize(n, result_batch_(a, b));
        T* d_out = result.data();
        const T* d_a = a.data();
        const T* d_b = b.data();
        bool explicit_broadcast = (a.batch_size() != b.batch_size());
        if (explicit_broadcast) {
            int nb = result.batch_size();
            if (a.batch_size() > b.batch_size()) {
                sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                        int r = idx[0], c = idx[1];
                        d_out[r * nb + c] = d_a[r * nb + c] - d_b[r];
                    });
                });
            } else {
                sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                        int r = idx[0], c = idx[1];
                        d_out[r * nb + c] = d_a[r] - d_b[r * nb + c];
                    });
                });
            }
        } else {
            int nb = a.batch_size();
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)n * nb), [=](sycl::id<1> idx) {
                    d_out[idx] = d_a[idx] - d_b[idx];
                });
            });
        }
        return result;
    }

    static sycl_vector<T> multiply_with_array(const sycl_vector<T>& vec, const sycl_vector<T>& arr) {
        int n = vec.rows();
        sycl_vector<T> result;
        result.resize(n, result_batch_(vec, arr));
        T* d_out = result.data();
        const T* d_v = vec.data();
        const T* d_a = arr.data();
        bool explicit_broadcast = (vec.batch_size() != arr.batch_size());
        if (explicit_broadcast) {
            const T* d_bigger = (vec.batch_size() > arr.batch_size()) ? d_v : d_a;
            const T* d_smaller = (vec.batch_size() > arr.batch_size()) ? d_a : d_v;
            int nb = result.batch_size();
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                    int r = idx[0], c = idx[1];
                    d_out[r * nb + c] = d_bigger[r * nb + c] * d_smaller[r];
                });
            });
        } else {
            int nb = vec.batch_size();
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)n * nb), [=](sycl::id<1> idx) {
                    d_out[idx] = d_v[idx] * d_a[idx];
                });
            });
        }
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
        result_batch_(dst, src);
        if (src.batch_size() == dst.batch_size()) {
            sycl_vector<T>::default_queue_->memcpy(dst.data(), src.data(),
                                       sizeof(T) * dst.rows() * dst.batch_size());
        } else if (src.batch_size() == 1) {
            // src.batch_size() == 1 && dst.batch_size() > 1: broadcast
            int n = dst.rows();
            int db = dst.batch_size();
            T* d_dst = dst.data();
            const T* d_src = src.data();
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(n, db), [=](sycl::id<2> idx) {
                    int r = idx[0], c = idx[1];
                    d_dst[r * db + c] = d_src[r];
                });
            });
        } else {
            // src.batch_size() > 1 && dst.batch_size() == 1
            throw std::runtime_error("sycl_vector: incompatible batch sizes for copy_data");
        }
    }

    static void hadamard_inplace(sycl_vector<T>& vec, const sycl_vector<T>& arr) {
        int n = vec.rows();
        int nb = vec.batch_size();
        result_batch_(vec, arr);
        T* d_v = vec.data();
        const T* d_a = arr.data();
        if (vec.batch_size() == arr.batch_size()) {
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)n * nb), [=](sycl::id<1> idx) {
                    d_v[idx] *= d_a[idx];
                });
            });
        } else if (arr.batch_size() == 1) {
            // arr.batch_size() == 1 && vec.batch_size() > 1: broadcast
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                    int r = idx[0], c = idx[1];
                    d_v[r * nb + c] *= d_a[r];
                });
            });
        } else {
            // arr.batch_size() > 1 && vec.batch_size() == 1
            throw std::runtime_error("sycl_vector: incompatible batch sizes for hadamard_inplace");
        }
    }

    static void hadamard_to(const sycl_vector<T>& in, const sycl_vector<T>& arr, sycl_vector<T>& out) {
        int n = out.rows();
        int nb = out.batch_size();
        if (result_batch_(in, arr) != nb)
            throw std::runtime_error("sycl_vector: output batch size mismatch");
        if (out.rows() != in.rows())
            throw std::runtime_error("sycl_vector: output row size mismatch");
        const T* d_in = in.data();
        const T* d_a = arr.data();
        T* d_out = out.data();
        bool explicit_broadcast = (in.batch_size() != arr.batch_size());
        if (explicit_broadcast) {
            const T* d_bigger = (in.batch_size() > arr.batch_size()) ? d_in : d_a;
            const T* d_smaller = (in.batch_size() > arr.batch_size()) ? d_a : d_in;
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<2>(n, nb), [=](sycl::id<2> idx) {
                    int r = idx[0], c = idx[1];
                    d_out[r * nb + c] = d_bigger[r * nb + c] * d_smaller[r];
                });
            });
        } else {
            sycl_vector<T>::default_queue_->submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>((size_t)n * nb), [=](sycl::id<1> idx) {
                    d_out[idx] = d_in[idx] * d_a[idx];
                });
            });
        }
    }

    // batch-aware resize seam (the vector carries the batch count itself)
    static void resize(sycl_vector<T>& dest, int n, int batch = 1) {
        dest.resize(n, batch);
    }

    static int batch_size(const sycl_vector<T>& v) { return v.batch_size(); }
};

}

}

}

}

#endif
