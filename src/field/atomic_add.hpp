#ifndef PSUM_FIELD_ATOMIC_ADD_HPP
#define PSUM_FIELD_ATOMIC_ADD_HPP

#include <concepts>
#include <array>
#include <ostream>
#include <Eigen/Dense>
#include <sycl/sycl.hpp>

namespace psum {

namespace field {

    // atomic add for SYCL device memory
    template <typename ScalarType, int QuantitySize, typename LType, typename RType>
    void _atomic_add_(LType& a, const RType& w) {
        if constexpr (QuantitySize == 1) {
            auto ref_v = sycl::atomic_ref<ScalarType,
                                          sycl::memory_order::relaxed,
                                          sycl::memory_scope::device>(a);
            ref_v.fetch_add(w);
        } else {
            for (int i = 0; i < QuantitySize; i++) {
                auto ref_v = sycl::atomic_ref<ScalarType,
                                              sycl::memory_order::relaxed,
                                              sycl::memory_scope::device>(a[i]);
                ref_v.fetch_add(w[i]);
            }
        }
    }
    
   // atomic add is used to accumulate value on device memory
    template <typename ScalarType, int QuantitySize, typename LType, typename RType>
    void _atomic_add_(LType& a, const RType& w, const ScalarType& c) {
        if constexpr (QuantitySize == 1) {
            auto ref_v = sycl::atomic_ref<ScalarType,
                                          sycl::memory_order::relaxed,
                                          sycl::memory_scope::device>(a);
            ref_v.fetch_add(w * c);
        } else {
            for (int i = 0; i < QuantitySize; i++) {
                auto ref_v = sycl::atomic_ref<ScalarType,
                                              sycl::memory_order::relaxed,
                                              sycl::memory_scope::device>(a[i]);
                ref_v.fetch_add(w[i] * c);
            }
        }
    }
}

}

#endif