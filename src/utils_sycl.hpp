#ifndef UTILS_SYCL_HPP
#define UTILS_SYCL_HPP

#include <sycl/sycl.hpp>

namespace psum {

// is is a slight wrapper for existing sycl functions.
namespace utils_sycl {

    template <typename Func>
    concept handler_to_device_func =
        requires(Func f, sycl::handler &h, const size_t &i) {
            { f(h)(i) } -> std::same_as<void>;
        };

    template <typename FuncType>
    requires handler_to_device_func<FuncType>
    void for_each(sycl::queue& q_, size_t loop_size, FuncType&& func) {
        q_.submit([&](sycl::handler& h) {
            auto v_func = func(h);
            h.parallel_for(sycl::range<1>(loop_size), [=](sycl::id<1> idx) {
                v_func(idx);
            });
        }).wait();
    }

    template <typename ScalarType, typename RType>
    void atomic_add(ScalarType& a, const RType& w) {
        auto ref_v = sycl::atomic_ref<ScalarType,
                                      sycl::memory_order::relaxed,
                                      sycl::memory_scope::device>(a);
        ref_v.fetch_add(w);
    }

    class single_shared_variable_manager {
    private:
        std::vector<std::pair<void*, sycl::queue>> allocations_;
        std::mutex mutex_;

        single_shared_variable_manager() = default;
        ~single_shared_variable_manager() {
            for (auto& [ptr, q] : allocations_)
                sycl::free(ptr, q);
        }

        single_shared_variable_manager(const single_shared_variable_manager&) = delete;
        single_shared_variable_manager& operator=(const single_shared_variable_manager&) = delete;

    public:
        static single_shared_variable_manager& instance() {
            static single_shared_variable_manager inst;
            return inst;
        }

        template <typename T>
        T* allocate(sycl::queue& q) {
            std::lock_guard<std::mutex> lock(mutex_);
            T* ptr = sycl::malloc_shared<T>(1, q);
            allocations_.push_back(std::make_pair(reinterpret_cast<void*>(ptr), q));
            return ptr;
        }
    };

    template <typename Value>
    Value* shared_variable(sycl::queue& q) {
        return single_shared_variable_manager::instance().allocate<Value>(q);
    }

    struct view_as_spin_lock {
        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device> atomic_var;
        view_as_spin_lock(int& lock_var) : atomic_var(lock_var) {}
        template <int max_spin_count = 4000>
        bool try_lock() {
            for (int i = 0; i < max_spin_count; ++i) {
                if (atomic_var.load(sycl::memory_order::relaxed) == 0) {
                    if (atomic_var.exchange(1, sycl::memory_order::acquire) == 0) 
                        return true;
                }
                sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
            }
            return false;
        }
        void unlock() {
            atomic_var.store(0, sycl::memory_order::release);
        }
    };
}


}

#endif