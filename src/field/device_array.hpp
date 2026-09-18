#ifndef PSUM_FIELD_DEVICE_ARRAY_HPP
#define PSUM_FIELD_DEVICE_ARRAY_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include "../serialization/container_sl.hpp"

namespace psum {

namespace field {

    template <typename Func, typename Data>
    concept handler_to_device_func_const =
        requires(Func f, sycl::handler &h, const Data &p) {
            { f(h)(p) } -> std::same_as<void>;
        };

    template <typename Func, typename Data>
    concept handler_to_device_func_mutable =
        requires(Func f, sycl::handler &h, Data &p) {
            { f(h)(p) } -> std::same_as<void>;
        };

    template <typename T>
    class device_array_acc {
    public:
        using value_type = T;

        device_array_acc(T* data, size_t size)
            : data_(data), size_(size) {}

        inline size_t size() const {
            return size_;
        }

        /**
         * this function is not designed to be thread-safe for same index.
         */
        inline T& operator[](size_t idx) const {
            return data_[idx];
        }

        T* data() const {
            return data_;
        }

    private:
        T* data_;
        size_t size_;
    };

    template <typename T>
    class device_array {
    public:
        using value_type = T;
        using acc_type = device_array_acc<T>;
        using is_device_container_t = std::true_type;

        device_array(const sycl::queue& q, size_t size = 8192) : q_(q), size_(size) {
            data_ = sycl::malloc_device<T>(size_, q_);
            if (!data_)
                throw std::bad_alloc();
        }

        device_array(const sycl::queue& q, const std::vector<T>& host_vec) : device_array(q, host_vec.size()) {
            if constexpr (!std::is_same_v<T, bool>) {
                q_.memcpy(data_, host_vec.data(), host_vec.size() * sizeof(T)).wait();
            } else {
                std::unique_ptr<bool[]> raw(new bool[host_vec.size()]);
                for (size_t i = 0; i < host_vec.size(); ++i)
                    raw[i] = host_vec[i];
                q_.memcpy(data_, raw.get(), host_vec.size() * sizeof(bool)).wait();
            }
        }

        ~device_array() {
            if (data_) sycl::free(data_, q_);
        }

        device_array(const device_array&) = delete;
        device_array& operator=(const device_array&) = delete;

        device_array(device_array&& other) noexcept
            : q_(other.q_), data_(other.data_), size_(other.size_) {
            other.data_ = nullptr;
            other.size_ = 0;
        }
        
        device_array& operator=(device_array&& other) noexcept {
            if (this != &other) {
                if (data_) sycl::free(data_, q_);
            
                q_ = other.q_;
                data_ = other.data_;
                size_ = other.size_;
            
                other.data_ = nullptr;
                other.size_ = 0;
            }
            return *this;
        }

        size_t size() const {
            return size_;
        }

        T* data() const {
            return data_;
        }

        // 'h' is just a placeholder to limit the scope of 'get_access'.
        acc_type get_access(sycl::handler& h) const {
            return acc_type(data_, size_);
        }

        std::vector<T> to_host() const {
            std::vector<T> result(size());
            if constexpr (!std::is_same_v<T, bool>) {
                q_.memcpy(result.data(), data_, size() * sizeof(T)).wait();
            } else {
                std::unique_ptr<bool[]> raw(new bool[size()]);
                q_.memcpy(raw.get(), data_, size() * sizeof(bool)).wait();
                result.assign(raw.get(), raw.get() + size());
            }
            return result;
        }

        void copy(const std::vector<T>& host_vec) {
            if (host_vec.size() != size_)
                throw std::runtime_error("Error: host_vec size not match.");
            if constexpr (!std::is_same_v<T, bool>) {
                q_.memcpy(data_, host_vec.data(), host_vec.size() * sizeof(T)).wait();
            } else {
                std::unique_ptr<bool[]> raw(new bool[host_vec.size()]);
                for (size_t i = 0; i < host_vec.size(); ++i)
                    raw[i] = host_vec[i];
                q_.memcpy(data_, raw.get(), host_vec.size() * sizeof(bool)).wait();
            }
        }

        sycl::queue get_queue() const { return q_; }

        template <typename FuncType>
        requires handler_to_device_func_const<FuncType, T>
        void for_each(FuncType&& func) const {
            size_t data_size = size();
            q_.submit([&](sycl::handler& h) {
                auto data_acc = get_access(h);
                auto v_func = func(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    v_func(data_acc[idx]);
                });
            }).wait();
        }

        template <typename FuncType>
        requires handler_to_device_func_mutable<FuncType, T>
        void for_each(FuncType&& func) {
            size_t data_size = size();
            q_.submit([&](sycl::handler& h) {
                auto data_acc = get_access(h);
                auto v_func = func(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    v_func(data_acc[idx]);
                });
            }).wait();
        }

        void fill(const T& value) {
            for_each([value](sycl::handler& h) {
                return [value](T& data) {
                    data = value;
                };
            });
        }
        
    private:
        mutable sycl::queue q_;
        T* data_;
        size_t size_;
    };

}

namespace serialization {

    template <typename T>
    void save(mas_file& fp, const std::string& name, const psum::field::device_array<T>& obj) {
        sycl::queue q = obj.get_queue();
        std::string device_name = q.get_device().get_info<sycl::info::device::name>();
        auto devices = q.get_device().get_platform().get_devices();
        int same_name_device_count = 0;
        for (auto& d : devices) {
            if (d.get_info<sycl::info::device::name>() == device_name) {
                same_name_device_count++;
                if (d == q.get_device())
                    break;
            }
        }
        std::string device_name_with_count = "[#" + std::to_string(same_name_device_count - 1) + "]" + device_name;
        save(fp, name + ".device", device_name_with_count);
        save(fp, name + ".content", obj.to_host());
    }

    template <typename T>
    void load(mas_file& fp, const std::string& name, psum::field::device_array<T>& obj) {
        std::string device_name;
        load(fp, name + ".device", device_name);
        auto devices = sycl::platform(sycl::default_selector()).get_devices();
        int device_count = 0;
        std::string::size_type pos = device_name.find("[#");
        if (pos != std::string::npos) {
            std::string::size_type end_pos = device_name.find("]", pos);
            if (end_pos != std::string::npos) {
                std::string count_str = device_name.substr(pos + 2, end_pos - pos - 2);
                device_count = std::stoi(count_str);
                device_name = device_name.substr(end_pos + 1);
            }
        }
        int same_name_device_count = 0;
        for (auto& d : devices) {
            if (d.get_info<sycl::info::device::name>() == device_name) {
                same_name_device_count++;
                if (device_count == 0 || same_name_device_count == device_count + 1) {
                    std::vector<T> host_vec;
                    load(fp, name + ".content", host_vec);
                    obj = psum::field::device_array<T>(sycl::queue(d), host_vec);
                    return;
                }
            }
        }
        // device not found.
        throw std::runtime_error("Error: device not found.");
    }
}

}

#endif