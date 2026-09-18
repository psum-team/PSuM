#ifndef PSUM_PARTICLE_BOUNDARY_DEVICE_PARTITIONED_ARRAY_HPP
#define PSUM_PARTICLE_BOUNDARY_DEVICE_PARTITIONED_ARRAY_HPP

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <algorithm>
#include "../field/device_array.hpp"

namespace psum {

namespace particle_boundary {

    template <typename T>
    class device_partitioned_array_acc {
        using array_acc_size = field::device_array_acc<size_t>;
        using array_acc_data = field::device_array_acc<T>;
    public:
        using value_type = T;

        device_partitioned_array_acc(
            const array_acc_size& prefix_sum,
            const array_acc_size& partition_sizes,
            const array_acc_data& data
        ) : prefix_sum_(prefix_sum), 
            partition_sizes_(partition_sizes),
            data_(data) {}

        inline size_t num_partitions() const {
            return prefix_sum_.size();
        }

        inline size_t data_size() const {
            return data_.size();
        }

        inline size_t partition_size(size_t partition_idx) const {
            return partition_sizes_[partition_idx];
        }

        inline size_t partition_offset(size_t partition_idx) const {
            if (partition_idx == 0) {
                return 0;
            } else if (partition_idx < num_partitions()) {
                return prefix_sum_[partition_idx - 1];
            }
            return data_size();
        }

        inline T& operator()(size_t partition_idx, size_t element_idx) const {
            size_t offset = partition_offset(partition_idx);
            return data_[offset + element_idx];
        }

        inline T& operator()(size_t global_idx) const {
            return data_[global_idx];
        }

    private:
        array_acc_size prefix_sum_;
        array_acc_size partition_sizes_;
        array_acc_data data_;
    };

    template <typename T>
    class device_partitioned_array {
    public:
        using value_type = T;
        using acc_type = device_partitioned_array_acc<T>;
        using is_device_container_t = std::true_type;

        device_partitioned_array(const sycl::queue& q, size_t num_partitions = 1)
            : q_(q),
              num_partitions_(num_partitions),
              prefix_sum_(q, num_partitions),
              partition_sizes_(q, num_partitions),
              data_(q) {
        }

        device_partitioned_array(
            const sycl::queue& q,
            const std::vector<std::vector<T>>& host_data
        ) : q_(q), num_partitions_(host_data.size()),
              prefix_sum_(q, 1),
              partition_sizes_(q, 1),
              data_(q, 1) {
            if (num_partitions_ == 0) {
                throw std::runtime_error("Error: num_partitions cannot be zero.");
            }

            std::vector<size_t> prefix_sum_host(num_partitions_);
            std::vector<size_t> partition_sizes_host(num_partitions_);
            size_t total_size = 0;
            for (size_t i = 0; i < num_partitions_; ++i) {
                total_size += host_data[i].size();
                prefix_sum_host[i] = total_size;
                partition_sizes_host[i] = host_data[i].size();
            }

            std::vector<T> data_host_flat;
            data_host_flat.reserve(total_size);
            for (const auto& partition : host_data) {
                data_host_flat.insert(data_host_flat.end(), partition.begin(), partition.end());
            }

            prefix_sum_ = field::device_array<size_t>(q, prefix_sum_host);
            partition_sizes_ = field::device_array<size_t>(q, partition_sizes_host);
            data_ = field::device_array<T>(q, data_host_flat);
        }

        ~device_partitioned_array() = default;

        device_partitioned_array(const device_partitioned_array&) = delete;
        device_partitioned_array& operator=(const device_partitioned_array&) = delete;

        device_partitioned_array(device_partitioned_array&& other) noexcept
            : q_(other.q_),
              num_partitions_(other.num_partitions_),
              prefix_sum_(std::move(other.prefix_sum_)),
              partition_sizes_(std::move(other.partition_sizes_)),
              data_(std::move(other.data_)) {
            other.num_partitions_ = 0;
        }

        device_partitioned_array& operator=(device_partitioned_array&& other) noexcept {
            if (this != &other) {
                q_ = other.q_;
                num_partitions_ = other.num_partitions_;
                prefix_sum_ = std::move(other.prefix_sum_);
                partition_sizes_ = std::move(other.partition_sizes_);
                data_ = std::move(other.data_);
                other.num_partitions_ = 0;
            }
            return *this;
        }

        size_t num_partitions() const {
            return num_partitions_;
        }

        size_t data_size() const {
            return data_.size();
        }

        size_t* prefix_sum_data() const {
            return prefix_sum_.data();
        }

        T* data() const {
            return data_.data();
        }

        acc_type get_access(sycl::handler& h) const {
            return acc_type(prefix_sum_.get_access(h), partition_sizes_.get_access(h), data_.get_access(h));
        }

        std::vector<std::vector<T>> to_host() const {
            std::vector<std::vector<T>> result(num_partitions_);
            auto prefix_sum_host = prefix_sum_.to_host();
            auto data_flat = data_.to_host();

            for (size_t i = 0; i < num_partitions_; ++i) {
                size_t partition_size;
                size_t offset;

                if (i == 0) {
                    partition_size = prefix_sum_host[0];
                    offset = 0;
                } else {
                    partition_size = prefix_sum_host[i] - prefix_sum_host[i - 1];
                    offset = prefix_sum_host[i - 1];
                }

                result[i].assign(data_flat.begin() + offset, data_flat.begin() + offset + partition_size);
            }

            return result;
        }

        sycl::queue get_queue() const {
            return q_;
        }

    private:
        mutable sycl::queue q_;
        size_t num_partitions_;
        field::device_array<size_t> prefix_sum_;
        field::device_array<size_t> partition_sizes_;
        field::device_array<T> data_;
    };

}

}

#endif
