#ifndef PSUM_PARTICLE_COLLISION_GRID_MAPPED_DEVICE_VECTOR_HPP
#define PSUM_PARTICLE_COLLISION_GRID_MAPPED_DEVICE_VECTOR_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../../include/psum/particle_container.hpp"
#include "../field/device_array.hpp"
#include "device_exclusive_scan.hpp"

namespace psum {

namespace particle_collision {

    template<typename Position, typename Data, typename Grid>
    struct grid_mapped_device_vector {
    private:
        psum::particle_container::device_vector<std::pair<Position, Data>> data_;
        psum::field::device_array<size_t> prefix_sum_;
        psum::field::device_array<size_t> partition_sizes_;
        psum::field::device_array<size_t> partition_counters_;
        device_exclusive_scan_context<size_t> scan_ctx_;
        size_t *sum_shared;
        Grid grid_;
    public:
        grid_mapped_device_vector(sycl::queue q, Grid g): data_(q), prefix_sum_(q), partition_sizes_(q), partition_counters_(q), grid_(g) {
            auto n_cells_on_dim = grid_.get_cell_num();
            size_t n_cells = 1;
            for(auto n : n_cells_on_dim)
                n_cells *= n;
            prefix_sum_ = psum::field::device_array<size_t>(q, n_cells);
            partition_sizes_ = psum::field::device_array<size_t>(q, n_cells);
            partition_counters_ = psum::field::device_array<size_t>(q, n_cells);
            scan_ctx_.init(q);

            sum_shared = sycl::malloc_shared<size_t>(1, q);
        }
        ~grid_mapped_device_vector() {
            sycl::free(sum_shared, data_.get_queue());
        }
        void update(psum::particle_container::device_vector<std::pair<Position, Data>>& input_data) {
            auto q = input_data.get_queue();
            if (q.get_device() != data_.get_queue().get_device()) {
                throw std::runtime_error("Error: queue mismatch in grid_mapped_device_vector::update().");
            }
            
            size_t n = input_data.size();
            size_t num_cells = partition_sizes_.size();

            partition_sizes_.fill(0);
            partition_counters_.fill(0);
            q.wait();

            q.submit([&](sycl::handler& h) {
                auto acc = input_data.get_access(h);
                auto sizes_ptr = partition_sizes_.data();
                Grid g = grid_;
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    if (!g.inGrid(acc[i].first)) return;
                    uint32_t gid = g.c2i(g.nC(acc[i].first));
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, 
                                    sycl::memory_scope::device, 
                                    sycl::access::address_space::global_space> ref(sizes_ptr[gid]);
                    ref.fetch_add(1);
                });
            }).wait();

            *sum_shared = exclusive_scan_device<size_t>(scan_ctx_, partition_sizes_.data(), prefix_sum_.data(), num_cells);

            data_.resize(*sum_shared);

            q.submit([&](sycl::handler& h) {
                auto in_acc = input_data.get_access(h);
                auto out_acc = data_.get_access(h);
                auto offsets = prefix_sum_.data();
                auto counts = partition_counters_.data();
                Grid g = grid_;
                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    if (!g.inGrid(in_acc[i].first)) return;
                    uint32_t gid = g.c2i(g.nC(in_acc[i].first));
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, 
                                    sycl::memory_scope::device, 
                                    sycl::access::address_space::global_space> ref(counts[gid]);
                    size_t local_idx = ref.fetch_add(1);
                    out_acc[offsets[gid] + local_idx] = in_acc[i];
                });
            }).wait();
        }
    
        struct grid_mapped_device_vector_acc {
        private:
            using array_acc_size = psum::field::device_array_acc<size_t>;
            using array_acc_data = psum::particle_container::device_vector_acc<std::pair<Position, Data>>;
            array_acc_size prefix_sum_;
            array_acc_size partition_sizes_;
            array_acc_data data_;
        public:
            grid_mapped_device_vector_acc(
                const array_acc_size& prefix_sum,
                const array_acc_size& partition_sizes,
                const array_acc_data& data
            ) : prefix_sum_(prefix_sum), 
                partition_sizes_(partition_sizes),
                data_(data) {}

            inline size_t num_partitions() const {
                return prefix_sum_.size();
            }

            inline size_t partition_size(size_t partition_idx) const {
                return partition_sizes_[partition_idx];
            }

            inline size_t partition_offset(size_t partition_idx) const {
                if (partition_idx == 0) {
                    return 0;
                } else if (partition_idx < num_partitions()) {
                    return prefix_sum_[partition_idx];
                } else return prefix_sum_[num_partitions() - 1] + partition_size(num_partitions() - 1);
            }

            inline std::pair<Position, Data>& operator()(size_t partition_idx, size_t element_idx) const {
                size_t offset = partition_offset(partition_idx);
                return data_[offset + element_idx];
            }
        };

        using acc_type = grid_mapped_device_vector_acc;

        acc_type get_access(sycl::handler& h) {
            return acc_type(
                prefix_sum_.get_access(h),
                partition_sizes_.get_access(h),
                data_.get_access(h)
            );
        }

        auto& content() {
            return data_;
        }

        Grid& grid() {
            return grid_;
        }
    };

}

}

#endif