#ifndef PSUM_PARTICLE_COLLISION_DEVICE_EXCLUSIVE_SCAN_HPP
#define PSUM_PARTICLE_COLLISION_DEVICE_EXCLUSIVE_SCAN_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <sycl/sycl.hpp>
#include "../../include/psum/tag.hpp"
#include "../random.hpp"

namespace psum {

namespace particle_collision {

    // WG_SIZE must be a power of 2
    template <typename T, int WG_SIZE = 256>
    struct device_exclusive_scan_context {
        sycl::queue q;

        size_t capacity = 0;
        size_t max_groups = 0;

        T* block_sums = nullptr;
        T* block_offsets = nullptr;

        T* temp_sums = nullptr;
        T* temp_offsets = nullptr;

        void init(sycl::queue& queue) {
            q = queue;
        }

        void release() {
            if (block_sums)    sycl::free(block_sums, q);
            if (block_offsets) sycl::free(block_offsets, q);
            if (temp_sums)     sycl::free(temp_sums, q);
            if (temp_offsets)  sycl::free(temp_offsets, q);

            block_sums = nullptr;
            block_offsets = nullptr;
            temp_sums = nullptr;
            temp_offsets = nullptr;

            capacity = 0;
            max_groups = 0;
        }

        device_exclusive_scan_context(): q(sycl::queue{sycl::default_selector()}) {};
        ~device_exclusive_scan_context() { release();}
        device_exclusive_scan_context(const device_exclusive_scan_context&) = delete;
        device_exclusive_scan_context& operator=(const device_exclusive_scan_context&) = delete;

        void alloc_for_size(size_t N) {
            size_t needed_groups = (N + WG_SIZE - 1) / WG_SIZE;

            if (needed_groups <= max_groups)
                return;

            release();

            max_groups = needed_groups;

            size_t temp_size = needed_groups;
            size_t total_temp = 0;

            while (temp_size > 1) {
                temp_size = (temp_size + WG_SIZE - 1) / WG_SIZE;
                total_temp += temp_size;
            }

            block_sums    = sycl::malloc_device<T>(max_groups, q);
            block_offsets = sycl::malloc_device<T>(max_groups, q);

            temp_sums    = sycl::malloc_device<T>(total_temp, q);
            temp_offsets = sycl::malloc_device<T>(total_temp, q);
        }
    };

    template <typename T, int WG_SIZE>
    void local_scan(sycl::queue& q, T* input, T* output, T* block_sums, size_t N) {
        size_t num_groups = (N + WG_SIZE - 1) / WG_SIZE;

        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<T, 1> local(WG_SIZE, h);

            h.parallel_for(
                sycl::nd_range<1>(num_groups * WG_SIZE, WG_SIZE),
                [=](sycl::nd_item<1> item) {
                    size_t gid = item.get_global_id(0);
                    size_t lid = item.get_local_id(0);
                    size_t group = item.get_group(0);

                    T val = (gid < N) ? input[gid] : 0;
                    local[lid] = val;
                    item.barrier();

                    for (size_t stride = 1; stride < WG_SIZE; stride <<= 1) {
                        size_t idx = (lid + 1) * stride * 2 - 1;
                        if (idx < WG_SIZE)
                            local[idx] += local[idx - stride];
                        item.barrier();
                    }

                    if (lid == 0)
                        block_sums[group] = local[WG_SIZE - 1];

                    if (lid == 0)
                        local[WG_SIZE - 1] = 0;

                    item.barrier();

                    for (size_t stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
                        size_t idx = (lid + 1) * stride * 2 - 1;
                        if (idx < WG_SIZE) {
                            T t = local[idx - stride];
                            local[idx - stride] = local[idx];
                            local[idx] += t;
                        }
                        item.barrier();
                    }

                    if (gid < N)
                        output[gid] = local[lid];
                });
        }).wait();
    }

    template <typename T, int WG_SIZE>
    void scan_block_sums(device_exclusive_scan_context<T, WG_SIZE>& ctx, T* in, T* out, size_t n, size_t level_offset) {
        auto& q = ctx.q;

        if (n <= WG_SIZE) {
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<T, 1> local(WG_SIZE, h);

                h.parallel_for(
                    sycl::nd_range<1>(WG_SIZE, WG_SIZE),
                    [local, n, in, out](sycl::nd_item<1> item) {
                        size_t lid = item.get_local_id(0);

                        local[lid] = (lid < n) ? in[lid] : 0;
                        item.barrier();

                        for (size_t stride = 1; stride < WG_SIZE; stride <<= 1) {
                            size_t idx = (lid + 1) * stride * 2 - 1;
                            if (idx < WG_SIZE)
                                local[idx] += local[idx - stride];
                            item.barrier();
                        }

                        if (lid == 0)
                            local[WG_SIZE - 1] = 0;

                        item.barrier();

                        for (size_t stride = WG_SIZE >> 1; stride > 0; stride >>= 1) {
                            size_t idx = (lid + 1) * stride * 2 - 1;
                            if (idx < WG_SIZE) {
                                T t = local[idx - stride];
                                local[idx - stride] = local[idx];
                                local[idx] += t;
                            }
                            item.barrier();
                        }

                        if (lid < n)
                            out[lid] = local[lid];
                    });
            }).wait();
            return;
        }

        size_t groups = (n + WG_SIZE - 1) / WG_SIZE;

        T* next_sums    = ctx.temp_sums + level_offset;
        T* next_offsets = ctx.temp_offsets + level_offset;

        local_scan<T, WG_SIZE>(q, in, out, next_sums, n);

        scan_block_sums<T, WG_SIZE>(
            ctx, next_sums, next_offsets, groups,
            level_offset + groups);

        q.submit([&](sycl::handler& h) {
            h.parallel_for(
                sycl::range<1>(n),
                [=](sycl::id<1> i) {
                    size_t gid = i[0];
                    size_t g = gid / WG_SIZE;
                    out[gid] += next_offsets[g];
                });
        }).wait();
    }

    template <typename T, int WG_SIZE = 256>
    T exclusive_scan_device(device_exclusive_scan_context<T, WG_SIZE>& ctx, T* input, T* output, size_t N) {
        if (N == 0) return T{};
        ctx.alloc_for_size(N);

        auto& q = ctx.q;

        size_t num_groups = (N + WG_SIZE - 1) / WG_SIZE;

        // Phase 1
        local_scan<T, WG_SIZE>(
            q, input, output,
            ctx.block_sums, N);

        // Phase 2
        scan_block_sums<T, WG_SIZE>(
            ctx,
            ctx.block_sums,
            ctx.block_offsets,
            num_groups,
            0);

        // Phase 3
        q.submit([&](sycl::handler& h) {
            T* offsets = ctx.block_offsets;
            h.parallel_for(
                sycl::range<1>(N),
                [offsets, output](sycl::id<1> i) {
                    size_t gid = i[0];
                    size_t g = gid / WG_SIZE;
                    output[gid] += offsets[g];
                });
        }).wait();

        // total sum
        T last, last_input;
        q.memcpy(&last, &output[N - 1], sizeof(T)).wait();
        q.memcpy(&last_input, &input[N - 1], sizeof(T)).wait();

        return last + last_input;
    }

}

}

#endif