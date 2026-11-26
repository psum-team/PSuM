#ifndef PSUM_PARTICLE_CONTAINER_PARTICLE_CONTAINER_HPP
#define PSUM_PARTICLE_CONTAINER_PARTICLE_CONTAINER_HPP

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include "device_vector.hpp"
#include "validator.hpp"

namespace psum {

namespace particle_container {

    namespace detail {
        class slot_permutation {
            std::size_t extent_;
            std::array<std::uint64_t, 4> round_keys_;
            unsigned half_bits_;
            std::uint64_t mask_;

        public:
            constexpr slot_permutation(std::size_t extent, std::array<std::uint64_t, 4> round_keys)
                : extent_(extent), round_keys_(round_keys),
                  half_bits_(extent ? (std::bit_width(extent - 1) + 1) / 2 : 0),
                  mask_((std::uint64_t{1} << half_bits_) - 1) {}

            constexpr slot_permutation(std::size_t extent, std::uint64_t seed)
                : slot_permutation(extent, {seed, seed + 1, seed + 2, seed + 3}) {}

            // Map a slot in [0, extent_) back into that range by cycle walking.
            constexpr std::size_t operator()(std::size_t index) const {
                do {
                    std::uint64_t left = index & mask_;
                    std::uint64_t right = index >> half_bits_;
                    for (unsigned round = 0; round < 4; ++round) {
                        std::uint64_t mixed = right ^ round_keys_[round];
                        mixed ^= mixed >> 33;
                        mixed *= 0xff51afd7ed558ccdULL;
                        mixed ^= mixed >> 33;
                        mixed *= 0xc4ceb9fe1a85ec53ULL;
                        mixed ^= mixed >> 33;
                        const auto next = left ^ (mixed & mask_);
                        left = right;
                        right = next;
                    }
                    index = (right << half_bits_) | left;
                } while (index >= extent_);
                return index;
            }
        };
    }

    template<typename Particle, template<typename> typename Validator>
    requires is_validator<Validator<Particle>, Particle>
    class particle_group {
    public:
        using value_type = Particle;
        using validator = Validator<Particle>;
        using is_device_container_t = std::true_type;

        particle_group(sycl::queue& q, size_t capacity = 8192) : q_(q), data_(q, capacity), invalid_indexes_(q, capacity) {}
        ~particle_group() {}

        particle_group(const particle_group<Particle, Validator>&) = delete;
        particle_group(particle_group<Particle, Validator>&&) = delete;
        particle_group<Particle, Validator>& operator=(const particle_group<Particle, Validator>&) = delete;
        particle_group<Particle, Validator>& operator=(particle_group<Particle, Validator>&&) = delete;

        void reserve(size_t capacity) {
            data_.reserve(capacity);
            invalid_indexes_.reserve(capacity);
        }

        size_t size() const {
            return data_.size() - invalid_indexes_.size();
        }

        // Number of addressable slots, including invalid particles.
        size_t extent() const { return extent_; }

        size_t capacity() const {
            return data_.capacity();
        }

        void clear() {
            data_.resize(0);
            invalid_indexes_.resize(0);
            extent_=0;
        }

        void shuffle() {
            compress();
            const size_t data_size = data_.size();
            if (data_size == 0) return;
            if (data_size != size())
                throw std::runtime_error("Error: invalid size in particle_group::shuffle");
            device_vector<Particle> old_data(q_, data_size);
            q_.memcpy(old_data.data(), data_.data(), data_size * sizeof(Particle)).wait();

            std::random_device rd;
            std::mt19937_64 gen(rd());
            const detail::slot_permutation permute(data_size, static_cast<size_t>(gen()));

            q_.submit([&](sycl::handler& h) {
                auto data_acc = data_.get_access(h);
                auto old_acc = old_data.get_access(h);

                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> item) {
                    data_acc[permute(item[0])] = old_acc[item[0]];
                });
            }).wait();
        }

        template <bool NeedValidataion>
        void insert(const device_vector<Particle>& particles) requires(NeedValidataion == false) {
            const size_t insert_size = particles.size();
            if (insert_size == 0) return;
            if (particles.get_queue().get_device() != q_.get_device()) {
                throw std::runtime_error("Error: queue mismatch in particle_group::insert");
            }
            size_t original_size = data_.size();
            size_t invalid_size = invalid_indexes_.size();
            if (invalid_size == 0) {
                data_.resize(original_size + insert_size); // may call reserve
                extent_=original_size+insert_size;
                invalid_indexes_.reserve(data_.capacity()); // keep same capacity
                q_.memcpy(data_.data() + original_size, particles.data(), insert_size * sizeof(Particle)).wait();
            } else if(insert_size <= invalid_size) {
                // reuse invalid indexes and data_.size() keep unchanged
                q_.submit([&](sycl::handler& h) {
                    auto data_acc = data_.get_access(h);
                    auto invalid_indexes_acc = invalid_indexes_.get_access(h);
                    auto particles_acc = particles.get_access(h);
                    h.parallel_for(sycl::range<1>(insert_size), [=](sycl::id<1> idx) {
                        data_acc[invalid_indexes_acc[invalid_size - 1 - idx]] = particles_acc[idx];
                    });
                }).wait();
                invalid_indexes_.resize(invalid_size - insert_size);
            } else {
                // invalid_size < insert_size
                // two stage insertion
                // use all invalid positions
                q_.submit([&](sycl::handler& h) {
                    auto data_acc = data_.get_access(h);
                    auto invalid_indexes_acc = invalid_indexes_.get_access(h);
                    auto particles_acc = particles.get_access(h);
                    h.parallel_for(sycl::range<1>(invalid_size), [=](sycl::id<1> idx) {
                        data_acc[invalid_indexes_acc[idx]] = particles_acc[idx];
                    });
                }).wait();
                invalid_indexes_.resize(0);
                size_t rest_insert_size = insert_size - invalid_size;
                data_.resize(original_size + rest_insert_size); // may call reserve
                extent_=original_size+rest_insert_size;
                invalid_indexes_.reserve(data_.capacity()); // keep same capacity
                q_.memcpy(data_.data() + original_size, particles.data() + invalid_size, rest_insert_size * sizeof(Particle)).wait();
            }
        }

        template <bool NeedValidation = true>
        void insert(const device_vector<Particle>& particles_in) {
            const size_t count = particles_in.size();
            if (count == 0) return;
            auto q = particles_in.get_queue();
            device_vector<Particle> particles(q, count);
            q.submit([&](sycl::handler& h) {
                auto source = particles_in.get_access(h);
                auto valid = particles.get_access(h);
                h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
                    const auto& p = source[i[0]];
                    if (validator::is_valid(p)) valid.push_back(p);
                });
            }).wait();
            insert<false>(particles);
        }

        template <bool NeedValidation = true>
        void insert(const std::vector<Particle>& particles) {
            if (particles.empty()) return;
            device_vector<Particle> d_vec(q_, particles);
            insert<NeedValidation>(d_vec);
        }

        template <typename FuncType>
        requires handler_to_device_func_const<FuncType, Particle>
        void for_each(FuncType&& func) const {
            size_t data_size = data_.size();
            if (data_size == 0) return;  // zero-size launch => CU:1 on CUDA backend
            q_.submit([&](sycl::handler& h) {
                auto data_acc = data_.get_access(h);
                auto p_func = func(h);
                if (data_size < 1024) {
                    constexpr size_t wg = 128;
                    h.parallel_for(sycl::nd_range<1>{sycl::range<1>((data_size + wg - 1) / wg * wg), sycl::range<1>(wg)}, [=](sycl::nd_item<1> item) {
                        if (size_t idx = item.get_global_id(0); idx < data_size && validator::is_valid(data_acc[idx]))
                            p_func(data_acc[idx]);
                    });
                    return;
                }
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    if (validator::is_valid(data_acc[idx])) {
                        p_func(data_acc[idx]);
                    }
                });
            }).wait();
        }

        template <typename FuncType>
        requires handler_to_device_func_mutable<FuncType, Particle>
        void for_each(FuncType&& func) {
            size_t data_size = data_.size();
            if (data_size == 0) return;  // zero-size launch => CU:1 on CUDA backend
            q_.submit([&](sycl::handler& h) {
                auto data_acc = data_.get_access(h);
                auto invalid_indexes_acc = invalid_indexes_.get_access(h);
                auto p_func = func(h);
                if (data_size < 1024) {
                    constexpr size_t wg = 128;
                    h.parallel_for(sycl::nd_range<1>{sycl::range<1>((data_size + wg - 1) / wg * wg), sycl::range<1>(wg)}, [=](sycl::nd_item<1> item) {
                        if (size_t idx = item.get_global_id(0); idx < data_size && validator::is_valid(data_acc[idx])) {
                            p_func(data_acc[idx]);
                            if (!validator::is_valid(data_acc[idx]))
                                invalid_indexes_acc.push_back(idx);
                        }
                    });
                    return;
                }
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    if (validator::is_valid(data_acc[idx])) {
                        p_func(data_acc[idx]);
                        if (!validator::is_valid(data_acc[idx]))
                            invalid_indexes_acc.push_back(idx);
                    }
                });
            }).wait();
        }

        void compress() {
            if (size() == 0) {clear(); return;}
            device_vector<Particle> compressed(q_, size());
            size_t data_size = data_.size();
            q_.submit([&](sycl::handler& h) {
                auto data_acc = data_.get_access(h);
                auto compressed_acc = compressed.get_access(h);
                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> idx) {
                    if (validator::is_valid(data_acc[idx])) {
                        compressed_acc.push_back(data_acc[idx]);
                    }
                });
            }).wait();
            clear();
            insert(compressed);
        }

        void shrink(size_t new_capacity) {
            compress();
            size_t old_size = size();
            size_t true_capacity = std::max(new_capacity, old_size);
            if (true_capacity == 0) return;
            device_vector<Particle> new_data(q_, true_capacity);
            device_vector<size_t> new_invalid_indexes(q_, true_capacity);
            q_.memcpy(new_data.data(), data_.data(), old_size * sizeof(Particle)).wait();
            data_ = std::move(new_data);
            data_.resize(old_size);
            extent_=old_size;
            invalid_indexes_ = std::move(new_invalid_indexes);
        }

        const auto& get_content() const {
            return data_;
        }

        const auto& get_invalid_indexes() const {
            return invalid_indexes_;
        }

    private:
        sycl::queue& q_;
        device_vector<Particle> data_;
        device_vector<size_t> invalid_indexes_;
        size_t extent_=0;
    };

}

namespace serialization {

    template<typename Particle, template<typename> typename Validator>
    void save(mas_file& fp, const std::string& name, const psum::particle_container::particle_group<Particle, Validator>& obj) {
        save(fp, name + ".data", obj.get_content());
    }

    template<typename Particle, template<typename> typename Validator>
    void load(mas_file& fp, const std::string& name, psum::particle_container::particle_group<Particle, Validator>& obj) {
        sycl::queue q{ sycl::default_selector_v };
        psum::particle_container::device_vector<Particle> d_vec(q);
        load(fp, name + ".data", d_vec);
        obj.clear();
        obj.insert(d_vec);
    }

}

}

#endif
