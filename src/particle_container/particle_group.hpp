#ifndef PSUM_PARTICLE_CONTAINER_PARTICLE_CONTAINER_HPP
#define PSUM_PARTICLE_CONTAINER_PARTICLE_CONTAINER_HPP

#include "device_vector.hpp"
#include "validator.hpp"

namespace psum {

namespace particle_container {

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

        size_t capacity() const {
            return data_.capacity();
        }

        void clear() {
            data_.resize(0);
            invalid_indexes_.resize(0);
        }

        void shuffle() {
            compress();
            const size_t data_size = data_.size();
            if (data_size == 0) return;
            if (data_size != size())
                throw std::runtime_error("Error: invalid size in particle_group::shuffle");
            device_vector<Particle> old_data(q_, data_size);
            q_.memcpy(old_data.data(), data_.data(), data_size * sizeof(Particle)).wait();

            uint32_t total_bits = 0;
            while ((size_t(1) << total_bits) < data_size) total_bits++;
            if (total_bits % 2 != 0) total_bits++; 
            uint32_t half_bits = total_bits / 2;
            uint64_t mask = (1ULL << half_bits) - 1;

            std::random_device rd;
            std::mt19937_64 gen(rd());
            size_t seed = gen();

            q_.submit([&](sycl::handler& h) {
                auto data_acc = data_.get_access(h);
                auto old_acc = old_data.get_access(h);

                h.parallel_for(sycl::range<1>(data_size), [=](sycl::id<1> item) {
                    size_t idx = item[0];
                    size_t current_idx = idx;

                    // // use Feistel network for permutation
                    auto feistel_op = [mask, half_bits](size_t x, size_t s) {
                        uint64_t L = x & mask;
                        uint64_t R = x >> half_bits;
                        
                        for (int i = 0; i < 4; ++i) {
                            uint64_t f = R ^ (s + i);
                            f ^= f >> 33;
                            f *= 0xff51afd7ed558ccdULL;
                            f ^= f >> 33;
                            f *= 0xc4ceb9fe1a85ec53ULL;
                            f ^= f >> 33;

                            uint64_t nextL = R;
                            uint64_t nextR = L ^ (f & mask); 
                            L = nextL;
                            R = nextR;
                        }
                        return (R << half_bits) | L;
                    };

                    // Cycle-Walking
                    do {
                        current_idx = feistel_op(current_idx, seed);
                    } while (current_idx >= data_size);

                    data_acc[current_idx] = old_acc[idx];
                });
            }).wait();
        }

        template <bool NeedValidataion>
        void insert(const device_vector<Particle>& particles) requires(NeedValidataion == false) {
            if (particles.size() == 0) return;
            if (particles.get_queue().get_device() != q_.get_device()) {
                throw std::runtime_error("Error: queue mismatch in particle_group::insert");
            }
            size_t original_size = data_.size();
            size_t insert_size = particles.size();
            size_t invalid_size = invalid_indexes_.size();
            if (invalid_size == 0) {
                data_.resize(original_size + insert_size); // may call reserve
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
                invalid_indexes_.reserve(data_.capacity()); // keep same capacity
                q_.memcpy(data_.data() + original_size, particles.data() + invalid_size, rest_insert_size * sizeof(Particle)).wait();
            }
        }

        template <bool NeedValidation = true>
        void insert(const device_vector<Particle>& particles_in) {
            if (particles_in.size() == 0) return;
            device_vector<Particle> particles(particles_in.get_queue(), particles_in.size());
            particles_in.for_each([&](sycl::handler &h) {
                auto particles_acc = particles.get_access(h);
                return [=](const Particle& p) {
                    if (validator::is_valid(p)) {
                        particles_acc.push_back(p);
                    }
                };
            });
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