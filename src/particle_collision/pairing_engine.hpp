#ifndef PSUM_PARTICLE_COLLISION_PAIRING_ENGINE_HPP
#define PSUM_PARTICLE_COLLISION_PAIRING_ENGINE_HPP

#include <sycl/sycl.hpp>
#include "grid_mapped_device_vector.hpp"
#include "../random.hpp"

namespace psum {

namespace particle_collision {

    template<typename ParticleContainer, typename Grid>
    struct pairing_engine {
    private:
        using pos_tag = psum::tag::property::position;
        using pType = ParticleContainer::value_type;
        using Position = std::remove_cvref_t<decltype(psum::tag::get<pos_tag>(std::declval<pType>()))>;
        psum::field::device_array<uint32_t> n_counts_;
        psum::field::device_array<uint32_t> ticket_counters_;
        grid_mapped_device_vector<Position, std::pair<pType*, pType*>, Grid> mapped_queue;
        using random_seed = psum::tag::property::random_seed;

    public:
        pairing_engine(sycl::queue q, Grid grid): n_counts_(q), ticket_counters_(q), mapped_queue(q, grid) {
            auto n_cells_on_dim = grid.get_cell_num();
            size_t n_cells = 1;
            for(auto n : n_cells_on_dim)
                n_cells *= n;

            n_counts_ = psum::field::device_array<uint32_t>(q, n_cells);
            ticket_counters_ = psum::field::device_array<uint32_t>(q, n_cells);
        }

        void deal(
            ParticleContainer& particles,
            psum::particle_container::device_vector<std::pair<Position, std::pair<pType*, pType*>>>& input_data,
            float oversample_factor = 2.0f
        ) {
            mapped_queue.update(input_data);

            size_t n_p = particles.size();
            size_t num_cells = n_counts_.size();
            auto q = n_counts_.get_queue();
            n_counts_.fill(0);
            ticket_counters_.fill(0);
            q.wait();

            particles.for_each([&](sycl::handler& h) {
                auto n_ptr = n_counts_.data();
                auto g = mapped_queue.grid();
                return [=](const pType& p) {
                    uint32_t gid = g.c2i(g.nC(psum::tag::get<pos_tag>(p)));
                    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, 
                                     sycl::memory_scope::device, 
                                     sycl::access::address_space::global_space> ref(n_ptr[gid]);
                    ref.fetch_add(1);
                };
            });

            particles.for_each([&](sycl::handler& h) {
                auto q_mapped_acc = mapped_queue.get_access(h);
                auto n_ptr = n_counts_.data();
                auto t_ptr = ticket_counters_.data();
                auto g = mapped_queue.grid();
                return [=](pType &p)
                {
                    uint32_t gid = g.c2i(g.nC(psum::tag::get<pos_tag>(p)));
                    uint32_t M = q_mapped_acc.partition_size(gid);
                    uint32_t N = n_ptr[gid];

                    if (M == 0 || N == 0) return;
                    if (t_ptr[gid] >= M) return;
                    
                    auto R = psum::random::view_as_rander(psum::tag::get<random_seed>(p));
                    float r = R();

                    float threshold = oversample_factor * (M < 3.0 ? 3.0f : float(M)) / (float)N;
                    if (r < threshold) {
                        sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, 
                                         sycl::memory_scope::device, 
                                         sycl::access::address_space::global_space> ref(t_ptr[gid]);
                        uint32_t ticket = ref.fetch_add(1);

                        if (ticket < M) {
                            auto& q_item_pair = q_mapped_acc(gid, ticket);
                            q_item_pair.second.second = &p;
                        }
                    }
                };
            });

            mapped_queue.content().get_queue().memcpy(
                input_data.data(), mapped_queue.content().data(), 
                mapped_queue.content().size() * sizeof(std::pair<Position, std::pair<pType*, pType*>>)
            ).wait();
        }
    };

}

}

#endif