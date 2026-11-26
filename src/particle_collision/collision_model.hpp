#ifndef PSUM_PARTICLE_COLLISION_COLLISION_MODEL_HPP
#define PSUM_PARTICLE_COLLISION_COLLISION_MODEL_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <sycl/sycl.hpp>
#include "foundation.hpp"
#include "collision_executor.hpp"
#include "context/collision_channel_context.hpp"
#include "context/species_contexts.hpp"
#include "pairing_engine.hpp"

namespace psum::particle_collision {
    namespace foundation {
        template<class Group>
        using mcc_group=std::bool_constant<!std::is_base_of_v<
            deferred_pairing_collision_tag,std::tuple_element_t<0,Group>>>;
    }

    // Execute MCC before generating all DPMCC requests for each particle.
    // The caller controls when to insert buffered products.
    template<class CollisionTypes,class ContextType>
    void execute_incident_collisions(ContextType& ctx,double dt) {
        using incident_species=typename foundation::collision_model_types<CollisionTypes>::incident_tuple;
        tag::foundation::for_each_type<incident_species>([&]<class SpeciesTag> {
            auto& species=ctx.template get<SpeciesTag>();
            if constexpr(foundation::collision_model_types<CollisionTypes>::template has_same_species_dpmcc<SpeciesTag>)
                species.particle_counts.setZero();
            auto& group=species.get_group();
            using group_type=std::remove_cvref_t<decltype(group)>;
            using particle=typename group_type::value_type;
            using validator=typename group_type::validator;
            using groups=typename foundation::incident_collision_groups<SpeciesTag,CollisionTypes>::type;
            using stages=tag::foundation::partition_tuple<groups,foundation::mcc_group>;
            using ordered_groups=decltype(std::tuple_cat(
                std::declval<typename stages::matched>(),std::declval<typename stages::remaining>()));
            group.for_each([&](sycl::handler& h) {
                auto acc=ctx.get_access(h);
                return [=](particle& p) {
                    tag::foundation::for_each_type<ordered_groups>([&]<class Channels> {
                        if(!validator::is_valid(p)) return;
                        using first=std::tuple_element_t<0,Channels>;
                        if constexpr(std::is_base_of_v<deferred_pairing_collision_tag,first>) {
                            if constexpr(foundation::same_species_channel<first>::value) {
                                // All self channels form one group: count each MCC survivor once,
                                // including particles which do not emit a request.
                                auto counts=acc.template get<typename first::incident_species_tag>().particle_counts;
                                auto pos=foundation::projected_position<decltype(counts)::Dimension,SpeciesTag>(
                                    tag::get<tag::property::position>(p));
                                const auto& grid=counts.getGrid();
                                if(grid.inGrid(pos))
                                    sycl::atomic_ref<uint32_t,sycl::memory_order::relaxed,
                                        sycl::memory_scope::device,sycl::access::address_space::global_space>(
                                            counts(grid.c2i(grid.nC(pos)))).fetch_add(1);
                            }
                            foundation::collision_request_executor<Channels>::execute(p,acc);
                        } else {
                            foundation::collision_executor<Channels>::template execute<validator>(p,dt,acc);
                        }
                    });
                };
            });
        });
    }

    template<bool SameSpecies,
             class Engine,class Buffer,class Incident,class CollisionBody>
    auto bind_pairing_channel(Engine& engine,Buffer& buffer,Incident& incident,CollisionBody make_collision) {
        struct binding {
            Engine& engine;
            Buffer& buffer;
            Incident& incident;
            [[no_unique_address]] CollisionBody make_collision;
            static constexpr bool same_species() { return SameSpecies; }
        };
        return binding{engine,buffer,incident,std::move(make_collision)};
    }

    // All channels in a segment use the same grid, queue, and background species.
    // Requests must have unique incident endpoints across the segment.
    template<class Background,class... Channel>
    void deal(Background& background,std::tuple<Channel...> channels) {
        if constexpr(sizeof...(Channel)>0) {
            auto& particles=background.get_group();
            auto& lead=std::get<0>(channels).engine;
            using background_tag=typename std::remove_reference_t<decltype(lead)>::background_species_tag;
            using validator=typename std::remove_reference_t<decltype(particles)>::validator;
            const size_t extent=particles.extent();

            [&]<size_t... I>(std::index_sequence<I...>) {
                constexpr bool same_species=(Channel::same_species() && ...);
                static_assert(same_species || (!Channel::same_species() && ...),
                    "a collision segment must contain only same-species or only cross-species channels");
                auto* data=particles.get_content().data();
                uint8_t* incident_marks=nullptr;
                auto make_marker=[&] {
                    if constexpr(same_species) {
                        // Reset once, only after a channel has live requests; share across the segment.
                        if(!incident_marks)incident_marks=background.prepare_incident_marks(extent);
                        auto* marked=incident_marks;
                        return [=](const auto& payload) { marked[payload.first-data]=1; };
                    } else {
                        return [](const auto&) {};
                    }
                };
                auto prepare=[&](auto& channel) {
                    channel.engine.prepare_pairing(channel.buffer,[&](sycl::handler& h) {
                        return channel.incident.get_particle_access(h);
                    },make_marker);
                };
                (prepare(std::get<I>(channels)),...);

                std::array<size_t,sizeof...(Channel)+1> offsets{};
                auto add_requests=[&]<size_t J>() {
                    const size_t count=std::get<J>(channels).engine.total_size_;
                    offsets[J+1]=offsets[J]+count;
                };
                (add_requests.template operator()<I>(),...);
                const size_t requests=offsets.back();
                std::random_device random;
                const uint64_t seed=(uint64_t(random())<<32)|uint64_t(random());
                if(extent && requests) {
                    const size_t shift=std::uniform_int_distribution<size_t>{0,extent-1}(random);
                    const auto* marked=incident_marks;
                    const psum::particle_container::detail::slot_permutation permute(extent,seed);
                    lead.data_.get_queue().submit([&](sycl::handler& h) {
                        auto selectors=std::tuple{std::get<I>(channels).engine.selection_body(h)...};
                        auto grid=lead.grid_;
                        h.parallel_for(sycl::range<1>(extent),[=](sycl::id<1> item) {
                            size_t index=permute(item[0])+shift;
                            if(index>=extent)index-=extent;
                            if constexpr(same_species) if(marked[index])return;
                            auto& p=data[index];
                            if(!validator::is_valid(p))return;
                            auto pos=foundation::projected_position<decltype(grid)::Dimension,background_tag>(
                                psum::tag::get<psum::tag::property::position>(p));
                            if(!grid.inGrid(pos))return;
                            const uint32_t cell=grid.c2i(grid.nC(pos));
                            (void)(std::get<I>(selectors)(p,cell) || ...);
                        });
                    }).wait();
                }

                auto make_executor=[&](auto& channel,sycl::handler& h) {
                    auto collide=channel.make_collision(h);
                    auto* mapped=channel.engine.data_.data();
                    return [=](size_t index) {
                        auto& payload=mapped[index];
                        if(!payload.second)return;
                        collide(payload);
                    };
                };
                if(requests)lead.data_.get_queue().submit([&](sycl::handler& h) {
                    auto executors=std::tuple{make_executor(std::get<I>(channels),h)...};
                    h.parallel_for(sycl::range<1>(requests),[=](sycl::id<1> item) {
                        auto execute=[&]<size_t J>() {
                            if(item[0]>=offsets[J] && item[0]<offsets[J+1])
                                std::get<J>(executors)(item[0]-offsets[J]);
                        };
                        (execute.template operator()<I>(),...);
                    });
                }).wait();
                (std::get<I>(channels).buffer.resize(0),...);
            }(std::index_sequence_for<Channel...>{});
        }
    }

    namespace foundation {
        // Build the per-channel body used by the segment execution kernel.
        template<class CollisionType,bool CheckValidity,class ContextType>
        auto pair_collision_body(sycl::handler& h,ContextType& ctx) {
            using incident_tag=incident_species_of<CollisionType>;
            using background_tag=background_species_of<CollisionType>;
            auto acc=ctx.get_access(h);
            sycl::stream warnings(1024,128,h);
            auto& first = ctx.template get<incident_tag>();
            auto& second = ctx.template get<background_tag>();
            auto incident = first.get_particle_access(h);
            auto background = second.get_particle_access(h);
            using first_validator = typename std::remove_reference_t<decltype(first.get_group())>::validator;
            using second_validator = typename std::remove_reference_t<decltype(second.get_group())>::validator;
            auto slot_recorder = [&]<class SpeciesTag>(auto& species) {
                if constexpr(foundation::consumes_species<CollisionType,SpeciesTag>) {
                    auto slots = species.get_group().get_invalid_indexes().get_access(h);
                    return [=](size_t slot) { slots.push_back(slot); };
                } else {
                    return [](size_t) {};
                }
            };
            auto record_first = slot_recorder.template operator()<incident_tag>(first);
            auto record_second = slot_recorder.template operator()<background_tag>(second);
            return [=](auto payload){
                const auto first_slot=foundation::endpoint_slot(incident.data,incident.extent,payload.first);
                const auto second_slot=foundation::endpoint_slot(background.data,background.extent,payload.second);
                if(first_slot==incident.extent || second_slot==background.extent)return;
                foundation::execute_pair<CollisionType,CheckValidity,first_validator,second_validator>(
                    payload,acc,warnings,
                    [&](bool erase_first,bool erase_second) {
                        if(erase_first) record_first(first_slot);
                        if(erase_second) record_second(second_slot);
                    });
            };
        }
    }

    namespace foundation {
        // Bind the time step to the resources used by MCC and DPMCC.
        template<class ContextType>
        struct collision_step_context {
            ContextType& resources;
            double dt;

            struct acc_type : ContextType::acc_type { double dt; };
            template<class ResourceTag> decltype(auto) get(){return resources.template get<ResourceTag>();}
            acc_type get_access(sycl::handler& h){return {resources.get_access(h),dt};}
        };

        template<bool CheckValidity,class ContextType,class... Collision>
        auto bind_pairing_channels(collision_step_context<ContextType>& step,std::tuple<Collision...>*) {
            auto bind=[&]<class Channel>() {
                auto& channel=step.template get<Channel>();
                return bind_pairing_channel<same_species_channel<Channel>::value>(
                    channel.engine,channel.buffer,step.template get<incident_species_of<Channel>>(),
                    [&step](sycl::handler& h) {
                        return pair_collision_body<Channel,CheckValidity>(h,step);
                    });
            };
            return std::tuple{bind.template operator()<Collision>()...};
        }

        template<class BackgroundTag,class ContextType,class... Collision>
        void execute_background_batch(collision_step_context<ContextType>& step,std::tuple<Collision...>*) {
            using channels=decltype(std::tuple_cat(std::declval<std::conditional_t<
                std::is_same_v<BackgroundTag,background_species_of<Collision>>,std::tuple<Collision>,std::tuple<>>>()...));
            auto& bg=step.template get<BackgroundTag>();

            using segments=tag::foundation::partition_tuple<channels,same_species_channel>;
            using same_species_channels=typename segments::matched;
            using cross_species_channels=typename segments::remaining;

            // Earlier batches or the first segment may have consumed queued incident particles.
            deal(bg,bind_pairing_channels<true>(step,static_cast<same_species_channels*>(nullptr)));
            deal(bg,bind_pairing_channels<true>(step,static_cast<cross_species_channels*>(nullptr)));
        }
    }

    // Generate requests once per incident species, execute background batches, then insert all products.
    // Exceptions do not roll back; discard or rebuild resources after overflow.
    template<class ModelType,class ContextType>
    requires (has_collisions<ModelType> && foundation::context<ContextType>)
    void execute_collision_model(ContextType& ctx,double dt){
        if(!std::isfinite(dt) || dt<0) throw std::invalid_argument("invalid DPMCC dt");
        if(dt==0) return;

        using all_channels=typename ModelType::collision_types_tuple;
        using model_types=foundation::dpmcc_model_types<all_channels>;
        using collision_types=typename model_types::collision_types_tuple;
        using incidents=typename foundation::collision_model_types<all_channels>::incident_tuple;
        // Lazy host-only cache, used solely for reservations before collisions start.
        std::array<std::optional<size_t>,std::tuple_size_v<incidents>> initial_counts{};
        auto initial_count=[&]<class IncidentTag>() {
            auto& count=initial_counts[tag::foundation::index_in_tuple<IncidentTag,incidents>::value];
            if(!count)count=ctx.template get<IncidentTag>().get_group().size();
            return *count;
        };

        using products=typename foundation::collision_model_types<all_channels>::product_tuple;
        tag::foundation::for_each_type<products>([&]<class ProductTag> {
            auto& buffer=ctx.template get<ProductTag>().buffer;
            const size_t pending=buffer.size();
            auto required_capacity=[&](auto count) {
                size_t required=pending;
                // A particle selects at most one channel per group; distinct groups accumulate.
                tag::foundation::for_each_type<incidents>([&]<class IncidentTag> {
                    using groups=typename foundation::incident_collision_groups<IncidentTag,all_channels>::type;
                    tag::foundation::for_each_type<groups>([&]<class Channels> {
                        constexpr size_t products_per_event=foundation::max_product_count<ProductTag,Channels>;
                        if constexpr(products_per_event>0)
                            required+=count.template operator()<IncidentTag>()*products_per_event;
                    });
                });
                return required;
            };
            auto extent=[&]<class Tag> { return ctx.template get<Tag>().get_group().extent(); };
            if(required_capacity(extent)>buffer.capacity())
                buffer.reserve(required_capacity(initial_count));
        });

        tag::foundation::for_each_type<collision_types>([&]<class Channel> {
            using incident_tag=foundation::incident_species_of<Channel>;
            auto& buffer=ctx.template get<Channel>().buffer;
            const size_t pending=buffer.size();
            const size_t capacity=buffer.capacity();
            // Extent is a host-only upper bound; count live particles only if it does not fit.
            if(pending+ctx.template get<incident_tag>().get_group().extent()<=capacity)return;
            const size_t required=pending+initial_count.template operator()<incident_tag>();
            if(required>capacity)
                buffer.reserve(std::max(required,capacity+capacity/2));
        });
        foundation::collision_step_context<ContextType> step{ctx,dt};
        execute_incident_collisions<all_channels>(step,dt);
        using backgrounds=typename foundation::collision_model_types<collision_types>::background_tuple;
        tag::foundation::for_each_type<backgrounds>([&]<class BackgroundTag> {
            foundation::execute_background_batch<BackgroundTag>(step,static_cast<collision_types*>(nullptr));
        });
        // Check all product buffers before insertion can reuse slots or move particle storage.
        tag::foundation::for_each_type<products>([&]<class Tag> {
            ctx.template get<Tag>().buffer.if_overflow();
        });
        flush_product_buffers<typename ContextType::tags_tuple>(ctx);
        using first_species_tag=std::tuple_element_t<0,typename model_types::species_tuple>;
        ctx.template get<first_species_tag>().get_group().get_content().get_queue().wait_and_throw();
    }
}
#endif
