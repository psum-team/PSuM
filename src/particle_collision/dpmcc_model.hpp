#ifndef PSUM_PARTICLE_COLLISION_DPMCC_MODEL_HPP
#define PSUM_PARTICLE_COLLISION_DPMCC_MODEL_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <sycl/sycl.hpp>
#include "pairing_engine.hpp"
#include "mcc_model.hpp"

namespace psum {

namespace particle_collision {

    struct base_of_deferred_pairing_collision {};

    namespace foundation {
        template <typename Collision>
        struct deferred_pairing_collision_matched {
            inline static constexpr bool value = std::is_base_of_v<
                base_of_deferred_pairing_collision, std::remove_cvref_t<Collision>
            >;
        };

        template <typename Tuple>
        struct dp_filter;

        template <typename... Ts>
        struct dp_filter<std::tuple<Ts...>> {
            using type = decltype(std::tuple_cat(
                std::declval<
                    std::conditional_t<deferred_pairing_collision_matched<Ts>::value, 
                        std::tuple<Ts>, 
                        std::tuple<>
                    >
                >()...
            ));
        };

        template <typename Tuple>
        using deferred_pairing_filter_tuple = typename dp_filter<Tuple>::type;
    }

    template <typename DPCollision, typename Particle>
    struct dpmcc_context {
        using Position = std::remove_cvref_t<decltype(psum::tag::get<psum::tag::property::position>(std::declval<Particle>()))>;
        using buffer_type = psum::particle_container::device_vector<std::pair<Position, std::pair<Particle*, Particle*>>>;
        using acc_type = buffer_type::acc_type;
        buffer_type buffer;
        dpmcc_context(): buffer(sycl::queue{sycl::default_selector()}) {}
        // this function is only used for buffer initialization with CORRECT sycl::queue!!
        template <typename ParticleContainer>
        void bind(ParticleContainer& particles){
            buffer = buffer_type(particles.get_content().get_queue(), std::max((size_t)8192, (size_t)(particles.size() * 0.2)));
        }
        buffer_type::acc_type get_access(sycl::handler& h) {
            return buffer.get_access(h);
        }
        template <typename Context>
        void clean_buffer(Context& ctx) {
            if (buffer.size() > buffer.capacity() / 2)
                buffer.reserve(buffer.capacity() * 2);
            buffer.for_each([&](sycl::handler &h) {
                auto ctx_acc = ctx.get_access(h);
                return [=](auto& query) {
                    auto ptr1 = query.second.first;
                    auto ptr2 = query.second.second;
                    if (ptr1 == nullptr || ptr2 == nullptr) return;
                    DPCollision::pair_collide(*ptr1, *ptr2, ctx_acc);
                    ptr1 = nullptr;
                    ptr2 = nullptr;
                };
            });
            buffer.resize(0);
        }
    };

    template <typename Tuple, typename Particle>
    struct dpmcc_model_context; 

    template <typename... Ts, typename Particle>
    struct dpmcc_model_context<std::tuple<Ts...>, Particle>: public std::tuple<dpmcc_context<Ts, Particle>...> {
        using base_type = std::tuple<dpmcc_context<Ts, Particle>...>;
        using acc_tuple = std::tuple<typename dpmcc_context<Ts, Particle>::acc_type...>;

        struct acc_type : public acc_tuple{
            template<typename... Args>
            acc_type(Args&&... args) 
                : acc_tuple(std::forward<Args>(args)...) {}

            template <typename Col>
            static constexpr int index = psum::tag::foundation::index_in_tuple<Col, std::tuple<Ts...>>::value;

            template <typename Col>
            decltype(auto) get() const {
                static_assert(index<Col> != -1, "Collision not found in dpmcc_module_context_extension.");
                return std::get<index<Col>>(*this);
            }
        };

        template <typename Col>
        static constexpr int index = psum::tag::foundation::index_in_tuple<Col, std::tuple<Ts...>>::value;

        template <typename Col>
        dpmcc_context<Col, Particle>& get() {
            static_assert(index<Col> != -1, "Collision not found in dpmcc_module_context_extension.");
            return std::get<index<Col>>(*this);
        }

        acc_type get_access(sycl::handler& h) {
            return acc_type(
                get<Ts>().get_access(h)...
            );
        }
    };

    template <typename AllCollision, typename ParticleGroup>
    using standard_dpmccm_context = dpmcc_model_context<
            foundation::deferred_pairing_filter_tuple<typename AllCollision::col_struct_tuple>,
            typename ParticleGroup::value_type
        >;

    template <typename Col, typename ContextAcc, typename Particle>
    void insert_deferred_pairing_in_context(ContextAcc& ctx_acc, Particle& p) {
        ctx_acc.template get<Col>().push_back({psum::tag::get<psum::tag::property::position>(p), {&p, nullptr}});
    }

    template <typename ContextA, typename ContextB>
    struct combined_context {
        ContextA a_ctx_;
        ContextB b_ctx_;

        template <typename Tag>
        decltype(auto) get() {
            if constexpr (ContextA::template exist<Tag>()) {
                return a_ctx_.template get<Tag>();
            } 
            else {
                return b_ctx_.template get<Tag>();
            }
        }

        struct acc_type {
            ContextA::acc_type a_ctx_acc_;
            ContextB::acc_type b_ctx_acc_;

            template <typename Tag>
            decltype(auto) get() const {
                if constexpr (ContextA::template exist<Tag>()) {
                    return a_ctx_acc_.template get<Tag>();
                } 
                else {
                    return b_ctx_acc_.template get<Tag>();
                }
            }
        };

        acc_type get_access(sycl::handler& h) {
            return acc_type{
                a_ctx_.get_access(h),
                b_ctx_.get_access(h)
            };
        }
    };

    template <typename AllCollisionTuple, typename Context, typename PairingEngine>
    requires (has_sub_models<AllCollisionTuple>)
    void execute_dpmcc_model(Context& ctx, double dt, PairingEngine& pe) {
        execute_mcc_model<AllCollisionTuple>(ctx, dt);
        typedef foundation::deferred_pairing_filter_tuple<typename AllCollisionTuple::col_struct_tuple> DPCollisionTuple;
        std::apply([&](auto&&... args) {
            (execute_single_dp_collision<typename std::decay_t<decltype(args)>>(ctx, dt, pe),...);
        }, DPCollisionTuple{});
    }

    template <typename ColTag, typename Context, typename PairingEngine>
    void execute_single_dp_collision(Context& ctx, double dt, PairingEngine& pe) {
        auto& group = ctx.template get<typename ColTag::background_species>().get_group();
        auto& buffer = ctx.template get<ColTag>().buffer;
        pe.deal(group, buffer);
        ctx.template get<ColTag>().clean_buffer(ctx);
    }

}

}

#endif