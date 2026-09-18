#ifndef PSUM_PARTICLE_COLLISION_MCC_MODEL_HPP
#define PSUM_PARTICLE_COLLISION_MCC_MODEL_HPP

#include <concepts>
#include <tuple>
#include <type_traits>
#include <cmath>
#include <sycl/sycl.hpp>
#include "foundation.hpp"
#include "../../include/psum/particle_container.hpp"

namespace psum {

namespace particle_collision {

    template<typename... SubModels>
    struct mcc_model {
        using sub_models_tuple = std::tuple<SubModels...>;
        using col_struct_tuple = decltype(std::tuple_cat(std::declval<typename SubModels::col_struct_tuple>()...));
    };

    template <typename T>
    concept has_sub_models = requires {
        typename std::remove_cvref_t<T>::sub_models_tuple;
    };

    template <typename Incident, typename Collisions>
    struct mcc_submodel_for_incident: public std::pair<Incident, Collisions> {
        using col_struct_tuple = Collisions::col_struct_tuple;
    };

    // combination of related data for a particle species in Monte Carlo Collision
    template <typename ParticleContainer, typename Field>
    struct mcc_species_context {
        std::function<typename Field::acc_type(sycl::handler&)> field_acc_getter;
        std::function<ParticleContainer&()> group_ref_getter;
        using buffer_type = psum::particle_container::device_vector<typename ParticleContainer::value_type>;
        buffer_type buffer;
        mcc_species_context(): buffer(sycl::queue{sycl::default_selector()}) {}
        void bind(ParticleContainer& particles, Field& field){
            field_acc_getter = [&field](sycl::handler& h) { return field.get_access(h); };
            group_ref_getter = [&particles]()->ParticleContainer& { return particles; };
            buffer = buffer_type(particles.get_content().get_queue(), std::max((size_t)8192, (size_t)(particles.size() * 0.2)));
        }
        struct acc_type {
            Field::acc_type field_acc;
            buffer_type::acc_type buffer_acc;
        };
        acc_type get_access(sycl::handler& h) {
            return {field_acc_getter(h), buffer.get_access(h)};
        }
        auto& get_group() {
            return group_ref_getter();
        }
        void clean_buffer() {
            if (buffer.size() > buffer.capacity() / 2)
                buffer.reserve(buffer.capacity() * 2);
            get_group().insert(buffer);
            buffer.clear();
        }
    };

    // mcc_module_context< tag_bind<Tag_of_Species, species_context>, tag_bind<Tag_of_Collisions, collision_context>,... >
    // combination of all related species and collision contexts
    template <typename... TagsAndTypes>
    struct mcc_module_context : public tag::tagged_struct<TagsAndTypes...> {
        using base_type = tag::tagged_struct<TagsAndTypes...>;

        using accessor_tuple = std::tuple<typename TagsAndTypes::back::acc_type...>;

        struct acc_type : public accessor_tuple {
            template<typename... Args>
            acc_type(Args&&... args) 
                : accessor_tuple(std::forward<Args>(args)...) {}

            template <typename Tag>
            static constexpr int index = psum::tag::foundation::index_in_tuple<Tag, typename base_type::tuple_of_tag>::value;

            template <typename Tag>
            decltype(auto) get() const {
                static_assert(index<Tag> != -1, "Tag not found in mcc_module_context.");
                return std::get<index<Tag>>(*this);
            }
        };

        using base_type::base_type;

        acc_type get_access(sycl::handler& h) {
            return acc_type(
                psum::tag::get<typename TagsAndTypes::front>(*this).get_access(h)...
            );
        }

        template <typename Tag>
        static constexpr bool exist() {
            return psum::tag::foundation::index_in_tuple<Tag, typename base_type::tuple_of_tag>::value != -1;
        }

        template <typename Tag>
        decltype(auto) get() {
            return psum::tag::get<Tag>(*static_cast<base_type*>(this));
        }
    };

    template <typename SpeciesTuple, typename ParticleGroup, typename Field>
    struct standard_mccm_context_builder;

    template <typename... Tags, typename ParticleGroup, typename Field>
    struct standard_mccm_context_builder<std::tuple<Tags...>, ParticleGroup, Field> {
        using species_ctx = psum::particle_collision::mcc_species_context<ParticleGroup, Field>;
        using type = psum::particle_collision::mcc_module_context<
            psum::tag::tag_bind<Tags, species_ctx>...
        >;
    };

    template <typename SpeciesTuple, typename ParticleGroup, typename Field>
    using standard_mccm_context = typename standard_mccm_context_builder<SpeciesTuple, ParticleGroup, Field>::type;

    template <typename Tag>
    struct density_interp_1d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp({pos.x()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_2d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp({pos.x(), pos.y()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_2d_zr {
        static double compute(const auto& pos, const auto& ctx_acc) {
            double r = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
            return psum::field::interp({pos.z(), r}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_3d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp({pos.x(), pos.y(), pos.z()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_nearest_1d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp_nearest({pos.x()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_nearest_2d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp_nearest({pos.x(), pos.y()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_nearest_2d_zr {
        static double compute(const auto& pos, const auto& ctx_acc) {
            double r = std::sqrt(pos.x() * pos.x() + pos.y() * pos.y());
            return psum::field::interp_nearest({pos.z(), r}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag>
    struct density_interp_nearest_3d {
        static double compute(const auto& pos, const auto& ctx_acc) {
            return psum::field::interp_nearest({pos.x(), pos.y(), pos.z()}, ctx_acc.template get<Tag>().field_acc);
        }
    };

    template <typename Tag, typename ContextAcc, typename Particle>
    void insert_particle_in_context(ContextAcc& ctx_acc, const Particle& p) {
        ctx_acc.template get<Tag>().buffer_acc.push_back(p);
    }

    template <typename AllCollisionTuple, typename Context>
    requires (has_sub_models<AllCollisionTuple>)
    void execute_mcc_model(Context& ctx, double dt) {
        std::apply([&](auto&&... pair_item) {
            (execute_single_collision<typename std::decay_t<decltype(pair_item)>::first_type, 
                                typename std::decay_t<decltype(pair_item)>::second_type>(ctx, dt), ...);
        }, typename AllCollisionTuple::sub_models_tuple{});
    }

    template <typename Tag, typename Executor, typename Context>
    void execute_single_collision(Context& ctx, double dt) {
        auto& group = ctx.template get<Tag>().get_group();
        using Particle = typename std::remove_cvref_t<decltype(group)>::value_type;
        
        group.for_each([&](sycl::handler& h) {
            auto ctx_acc = ctx.get_access(h);
            return [=](Particle& p) {
                Executor::execute(p, dt, ctx_acc);
            };
        });
    }

    template <typename SpeciesTuple, typename Context>
    void execute_all_clean_buffers(Context& ctx) {
        std::apply([&](auto&&... args) {
            auto one_clean_buffer = [&](auto v_tag) {
                typedef std::remove_cvref_t<decltype(v_tag)> tag_type;
                ctx.template get<tag_type>().clean_buffer();
            };
            (one_clean_buffer(args),...);
        }, SpeciesTuple{});
    }

}

}

#endif