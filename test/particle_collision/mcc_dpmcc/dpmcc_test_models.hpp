#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>
#include <string_view>
#include <vector>
#include <Eigen/Dense>
#include "../../../include/psum/tag.hpp"
#include "../../../include/psum/field.hpp"
#include "../../../include/psum/particle_container.hpp"
#include "../../../src/particle_collision/foundation.hpp"
#include "../../../src/particle_collision/species_tags.hpp"
#include "../../../src/particle_collision/collision_model.hpp"

using namespace psum;
using namespace tag;
using namespace tag::property;
using namespace psum::particle_collision;

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_container::particle_group<particle, particle_container::pos_x_nan_is_invalid>;
using Field = field::node_field2D<double>;


namespace psum::particle_collision {
    template<>struct position_projection<species_tags::electron> {
        template<class P>static auto project(const P& p){return std::array<double,2>{p[0],p[1]};}
    };
    template<>struct position_projection<species_tags::Xenon> {
        template<class P>static auto project(const P& p){return std::array<double,2>{p[0],p[1]};}
    };
    template<>struct position_projection<species_tags::Xenon_pos_1> : position_projection<species_tags::Xenon> {};
}

struct electron_xenon_tags {
    using incident_species_tag = species_tags::electron;
    using background_species_tag = species_tags::Xenon;
};
struct electron_xenon_channel : electron_xenon_tags {
    static double collision_cross_section(double) { return 1; }
};

// Synthetic Maxwell-molecule cross section: sigma*g is a fixed test rate.
double unit_rate_cross_section(const auto& pair,double rate=1) {
    const double g=(get<velocity>(*pair.first)-get<velocity>(*pair.second)).norm();
    return g>0?rate/g:0;
}
struct electron_xenon_pair : electron_xenon_tags,deferred_pairing_collision_tag {
    static double request_probability(const auto& p,const auto&) {
        return get<velocity>(p).squaredNorm()>0?1.0:0.0;
    }
};
struct pair_reaction : electron_xenon_pair,reaction_collision_tag {};

struct consume_pair : pair_reaction {
    using consumed_species_tags = std::tuple<incident_species_tag,background_species_tag>;
    using product_species_tags = std::tuple<species_tags::Xenon_pos_1>;
    static double pair_collision_cross_section(const auto& pair,const auto& ctx) {
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(pair.first); });
        static_assert(!requires { ctx.template insert_particle<species_tags::Xenon_pos_1>(particle{}); });
        return unit_rate_cross_section(pair);
    }
    static void pair_collide(auto& pair,const auto& ctx) {
        static_assert(!requires { ctx.template get<incident_species_tag>().group_acc; });
        const auto incident = ctx.template get<incident_species_tag>();
        const auto background = ctx.template get<background_species_tag>();
        static_assert(std::is_same_v<decltype(ctx.dt),double>);
        static_assert(!requires { incident.field_acc.getGrid(); });
        static_assert(requires { background.field_acc.getGrid(); });
        static_assert(!requires { ctx.template get<species_tags::Xenon_pos_1>().field_acc.getGrid(); });
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(); });
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(&get<velocity>(*pair.first)); });
        static_assert(!requires { ctx.template erase_particle<species_tags::Xenon_pos_1>(pair.second); });
        static_assert(!requires { ctx.template insert_particle<incident_species_tag>(*pair.first); });
        static_assert(!requires { ctx.template insert_particle<background_species_tag>(*pair.second); });
        ctx.template erase_particle<incident_species_tag>(pair.first);
        ctx.template erase_particle<incident_species_tag>(pair.first);
        ctx.template erase_particle<background_species_tag>(pair.second);
        // A requested endpoint remains readable until the callback returns.
        ctx.template insert_particle<species_tags::Xenon_pos_1>(*pair.first);
    }
};
struct produce_ion : pair_reaction {
    using consumed_species_tags = std::tuple<>;
    using product_species_tags = std::tuple<species_tags::Xenon_pos_1>;

    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }

    static void pair_collide(const auto& pair,const auto& ctx) {
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(pair.first); });
        static_assert(!requires { ctx.template erase_particle<background_species_tag>(pair.second); });
        ctx.template insert_particle<species_tags::Xenon_pos_1>(*pair.first);
    }
};
struct observe_mcc : electron_xenon_channel {
    static double collision_cross_section(double v2) { return v2 < 4 ? 1 : 0; }

    static void collide(particle& p,const auto& ctx) {
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(); });
        static_assert(!requires { ctx.template insert_particle<species_tags::Xenon_pos_1>(p); });
        get<velocity>(p).y() = 0.5;
    }
};
struct observe_pair : electron_xenon_pair {
    using background_species_tag = species_tags::Xenon_pos_1;

    static double pair_collision_cross_section(const auto& pair,const auto&) { return unit_rate_cross_section(pair); }

    static void pair_collide(auto& pair,const auto& ctx) {
        static_assert(!requires { ctx.template erase_particle<incident_species_tag>(pair.first); });
        static_assert(!requires { ctx.template insert_particle<background_species_tag>(*pair.first); });
        get<velocity>(*pair.first).z() = 0.5;
    }
};
template<class SpeciesTag>
struct same_species_pair : deferred_pairing_collision_tag {
    using incident_species_tag = SpeciesTag;
    using background_species_tag = SpeciesTag;
    static double request_probability(const auto&,const auto&) { return 0; }
    static double pair_collision_cross_section(const auto&,const auto&) { return 0; }
    static void pair_collide(auto&,const auto&) {}
};
