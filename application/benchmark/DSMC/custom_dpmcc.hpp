#ifndef CUSTOM_MCCM_HPP
#define CUSTOM_MCCM_HPP

#include <psum/psum.hpp>
#include "n2_vhs.hpp"
#include <iostream>
#include <memory>

struct N2_dsmc_model {
    using N2_tag = psum::particle_collision::species_tags::Nitrogen;
    using species_in_model = std::tuple<N2_tag>;

    struct col_N2_to_N2_vhs : psum::particle_collision::deferred_pairing_collision_tag {
        static double request_probability(const auto&,const auto&) {
            return 0.1;
        }
        using incident_species_tag = N2_tag;
        using background_species_tag = N2_tag;
        static double pair_collision_cross_section(const auto& pair,const auto&) {
            const auto relative=get<psum::velocity>(*pair.first)-get<psum::velocity>(*pair.second);
            return n2::vhs_cross_section(relative.squaredNorm());
        }

        static void pair_collide(auto& pair,const auto&) {
            auto& p1=*pair.first;auto& p2=*pair.second;
            auto R=psum::random::view_as_rander(get<psum::random_seed>(p1));
            auto v1=get<psum::velocity>(p1);auto v2=get<psum::velocity>(p2);
            auto rel_v=v1-v2;double v_relative_length=rel_v.norm();
            if(v_relative_length != 0)
            {
                auto v_mean = (v1 + v2) * 0.5;
                double preCollisionEi1 = get<psum::internal_energy>(p1);
                double preCollisionEi2 = get<psum::internal_energy>(p2);
                int iDof = 2; //degree of freedom of N2: 2 rotation, and vibrational is neglected
                double omega_N2 = 0.74;
                double pmass = 4.652E-26;
                double ChiB = 2.5 - omega_N2;

                double availableEnergy = 0.25 * pmass * v_relative_length * v_relative_length;
                double inverseCollisionNumber = 1.0;

                // p1
                if (inverseCollisionNumber > R())
                {
                    availableEnergy += preCollisionEi1;
                    double e_ratio = 1.0 - pow(R(), 1.0 / ChiB);
                    get<psum::internal_energy>(p1) = e_ratio*availableEnergy;
                    availableEnergy -= get<psum::internal_energy>(p1);
                }
                // p2
                if (inverseCollisionNumber > R())
                {
                    availableEnergy += preCollisionEi2;
                    double e_ratio = 1.0 - pow(R(), 1.0 / ChiB);
                    get<psum::internal_energy>(p2) = e_ratio*availableEnergy;
                    availableEnergy -= get<psum::internal_energy>(p2);
                }

                // Rescale the translational energy
                double v_r_new = sqrt(4.0 * availableEnergy / pmass);

                auto v_relative_direction = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
                get<psum::velocity>(p1) = v_mean - v_relative_direction * (v_r_new / 2);
                get<psum::velocity>(p2) = v_mean + v_relative_direction * (v_r_new / 2);
            }
        }
    };

    template <typename ParticleGroup, typename Field>
    static auto make(ParticleGroup& N2, Field& f_N2) {
        using namespace psum::particle_collision;
        using namespace psum::particle_collision::foundation;

        using all_collision = collision_model<col_N2_to_N2_vhs>;

        using species_context_type=species_contexts<all_collision,ParticleGroup,Field>;
        using collision_context=collision_channel_contexts<all_collision,ParticleGroup,species_context_type>;
        using total_context=context_composition<species_context_type,collision_context>;
        struct binding{ParticleGroup* group;Field* density;};
        auto q=N2.get_content().get_queue();
        auto species=std::make_shared<species_context_type>(binding{&N2,&f_N2});
        auto collisions=std::make_shared<collision_context>(q,*species);
        return [species,collisions](double dt) {
            total_context context{*species,*collisions};
            psum::particle_collision::execute_collision_model<all_collision>(context,dt);
        };
    }
};

#endif
