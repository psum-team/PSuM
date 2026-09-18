#ifndef CUSTOM_MCCM_HPP
#define CUSTOM_MCCM_HPP

#include <psum/psum.hpp>
#include "HeCCS.h"

inline double He_in_isotropic_halfv2(double v2)
{
    return He_in_isotropic(v2/2);
}
inline double He_in_backward_halfv2(double v2)
{
    return He_in_backward(v2/2);
}

struct He_mcc_model {
    using electron_tag = psum::particle_collision::species_tags::electron;
    using He_atom_tag = psum::particle_collision::species_tags::Helium;
    using He_ion1_tag = psum::particle_collision::species_tags::Helium_pos_1;
    using species_in_model = std::tuple<electron_tag, He_atom_tag, He_ion1_tag>;

    static constexpr double unit_charge = 1.602e-19;
    static constexpr double ele_mass = 9.10956E-31;
    static constexpr double He_mass = 6.67E-27;
    static constexpr double atom_temperature_K = 300;

    struct col_ele_to_atom_elastic {
        inline static double collision_cross_section(double in) {return He_elastic(in);}
        inline static void collide(auto& p, const auto&) {
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            auto randv = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
            get<psum::velocity>(p) = (randv * get<psum::velocity>(p).norm());
        }
    };

    struct col_ele_to_atom_excitation_triplet {
        inline static double collision_cross_section(double in) {return He_triplet(in);}
        inline static void collide(auto& p, const auto&) {
            double threshold_exc1_ev = 19.82;
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            auto randv = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
            double v2_new = get<psum::velocity>(p).squaredNorm() - 2 * unit_charge * threshold_exc1_ev / ele_mass;
            if (v2_new > 0) get<psum::velocity>(p) = randv * sqrt(v2_new);
        }
    };

    struct col_ele_to_atom_excitation_singlet {
        inline static double collision_cross_section(double in) {return He_singlet(in);}
        inline static void collide(auto& p, const auto&) {
            double threshold_exc2_ev = 20.61;
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            auto randv = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
            double v2_new = get<psum::velocity>(p).squaredNorm() - 2 * unit_charge * threshold_exc2_ev / ele_mass;
            if (v2_new > 0) get<psum::velocity>(p) = randv * sqrt(v2_new);
        }
    };

    struct col_ele_to_atom_ionization {
        inline static double collision_cross_section(double in) {return He_ioni(in);}
        inline static void collide(auto& p, const auto& ctx_acc) {
            double threshold_ion_ev = 24.587;
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            double energy = get<psum::velocity>(p).squaredNorm() * ele_mass * 0.5;
            double energy_second = (energy - threshold_ion_ev * unit_charge) / 2;
            double energy_new = energy_second;
            if (energy_new < 0) return;

            auto randv = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
            get<psum::velocity>(p) = randv * sqrt(2 * energy_new / ele_mass);

            auto new_ele = p;
            get<psum::velocity>(new_ele) = psum::random::RandFunction3D::RandV_spherical(R.get_rander()) * sqrt(2 * energy_second / ele_mass);
            get<psum::random_seed>(new_ele) = R() * RAND_MAX;
            psum::particle_collision::insert_particle_in_context<electron_tag>(ctx_acc, new_ele);

            auto new_ion = p;
            get<psum::velocity>(new_ion) = psum::random::RandFunction3D::RandV_Maxwell(R.get_rander(), atom_temperature_K, He_mass);
            get<psum::random_seed>(new_ion) = R() * RAND_MAX;
            psum::particle_collision::insert_particle_in_context<He_ion1_tag>(ctx_acc, new_ion);
        }
    };

    struct col_ion_to_atom_isotropic {
        inline static double collision_cross_section(double in) {return He_in_isotropic_halfv2(in);}
        inline static void collide(auto& p, const auto&) {
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            auto vel = get<psum::velocity>(p);
            auto vel_atom = psum::random::RandFunction3D::RandV_Maxwell(R.get_rander(), atom_temperature_K, He_mass);
            auto vel_mean = (vel + vel_atom) * 0.5;
            double rel_v = (vel - vel_atom).norm();
            get<psum::velocity>(p) = vel_mean + rel_v * 0.5 * psum::random::RandFunction3D::RandV_spherical(R.get_rander());
        }
    };

    struct col_ion_to_atom_backward {
        inline static double collision_cross_section(double in) {return He_in_backward_halfv2(in);}
        inline static void collide(auto& p, const auto&) {
            auto R = psum::random::view_as_rander(get<psum::random_seed>(p));
            get<psum::velocity>(p) = psum::random::RandFunction3D::RandV_Maxwell(R.get_rander(), atom_temperature_K, He_mass);
        }
    };

    template <typename ParticleGroup, typename Field>
    static auto make(ParticleGroup& _atom, ParticleGroup& _ele, ParticleGroup& _ion, Field& f_atom, Field& f_elec, Field& f_ion1) {
        using namespace psum::particle_collision;
        using namespace psum::particle_collision::foundation;

        using collision_ele_to_atom = collision_processor<
            density_interp_1d<He_atom_tag>,
            std::tuple<
                col_ele_to_atom_elastic, 
                col_ele_to_atom_excitation_triplet, 
                col_ele_to_atom_excitation_singlet, 
                col_ele_to_atom_ionization
            >
        >;

        using collision_ion_to_atom = collision_processor<
            density_interp_1d<He_atom_tag>,
            std::tuple<
                col_ion_to_atom_isotropic,
                col_ion_to_atom_backward
            >
        >;

        using all_collision = mcc_model<
            mcc_submodel_for_incident<electron_tag, collision_ele_to_atom>,
            mcc_submodel_for_incident<He_ion1_tag, collision_ion_to_atom>
        >;

        using mccm_context = psum::particle_collision::standard_mccm_context<species_in_model, ParticleGroup, Field>;
        std::shared_ptr<mccm_context> mccm_ctx = std::make_shared<mccm_context>();
        (*mccm_ctx).template get<electron_tag>().bind(_ele, f_elec); // bind electrons to context
        (*mccm_ctx).template get<He_atom_tag>().bind(_atom, f_atom); // bind He atoms to context
        (*mccm_ctx).template get<He_ion1_tag>().bind(_ion, f_ion1); // bind He ions to context

        return [mccm_ctx](double dt) {
            psum::particle_collision::execute_mcc_model<all_collision>((*mccm_ctx), dt);
            psum::particle_collision::execute_all_clean_buffers<species_in_model>((*mccm_ctx));
        };
    }
};

#endif