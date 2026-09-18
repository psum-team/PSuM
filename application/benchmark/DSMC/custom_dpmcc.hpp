#ifndef CUSTOM_MCCM_HPP
#define CUSTOM_MCCM_HPP

#include <psum/psum.hpp>

inline double S_Variable_Hard_Sphere(double v2){
	double v = sqrt(v2);
	double boltz = 1.3805e-23;
	double pmass = 4.652E-26;
	double diaref = 4.17e-10;
	double T_ref = 273;
	double omega_N2 = 0.74;
	return acos(-1) * diaref * diaref * pow(2.0 * boltz * T_ref / (pmass * v * v), omega_N2 - 0.5) / std::tgamma(2.5 - omega_N2);
}

inline double S_Variable_Hard_Sphere_4X(double v2){
	return S_Variable_Hard_Sphere(v2) * 4.0;
}

struct N2_dsmc_model {
    using N2_tag = psum::particle_collision::species_tags::Nitrogen;
    using species_in_model = std::tuple<N2_tag>;

    struct col_N2_to_N2_vhs : psum::particle_collision::base_of_deferred_pairing_collision {
        inline static double collision_cross_section(double in) {return S_Variable_Hard_Sphere_4X(in);}
        inline static void collide(auto& p, const auto& ctx) {
            psum::particle_collision::insert_deferred_pairing_in_context<col_N2_to_N2_vhs>(ctx, p);
        }
        using background_species = N2_tag;
        inline static void pair_collide(auto& p1, auto& p2, const auto& ctx) {
            if (&p1 == &p2) return;
            auto* first = &p1 < &p2? &p1 : &p2;
            auto* second = &p1 > &p2 ? &p1 : &p2;
            if (psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(*first)).try_lock()) {
                if (psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(*second)).try_lock()) {
                    auto R = psum::random::view_as_rander(get<psum::random_seed>(p1));
                    if (R() < 0.5) {
                        auto v1 = get<psum::velocity>(p1);
                        auto v2 = get<psum::velocity>(p2);
                        double S_pre = collision_cross_section(v1.squaredNorm());
                        double pre_freq = S_pre * v1.norm();
                        auto rel_v = v1 - v2;
                        double v_relative_length = rel_v.norm();
                        double true_freq = S_Variable_Hard_Sphere(rel_v.squaredNorm()) * v_relative_length;
                        if (R() < true_freq/pre_freq)
                        {
                            if(v_relative_length != 0)
                            {
                                auto v_mean = (v1 + v2) * 0.5;
                                double preCollisionEi1 = get<psum::internal_energy>(p1);
                                double preCollisionEi2 = get<psum::internal_energy>(p2);
                                int iDof = 2; //degree of freedom of N2: 2 rotation, and vibrational is neglected
                                double omega_N2 = 0.74;
                                double pmass = 4.652E-26;
                                double ChiB = 2.5 - omega_N2;

                                double availableEnergy = 0.5 * pmass * v_relative_length * v_relative_length;
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
                                double v_r_new = sqrt(2.0 * availableEnergy / pmass);

                                auto v_relative_direction = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
                                get<psum::velocity>(p1) = v_mean - v_relative_direction * (v_r_new / 2);
                                get<psum::velocity>(p2) = v_mean + v_relative_direction * (v_r_new / 2);
                            }
                        }
                    }
                    psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(p1)).unlock();
                }
                psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(p2)).unlock();
            }
        }
    };

    template <typename ParticleGroup, typename Field>
    static auto make(ParticleGroup& N2, Field& f_N2, const psum::field::grid3D& pairing_grid) {
        using namespace psum::particle_collision;
        using namespace psum::particle_collision::foundation;

        using collision_N2_to_N2_vhs = collision_processor<
            density_interp_nearest_3d<N2_tag>,
            std::tuple<
                col_N2_to_N2_vhs
            >
        >;

        using all_collision = mcc_model<
            mcc_submodel_for_incident<N2_tag, collision_N2_to_N2_vhs>
        >;

        using mccm_context = psum::particle_collision::standard_mccm_context<species_in_model, ParticleGroup, Field>;
        using dpmcc_context = psum::particle_collision::standard_dpmccm_context<all_collision, ParticleGroup>;
        using total_context = combined_context<mccm_context, dpmcc_context>;

        using pengine = pairing_engine<ParticleGroup, psum::field::grid3D>;
        
        std::shared_ptr<total_context> context = std::make_shared<total_context>();
        std::shared_ptr<pengine> p_engine = std::make_shared<pengine>(N2.get_content().get_queue(), pairing_grid);
        (*context).template get<N2_tag>().bind(N2, f_N2); // bind N2 atoms to context
        (*context).template get<col_N2_to_N2_vhs>().bind(N2);

        return [context, p_engine](double dt) {
            psum::particle_collision::execute_dpmcc_model<all_collision>((*context), dt, *p_engine);
            psum::particle_collision::execute_all_clean_buffers<species_in_model>((*context));
        };
    }
};

#endif