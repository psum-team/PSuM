#include <iostream>
#include <fstream>
#include <psum/psum.hpp>

using namespace psum::prelude;
using property::position;
using property::velocity;
using property::internal_energy;
using property::random_seed;

struct is_hot : psum::tag::foundation::abstract_tag {
    inline const static std::string tag_name = "is_hot";
};

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<internal_energy, double>,
    tag_bind<lock_var, int>,
    tag_bind<random_seed, uint32_t>,
    tag_bind<is_hot, bool>
>;

using ParticleGroup = particle_container::particle_group<Particle, particle_container::pos_x_nan_is_invalid>;

using Field = field::node_field3D<double>;
using VelDistField = field::node_field1D<double>;

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
            if (&p1 >= &p2) return;
            if (psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(p2)).try_lock()) {
                if (psum::utils_sycl::view_as_spin_lock(get<psum::lock_var>(p1)).try_lock()) {
                    auto v1 = get<psum::velocity>(p1);
                    auto v2 = get<psum::velocity>(p2);
                    double S_pre = collision_cross_section(v1.squaredNorm());
                    double pre_freq = S_pre * v1.norm();
                    auto rel_v = v1 - v2;
                    double v_relative_length = rel_v.norm();
                    double true_freq = S_Variable_Hard_Sphere(rel_v.squaredNorm()) * v_relative_length;
                    auto R = psum::random::view_as_rander(get<psum::random_seed>(p1));
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

                            double availableEnergy =
                                0.25 * pmass * v_relative_length * v_relative_length
                            + preCollisionEi1
                            + preCollisionEi2;

                            double R1 = R();
                            double E_trans = availableEnergy * pow(R1, 1.0 / ChiB);

                            double E_internal = availableEnergy - E_trans;

                            double R2 = R();
                            get<psum::internal_energy>(p1) = E_internal * R2;
                            get<psum::internal_energy>(p2) = E_internal * (1.0 - R2);
                            double v_r_new = sqrt(4.0 * E_trans / pmass);

                            auto v_relative_direction = psum::random::RandFunction3D::RandV_spherical(R.get_rander());
                            get<psum::velocity>(p1) = v_mean - v_relative_direction * (v_r_new / 2);
                            get<psum::velocity>(p2) = v_mean + v_relative_direction * (v_r_new / 2);
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
            density_interp_3d<N2_tag>,
            std::tuple<
                col_N2_to_N2_vhs
            >
        >;

        using all_collision = mcc_model<
            mcc_submodel_for_incident<N2_tag, collision_N2_to_N2_vhs>
        >;

        using mccm_context = psum::particle_collision::standard_mccm_context<species_in_model, ParticleGroup, Field>;

        using dpmcc_context = dpmcc_model_context<
            deferred_pairing_filter_tuple<all_collision::col_struct_tuple>,
            typename ParticleGroup::value_type
        >;

        using pengine = pairing_engine<
            ParticleGroup,
            psum::field::grid3D
        >;
        
        using total_context = combined_context<mccm_context, dpmcc_context>;
        std::shared_ptr<total_context> context = std::make_shared<total_context>();
        std::shared_ptr<pengine> p_engine = std::make_shared<pengine>(N2.get_content().get_queue(), pairing_grid);
        (*context).template get<N2_tag>().bind(N2, f_N2); // bind N2 atoms to context
        (*context).template get<col_N2_to_N2_vhs>().bind(N2);

        return [context, p_engine](double dt) {
            psum::particle_collision::execute_mcc_model<all_collision>((*context), dt);
            (*p_engine).deal((*context).template get<col_N2_to_N2_vhs::background_species>().get_group(), (*context).template get<col_N2_to_N2_vhs>().buffer);
            (*context).template get<col_N2_to_N2_vhs>().clean_buffer(*context);
            psum::particle_collision::execute_all_clean_buffers<species_in_model>((*context));
        };
    }
};

void compute_velocity_distribution(ParticleGroup& N2, VelDistField& vx_dist, double v_min, double v_max) {
    vx_dist.setZero();
    
    N2.for_each([&](sycl::handler& h) {
        auto dist_acc = vx_dist.get_access(h);
        double dv = (v_max - v_min) / vx_dist.getGrid().numCells<0>();
        
        return [=](Particle& p) {
            double vx = get<velocity>(p).x();
            if (dist_acc.getGrid().inGrid({vx}))
                add_back({vx}, 1, dist_acc);
        };
    });
}

void compute_temperature(ParticleGroup& N2, sycl::queue& q, 
                         double* T_hot, double* n_hot,
                         double* T_cold, double* n_cold,
                         double mass) {
    *T_hot = 0; *n_hot = 0;
    *T_cold = 0; *n_cold = 0;
    
    double k_B = 1.380649e-23;
    
    N2.for_each([&](sycl::handler& h) {
        return [=](Particle& p) {
            auto v = get<velocity>(p);
            double v_sq = v.squaredNorm();
            bool hot = get<is_hot>(p);
            
            if (hot) {
                atomic_add(*T_hot, v_sq);
                atomic_add(*n_hot, 1.0);
            } else {
                atomic_add(*T_cold, v_sq);
                atomic_add(*n_cold, 1.0);
            }
        };
    });
    
    if (*n_hot > 0) *T_hot = mass * (*T_hot) / (3.0 * (*n_hot) * k_B);
    if (*n_cold > 0) *T_cold = mass * (*T_cold) / (3.0 * (*n_cold) * k_B);
}

void compute_momentum_energy(ParticleGroup& N2, sycl::queue& q, 
                             double* px, double* py, double* pz, 
                             double* energy_kinetic, double* energy_internal, double mass) {
    *px = 0; *py = 0; *pz = 0;
    *energy_kinetic = 0;
    *energy_internal = 0;
    
    N2.for_each([&](sycl::handler& h) {
        return [=](Particle& p) {
            auto v = get<velocity>(p);
            atomic_add(*px, v.x() * mass);
            atomic_add(*py, v.y() * mass);
            atomic_add(*pz, v.z() * mass);
            atomic_add(*energy_kinetic, 0.5 * v.squaredNorm() * mass);
            atomic_add(*energy_internal, get<internal_energy>(p));
        };
    });
}

int main() {
    sycl::queue q{sycl::default_selector{}};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    random::rander R;
    ParticleGroup N2(q);

    double Tprt_hot = 1000;
    double Tprt_cold = 300;
    double pmass = 4.652E-26;

    size_t n_particles = 1000000;
    std::vector<Particle> N2_init(n_particles);
    
    for (size_t i = 0; i < n_particles; i++) {
        auto& e = N2_init[i];
        get<position>(e) = Eigen::Vector3d(R(), R(), R());
        if (R() < 0.5) {
            get<velocity>(e) = random::RandFunction3D::RandV_Maxwell(R, Tprt_hot, pmass);
            get<is_hot>(e) = true;
            get<internal_energy>(e) = 1.38065e-23 * Tprt_hot;
        } else {
            get<velocity>(e) = random::RandFunction3D::RandV_Maxwell(R, Tprt_cold, pmass);
            get<is_hot>(e) = false;
            get<internal_energy>(e) = 1.38065e-23 * Tprt_cold;
        }
        get<internal_energy>(e) = 0.0;
        get<random_seed>(e) = R.gen_seed();
    }
    N2.insert(N2_init);

    double v_min = -6000;
    double v_max = 6000;
    int n_bins = 100;
    
    grid1D vx_grid({v_min}, {v_max}, {n_bins - 1});
    VelDistField vx_dist(q, vx_grid);

    grid3D g({0, 0, 0}, {1, 1, 1}, {8, 8, 8});
    Field f_N2(q, g);
    f_N2.setConstant(1e22);

    auto model = N2_dsmc_model::make(N2, f_N2, g);

    double* px = shared_variable<double>(q);
    double* py = shared_variable<double>(q);
    double* pz = shared_variable<double>(q);
    double* E_kin = shared_variable<double>(q);
    double* E_int = shared_variable<double>(q);
    double* T_hot = shared_variable<double>(q);
    double* n_hot = shared_variable<double>(q);
    double* T_cold = shared_variable<double>(q);
    double* n_cold = shared_variable<double>(q);

    std::ofstream energy_file("energy_momentum.plt");
    energy_file << "t,px,py,pz,E_kin,E_int,E_tot,T_hot,T_cold" << std::endl;

    int n_steps = 5000;
    int output_interval = 100;
    double dt = 0.5e-9;
    
    using AllVelDistField = field::host_node_field1D<double, 51>;
    AllVelDistField all_vx_dist(vx_grid);
    
    compute_velocity_distribution(N2, vx_dist, v_min, v_max);
    auto vx_data = vx_dist.getContent().to_host();
    for (int j = 0; j < n_bins; j++)
        all_vx_dist(j)[0] = vx_data[j];
    
    compute_momentum_energy(N2, q, px, py, pz, E_kin, E_int, pmass);
    compute_temperature(N2, q, T_hot, n_hot, T_cold, n_cold, pmass);
    energy_file << 0 << "," << *px << "," << *py << "," << *pz 
                << "," << *E_kin << "," << *E_int << "," << (*E_kin + *E_int)
                << "," << *T_hot << "," << *T_cold << std::endl;
    std::cout << "Step 0: P=(" << *px << ", " << *py << ", " << *pz  << ")" 
              << " E_kin=" << *E_kin << " E_int=" << *E_int << " E_tot=" << (*E_kin+*E_int)
              << " T_hot=" << *T_hot << " T_cold=" << *T_cold << std::endl;
    
    int output_idx = 1;
    for (int i = 1; i <= n_steps; i++) {
        model(dt);
        
        if (i % output_interval == 0) {
            compute_velocity_distribution(N2, vx_dist, v_min, v_max);
            vx_data = vx_dist.getContent().to_host();
            for (int j = 0; j < n_bins; j++)
                all_vx_dist(j)[output_idx] = vx_data[j];
            
            compute_momentum_energy(N2, q, px, py, pz, E_kin, E_int, pmass);
            compute_temperature(N2, q, T_hot, n_hot, T_cold, n_cold, pmass);
            energy_file << i << "," << *px << "," << *py << "," << *pz 
                        << "," << *E_kin << "," << *E_int << "," << (*E_kin + *E_int)
                        << "," << *T_hot << "," << *T_cold << std::endl;
            std::cout << "Step " << i << ": P=(" << *px << ", " << *py << ", " << *pz  << ")" 
                      << " E_kin=" << *E_kin << " E_int=" << *E_int << " E_tot=" << (*E_kin+*E_int)
                      << " T_hot=" << *T_hot << " T_cold=" << *T_cold << std::endl;
            output_idx++;
        }
    }
    
    all_vx_dist.plot("vx_dist_all.plt", "vx_distribution");
    energy_file.close();

}