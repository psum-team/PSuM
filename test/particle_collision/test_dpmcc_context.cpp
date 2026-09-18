#include <iostream>
#include <map>
#include <string>
#include <Eigen/Dense>
#include "../../include/psum/tag.hpp"
#include "../../include/psum/field.hpp"
#include "../../include/psum/particle_container.hpp"
#include "../../src/particle_collision/foundation.hpp"
#include "../../src/particle_collision/species_tags.hpp"
#include "../../src/particle_collision/mcc_model.hpp"
#include "../../src/particle_collision/dpmcc_model.hpp"
#include "../../src/utils_sycl.hpp"
#include <cxxabi.h>

using namespace std;
using namespace psum;
using namespace psum::utils_sycl;
using namespace tag;
using namespace tag::property;
using namespace psum::particle_collision;
using namespace psum::particle_collision::foundation;

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>,
    tag_bind<weight, double>
>;

using ParticleGroup = particle_container::particle_group<particle, particle_container::pos_x_nan_is_invalid>;
using Field = field::node_field2D<double>;

struct Xe_ioni_dpmcc_model {
    using ele_tag = species_tags::electron;
    using Xe_atom_tag = species_tags::Xenon;
    using Xe_ion1_tag = species_tags::Xenon_pos_1;

    struct col_ele_to_Xe_atom_ionization: base_of_deferred_pairing_collision {
        static double collision_cross_section(double v2) { return 0.5e-19; }
        inline static void collide(particle& p, const auto& ctx) {
            insert_deferred_pairing_in_context<col_ele_to_Xe_atom_ionization>(ctx, p);
        }
        using backgroud_speices = Xe_atom_tag;
        inline static void pair_collide(auto& p, auto& p_atom, const auto& ctx) {
            double energy = get<velocity>(p).squaredNorm() * 9.11e-31 * 0.5 / 1.602e-19;
            if (energy > 10) {
                double ioni_weight = std::min(get<weight>(p), get<weight>(p_atom));
                particle new_ele;
                get<position>(new_ele) = get<position>(p);
                get<velocity>(new_ele) = get<velocity>(p) * 0.5;
                get<weight>(new_ele) = ioni_weight;
                particle new_ion;
                get<position>(new_ion) = get<position>(p);
                get<velocity>(new_ion).setZero();
                get<weight>(new_ion) = ioni_weight;
                insert_particle_in_context<ele_tag>(ctx, new_ele);
                insert_particle_in_context<Xe_ion1_tag>(ctx, new_ion);
                get<velocity>(p) *= 0.5;
                get<weight>(p_atom) -= ioni_weight;
            }
        }
    };
};

void test_prototype() {

    using species_context = mcc_species_context<ParticleGroup, Field>;
    using mccm_context = mcc_module_context<
        tag_bind<species_tags::Xenon_pos_1, species_context>,
        tag_bind<species_tags::electron, species_context>,
        tag_bind<species_tags::Xenon, species_context>
    >;


    using collisions_by_ele = sequence_executor<
        collision_processor<
            density_interp_2d<species_tags::Xenon>,
            std::tuple<Xe_ioni_dpmcc_model::col_ele_to_Xe_atom_ionization>
        >
    >;

    using dpmcc_context = dpmcc_model_context<
        deferred_pairing_filter_tuple<collisions_by_ele::col_struct_tuple>,
        particle
    >;

    using pengine = pairing_engine<
        ParticleGroup,
        field::grid3D
    >;
        
    combined_context<mccm_context, dpmcc_context> context;

    sycl::queue q{sycl::default_selector{}};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    ParticleGroup ele(q), xe_atom(q), xe_ion1(q);
    field::grid2D g({-1.0, -1.0}, {1.0, 1.0}, {10, 10});
    field::node_field2D<double> f_ele(q, g), f_xe_atom(q, g), f_xe_ion1(q, g);
    constexpr bool a = context.a_ctx_.exist<species_tags::electron>(); // bind electrons to context
    context.get<species_tags::electron>().bind(ele, f_ele); // bind electrons to context
    context.get<species_tags::Xenon>().bind(xe_atom, f_xe_atom); // bind Xe atoms to context
    context.get<species_tags::Xenon_pos_1>().bind(xe_ion1, f_xe_ion1); // bind Xe ions to context
    context.get<Xe_ioni_dpmcc_model::col_ele_to_Xe_atom_ionization>().bind(ele);
    f_xe_atom.setConstant(1e20);
    pengine eng(q, field::grid3D({-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0}, {32, 32, 32}));

    random::rander R;

    std::vector<particle> electrons_init(100000);
    for (auto& e : electrons_init) {
        get<position>(e) = Eigen::Vector3d(R(), R(), R());
        get<velocity>(e) = random::RandFunction3D::RandV_Maxwell(R, 20*11600, 9.11e-31);
        get<weight>(e) = 1.0;
        get<random_seed>(e) = R.gen_seed();
    }
    ele.insert(electrons_init);

    std::vector<particle> atoms_init(100000);
    for (auto& a : atoms_init) {
        get<position>(a) = Eigen::Vector3d(R(), R(), R());
        get<velocity>(a) = Eigen::Vector3d(0.0, 0.0, 0.0);
        get<weight>(a) = 100.0;
        get<random_seed>(a) = R.gen_seed();
    }
    xe_atom.insert(atoms_init);

    double* tot_energy = shared_variable<double>(q);
    double* tot_weight = shared_variable<double>(q);
    auto calculate_energy_and_amount = [tot_energy, tot_weight](ParticleGroup& ps) {
        *tot_energy = 0.0;
        *tot_weight = 0.0;
        ps.for_each([&](sycl::handler &h) {
            return [=](particle& p) {
                atomic_add(*tot_energy, get<velocity>(p).squaredNorm() * 9.11e-31 * 0.5 / 1.602e-19);
                atomic_add(*tot_weight, get<weight>(p));
            };
        });
    };

    auto report_energy_and_amount = [&ele, &xe_atom, &xe_ion1, tot_energy, tot_weight, calculate_energy_and_amount]
    {
        calculate_energy_and_amount(ele);
        std::cout << "elec_E: " << *tot_energy << ", elec_sum: " << *tot_weight << std::endl;
        calculate_energy_and_amount(xe_ion1);
        std::cout << "ion1_E: " << *tot_energy << ", ion1_sum: " << *tot_weight << std::endl;
        calculate_energy_and_amount(xe_atom);
        std::cout << "atom_E: " << *tot_energy << ", atom_sum: " << *tot_weight << std::endl;
    };

    report_energy_and_amount();
    for (int i = 0; i < 100; i++) {
        ele.for_each([&](sycl::handler &h) {
            auto ctx_acc = context.get_access(h);
            return [=](particle& p) {
                collisions_by_ele::execute(p, 1e-10, ctx_acc);
            };
        });

        eng.deal(context.get<species_tags::Xenon>().get_group(),
                 context.get<Xe_ioni_dpmcc_model::col_ele_to_Xe_atom_ionization>().buffer);

        context.get<Xe_ioni_dpmcc_model::col_ele_to_Xe_atom_ionization>().clean_buffer(context);

        xe_ion1.insert(context.template get<species_tags::Xenon_pos_1>().buffer);
        ele.insert(context.template get<species_tags::electron>().buffer);
        context.template get<species_tags::Xenon_pos_1>().buffer.clear();
        context.template get<species_tags::electron>().buffer.clear();

        if (i % 20 == 0) {
            std::cout << "iter: " << i << "\t";
            report_energy_and_amount();
        }
    }
}

int main() {
    test_prototype();
}