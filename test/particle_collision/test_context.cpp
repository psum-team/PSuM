#include <iostream>
#include <map>
#include <string>
#include <Eigen/Dense>
#include "../../include/psum/tag.hpp"
#include "../../include/psum/field.hpp"
#include "../../include/psum/particle_container.hpp"
#include "../../src/particle_collision/foundation.hpp"
#include "../../src/particle_collision/mcc_model.hpp"
#include "../../src/utils_sycl.hpp"

using namespace psum;
using namespace utils_sycl;
using namespace tag;
using namespace tag::property;
using namespace psum::particle_collision;
using namespace psum::particle_collision::foundation;

#define MAKE_SPECIES(name) struct name : tag::foundation::abstract_tag { inline const static std::string tag_name = #name; }

using particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_container::particle_group<particle, particle_container::pos_x_nan_is_invalid>;
using Field = field::node_field2D<double>;

MAKE_SPECIES(mccm_electron_species);
MAKE_SPECIES(mccm_Xe_atom_species);
MAKE_SPECIES(mccm_Xe_ion1_species);

using species_context = mcc_species_context<ParticleGroup, Field>;
using mccm_context = mcc_module_context<
    tag_bind<mccm_electron_species, species_context>,
    tag_bind<mccm_Xe_atom_species, species_context>,
    tag_bind<mccm_Xe_ion1_species, species_context>
>;

void test_prototype() {

    struct Xe_atom_density_compute {
        static double compute(const Eigen::Vector3d& pos, const mccm_context::acc_type& ctx_acc) {
            return field::interp({pos.x(), pos.y()}, ctx_acc.get<mccm_Xe_atom_species>().field_acc);
        }
    };
    struct Xe_ion1_density_compute {
        static double compute(const Eigen::Vector3d& pos, const mccm_context::acc_type& ctx_acc) {
            return field::interp({pos.x(), pos.y()}, ctx_acc.get<mccm_Xe_ion1_species>().field_acc);
        }
    };

    struct col_ele_to_Xe_atom_elastic {
        static double collision_cross_section(double v2) { return 1.0e-19; }
        static void collide(particle& p, const mccm_context::acc_type&) {
            get<velocity>(p) *= 0.8;
        }
    };
    struct col_ele_to_Xe_atom_ionization {
        static double collision_cross_section(double v2) { return 0.5e-19; }
        static void collide(particle& p, const mccm_context::acc_type& ctx_acc) {
            double energy = get<velocity>(p).squaredNorm() * 9.11e-31 * 0.5 / 1.602e-19;
            if (energy > 10) {
                particle new_ele;
                get<position>(new_ele) = get<position>(p);
                get<velocity>(new_ele) = get<velocity>(p) * 0.5;
                particle new_ion;
                get<position>(new_ion) = get<position>(p);
                get<velocity>(new_ion).setZero();
                ctx_acc.get<mccm_electron_species>().buffer_acc.push_back(new_ele);
                ctx_acc.get<mccm_Xe_ion1_species>().buffer_acc.push_back(new_ion);
                get<velocity>(p) *= 0.5;
            }
        }
    };

    struct col_ele_to_Xe_ion1_excitation {
        static double collision_cross_section(double v2) { return 0.2e-18; }
        static void collide(particle& p, const mccm_context::acc_type&) {
            get<velocity>(p) *= 0.8;
        }
    };

    using collisions_by_ele = sequence_executor<
        collision_processor<
            Xe_atom_density_compute, 
            std::tuple<col_ele_to_Xe_atom_elastic, col_ele_to_Xe_atom_ionization>
        >,
        collision_processor<
            Xe_ion1_density_compute, 
            std::tuple<col_ele_to_Xe_ion1_excitation>
        >
    >;



    sycl::queue q{sycl::default_selector()};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    mccm_context context;
    ParticleGroup ele(q), xe_atom(q), xe_ion1(q);
    field::grid2D g({-1.0, -1.0}, {1.0, 1.0}, {10, 10});
    field::node_field2D<double> f_ele(q, g), f_xe_atom(q, g), f_xe_ion1(q, g);
    get<mccm_electron_species>(context).bind(ele, f_ele); // bind electrons to context
    get<mccm_Xe_atom_species>(context).bind(xe_atom, f_xe_atom); // bind Xe atoms to context
    get<mccm_Xe_ion1_species>(context).bind(xe_ion1, f_xe_ion1); // bind Xe ions to context
    f_xe_atom.setConstant(1e20);

    std::vector<particle> electrons_init(1000000);
    random::rander R;
    for (auto& e : electrons_init) {
        get<position>(e) = Eigen::Vector3d(0.0, 0.0, 0.0);
        get<velocity>(e) = random::RandFunction3D::RandV_Maxwell(R, 20*11600, 9.11e-31);
        get<random_seed>(e) = R.gen_seed();
    }
    ele.insert(electrons_init);

    double* tot_energy = shared_variable<double>(q);
    auto calculate_energy = [&ele, tot_energy]() {
        *tot_energy = 0.0;
        ele.for_each([&](sycl::handler &h) {
            return [=](particle& p) {
                atomic_add(*tot_energy, get<velocity>(p).squaredNorm() * 9.11e-31 * 0.5 / 1.602e-19);
            };
        });
    };

    calculate_energy();
    std::cout << "iter: 0, E: " << *tot_energy << std::endl;
    for (int i = 0; i < 1000; i++) {
        ele.for_each([&](sycl::handler &h) {
            auto ctx_acc = context.get_access(h);
            return [=](particle& p) {
                collisions_by_ele::execute(p, 1e-10, ctx_acc);
            };
        });

        xe_ion1.insert(get<mccm_Xe_ion1_species>(context).buffer);
        ele.insert(get<mccm_electron_species>(context).buffer);
        get<mccm_Xe_ion1_species>(context).buffer.clear();
        get<mccm_electron_species>(context).buffer.clear();

        if (i % 20 == 0) {
            std::cout << "iter: " << i << ", N of ele: " << ele.size() << ", N of Xe+: " << xe_ion1.size() << std::endl;
        }
    }
    calculate_energy();
    std::cout << "iter: " << 1000 << ", N of ele: " << ele.size() << ", N of Xe+: " << xe_ion1.size() << ", E: " << *tot_energy << std::endl;
}

int main() {
    test_prototype();
    return 0;
}