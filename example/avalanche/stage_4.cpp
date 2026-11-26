#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;

using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>,
    tag_bind<property::random_seed, uint32_t>
>;

using PG = particle_group<Particle, pos_x_nan_is_invalid>;
using Field = node_field2D<double>;
using HField = host_node_field2D<double>;

struct Avalanche_mcc_model {
    using e_tag = psum::particle_collision::species_tags::electron;
    using Ar_tag = psum::particle_collision::species_tags::Argon;

    struct col_ionization {
        using incident_species_tag=e_tag;
        using background_species_tag=Ar_tag;
        static double collision_cross_section(double v2) {
            return v2 > 4e12 ? 2e-20 : 0;
        }
        static void collide(auto& p, const auto& ctx_acc) {
            using namespace psum::particle_collision::foundation;
            auto R = psum::random::view_as_rander(tag::get<property::random_seed>(p));
            double speed = tag::get<property::velocity>(p).norm();

            auto dir1 = psum::random::RandFunction2D::RandV_spherical(R.get_rander());
            tag::get<property::velocity>(p) = dir1 * speed * 0.9;
        }
    };

    static auto make(PG& electrons, PG& atoms, Field& atom_dens) {
        using namespace psum::particle_collision;
        using namespace psum::particle_collision::foundation;


        using all_col = collision_model<col_ionization>;
        using Ctx = species_contexts<all_col, PG, Field>;
        struct Binding { PG* group; Field* density; };
        auto ctx = std::make_shared<Ctx>(
            Binding{&electrons, nullptr}, Binding{&atoms, &atom_dens});

        return [ctx](double dt) {
            execute_collision_model<all_col>(*ctx, dt);
        };
    }
};

void move_particles(PG& pg, const auto& grid, double dt) {
    pg.for_each([&](sycl::handler& h) {
        return [=](Particle& p) {
            tag::get<property::position>(p) += tag::get<property::velocity>(p) * dt;
            if (!grid.inGrid(tag::get<property::position>(p)))
                PG::validator::make_invalid(p);
        };
    });
}

int main() {
    sycl::queue q{sycl::default_selector_v};

    auto grid = simple_grid<2>({0.0, 0.0}, {0.1, 0.1}, {200, 200});

    PG electrons(q);
    PG atoms(q);
    PG ions(q);

    Field atom_dens(q, grid);
    Field ke_dens(q, grid);
    HField atom_host(grid);

    atom_host.setZero();
    Eigen::RowVector2d center{0.07, 0.05};
    double gas_r = 0.025;
    atom_host.for_each([&](size_t, double& v, const auto& pos) {
        v = 2e22 * exp(-((pos - center).squaredNorm() / gas_r / gas_r));
    });
    atom_host.plot("output/atom_dens.plt", "nAr", 0.0);
    atom_dens.copy(atom_host.getContent());

    auto mccm = Avalanche_mcc_model::make(electrons, atoms, atom_dens);

    ofstream count_file("output/count.plt");
    count_file << "variables=time,ele,ion" << endl;

    double dt = 1e-10;
    int steps = 600, interval = 20;

    for (int step = 0; step <= steps; ++step) {
        move_particles(electrons, grid, dt);
        move_particles(ions, grid, dt);

        mccm(dt);

        if (step % interval == 0) {
            ke_dens.setZero();
            electrons.for_each([&](sycl::handler& h) {
                auto da = ke_dens.get_access(h);
                return [=](Particle& p) {
                    double ke = 0.5 * tag::get<property::velocity>(p).squaredNorm();
                    add_back(tag::get<property::position>(p), ke, da);
                };
            });
            ke_dens.plot("output/ke_step_" + to_string(step) + ".plt", "ke", step * dt);

            count_file << step * dt << "\t" << electrons.size() << "\t" << ions.size() << endl;
            cout << "step " << step << ": ele=" << electrons.size() << ", ion=" << ions.size() << endl;
        }

        Particle p0;
        tag::get<property::position>(p0) = {0.001, 0.05};
        tag::get<property::velocity>(p0) = {6e6, 0.0};
        tag::get<property::random_seed>(p0) = global_random::rand_uint();
        electrons.insert({p0});
    }

    count_file.close();
    cout << "Avalanche stage 4: wrote ke snapshots and count.plt." << endl;

    return 0;
}
