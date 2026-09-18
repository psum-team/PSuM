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
    using Ar1_tag = psum::particle_collision::species_tags::Argon_pos_1;
    using species_in_model = std::tuple<e_tag, Ar_tag, Ar1_tag>;

    struct col_ionization {
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

    static auto make(PG& electrons, PG& atoms, PG& ions,
                     Field& ele_dens, Field& atom_dens, Field& ion_dens) {
        using namespace psum::particle_collision;
        using namespace psum::particle_collision::foundation;

        using processor_type = collision_processor<density_interp_2d<Ar_tag>, std::tuple<col_ionization>>;
        using all_col = mcc_model<mcc_submodel_for_incident<e_tag, processor_type>>;
        using Ctx = standard_mccm_context<species_in_model, PG, Field>;

        auto ctx = std::make_shared<Ctx>();
        ctx->template get<e_tag>().bind(electrons, ele_dens);
        ctx->template get<Ar_tag>().bind(atoms, atom_dens);
        ctx->template get<Ar1_tag>().bind(ions, ion_dens);

        return [ctx](double dt) {
            execute_mcc_model<all_col>(*ctx, dt);
            execute_all_clean_buffers<species_in_model>(*ctx);
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

    Field ele_dens(q, grid), atom_dens(q, grid), ion_dens(q, grid);
    HField atom_host(grid);

    atom_host.setZero();
    Eigen::RowVector2d center{0.07, 0.05};
    double gas_r = 0.025;
    atom_host.for_each([&](size_t, double& v, const auto& pos) {
        v = 2e22 * exp(-((pos - center).squaredNorm() / gas_r / gas_r));
    });
    atom_host.plot("output/atom_dens.plt", "nAr", 0.0);
    atom_dens.copy(atom_host.getContent());

    auto mccm = Avalanche_mcc_model::make(electrons, atoms, ions, ele_dens, atom_dens, ion_dens);

    ofstream count_file("output/count.plt");
    count_file << "variables=time,ele,ion" << endl;

    double dt = 1e-10;
    int steps = 600, interval = 20;

    for (int step = 0; step <= steps; ++step) {
        move_particles(electrons, grid, dt);
        move_particles(ions, grid, dt);

        mccm(dt);

        if (step % interval == 0) {
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
    cout << "Avalanche stage 3: wrote count.plt." << endl;

    return 0;
}
