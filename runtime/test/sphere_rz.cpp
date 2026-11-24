// 2d axisymmetric charge-cloud expansion on the C runtime
#include "psum_runtime_api.h"
#include <cmath>
#include <random>
#include <string>

int main(int argc, char** argv) {
    const bool robin = argc > 1 && std::string(argv[1]) == "robin";
    const double zlo = -0.4, zhi = 0.4, rhi = 0.4;
    const int nz = 64, nr = 32;
    const double r0 = 0.1, A = 2.5e-3;
    const double Q = 4 * M_PI * A;
    const int Np = 100000;
    const double qi = Q / Np;
    const double dt = 0.01;
    const int steps = 60;
    const double eps_lib = 8.854e-12;

    psum_engine_t eng = psum_engine_create();
    const double lo[2] = {zlo, 0.0}, hi[2] = {zhi, rhi};
    const int n[2] = {nz, nr};
    { psum_obj_grid_desc d = { .name="g", .dim=2, .lower={lo[0], lo[1]}, .upper={hi[0], hi[1]}, .node_count={n[0], n[1]} }; psum_engine_grid_create(eng, &d); }
    { psum_obj_field_desc d = { .name="U", .on_device=1, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(eng, &d, "g"); }
    { psum_obj_field_desc d = { .name="phi", .on_device=1, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(eng, &d, "g"); }
    const int bc[4] = {robin ? PSUM_BC_ROBIN : PSUM_BC_DIRICHLET, PSUM_BC_PERIODIC,
                       robin ? PSUM_BC_ROBIN : PSUM_BC_DIRICHLET,
                       robin ? PSUM_BC_ROBIN : PSUM_BC_DIRICHLET};
    const double robin_center[2] = {0.0, 0.0};
    { psum_opt_poisson_desc d = { .solver_kind=PSUM_POISSON_2D_AXISYMMETRIC, .bc_kinds=bc, .robin_center=robin ? robin_center : NULL, .grid="g", .backend="sycl_multigrid_gpu", .options="I 65 J 33 K 1 depth 3 max_cycles 100 pre_relax 4 post_relax 2", .epsilon=8.854e-12 };
      psum_engine_poisson_init(eng, &d); }
    psum_engine_poisson_bc_value(eng, PSUM_POISSON_2D_AXISYMMETRIC, 0, 0.0);
    psum_engine_poisson_bc_value(eng, PSUM_POISSON_2D_AXISYMMETRIC, 2, 0.0);
    psum_engine_poisson_bc_value(eng, PSUM_POISSON_2D_AXISYMMETRIC, 3, 0.0);
    { psum_obj_species_desc d = { .name="cloud", .dim=3, .has_global_mass=0, .global_mass=0, .has_global_charge=0, .global_charge=0, .has_global_weight=0, .global_weight=0 }; psum_engine_particles_create(eng, &d); }

    std::mt19937_64 rng(20261118);
    std::uniform_real_distribution<double> u(-r0, r0);
    int made = 0;
    while (made < Np) {
        double x = u(rng), y = u(rng), z = u(rng);
        if (x * x + y * y + z * z > r0 * r0) continue;
        double p[3] = {x, y, z}, v[3] = {0, 0, 0};
        psum_engine_particles_append(eng, "cloud", p, v);
        made++;
    }

    for (int step = 0; step < steps; step++) {
        psum_engine_field_zero(eng, "U");
        { psum_opt_deposit_desc d = { .species="cloud", .field="U", .e_id=PSUM_E_DIRECT_AXISYM, .weight=qi }; psum_engine_deposit(eng, &d); }
        psum_engine_deposit_to_density(eng, "U", PSUM_GEOMETRY_AXISYMMETRIC, eps_lib);
        psum_engine_poisson_solve_d(eng, PSUM_POISSON_2D_AXISYMMETRIC, "U", "phi");
        { psum_opt_leapfrog_desc d = { .species="cloud", .e_field="phi", .e_id=PSUM_E_GRADIENT_AXISYM, .q_m=1.0, .dt=dt, .deposit_field="U", .deposit_weight=0.0 }; psum_engine_leapfrog(eng, &d); }
    }

    psum_engine_species_save_mas(eng, "cloud", "/tmp/opencode/sphere_cloud.mas");
    psum_engine_field_save_mas(eng, "phi", "/tmp/opencode/sphere_phi.mas");
    psum_engine_destroy(eng);
    return 0;
}
