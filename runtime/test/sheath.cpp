// the benchmark/sheath case on the C runtime
#include <cmath>
#include <cstdlib>
#include <vector>
#include "psum_runtime_api.h"

// Maxwellian x-velocity from plain rand(): sin(2 pi R) sqrt(-ln R) * sqrt(2 kT/m)
static double maxwell_x(double T, double mass) {
    double r1 = (double)std::rand() / RAND_MAX, r2 = (double)std::rand() / RAND_MAX;
    return std::sin(2 * 3.1415926535897932 * r1) * std::sqrt(-std::log(r2)) * std::sqrt(2.0 * 1.38065e-23 * T / mass);
}

int main(int argc, char** argv) {
    const long iters = (argc > 1) ? std::atol(argv[1]) : 200000;

    psum_engine_t e = psum_engine_create();

    const double dx = 0.04 / 100, dt = 1.3e-10;
    const int I = 100;
    const double length = dx * I;
    const double mass_ele = 9.1e-31;
    const double mass_ar = mass_ele * 39.9 * 1836;
    const double Te = 2.2 * 11700, Ti = 0.5 * 11700;
    const double weight = 6250000;
    const double q_e = 1.602e-19;

    const double lo[1] = {0.0}, up[1] = {length};
    const int nc[1] = {I};
    { psum_obj_grid_desc d = { .name="g", .dim=1, .lower={lo[0]}, .upper={up[0]}, .node_count={nc[0]} }; psum_engine_grid_create(e, &d); }
    { psum_obj_field_desc d = { .name="q", .on_device=1, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d, "g"); }
    { psum_obj_field_desc d = { .name="phi", .on_device=1, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d, "g"); }
    { psum_obj_field_desc d = { .name="hq", .on_device=0, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d, "g"); }
    { psum_obj_field_desc d = { .name="hphi", .on_device=0, .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d, "g"); }
    { psum_obj_species_desc d = { .name="e", .dim=1, .has_global_mass=0, .global_mass=0, .has_global_charge=0, .global_charge=0, .has_global_weight=0, .global_weight=0 }; psum_engine_particles_create(e, &d); }
    { psum_obj_species_desc d = { .name="i", .dim=1, .has_global_mass=0, .global_mass=0, .has_global_charge=0, .global_charge=0, .has_global_weight=0, .global_weight=0 }; psum_engine_particles_create(e, &d); }

    const int bc[2] = {PSUM_BC_DIRICHLET, PSUM_BC_DIRICHLET};
    { psum_opt_poisson_desc d = { .solver_kind=PSUM_POISSON_1D, .bc_kinds=bc, .robin_center=NULL, .grid="g", .backend="native", .options=NULL, .epsilon=8.854e-12 };
      psum_engine_poisson_init(e, &d); }
    psum_engine_poisson_bc_value(e, PSUM_POISSON_1D, 0, 0.0);
    psum_engine_poisson_bc_value(e, PSUM_POISSON_1D, 1, 12.5);

    std::srand(20261119);
    std::vector<double> xbuf, vbuf;
    size_t dead_e = 0, dead_i = 0;

    for (long iter = 0; iter < iters; iter++) {
        psum_engine_deposit_to_density(e, "q", PSUM_GEOMETRY_CARTESIAN, q_e);
        psum_engine_host_field_copy(e, "hq", "q");
        psum_engine_poisson_solve_h(e, PSUM_POISSON_1D, "hq", "hphi");
        psum_engine_field_copy(e, "phi", "hphi");

        psum_engine_field_zero(e, "q");
        { psum_opt_leapfrog_desc d = { .species="e", .e_field="phi", .e_id=0, .q_m=-q_e / mass_ele, .dt=dt, .deposit_field="q", .deposit_weight=-weight };
          psum_engine_leapfrog(e, &d); }
        { psum_opt_leapfrog_desc d = { .species="i", .e_field="phi", .e_id=0, .q_m=q_e / mass_ar, .dt=dt, .deposit_field="q", .deposit_weight=weight };
          psum_engine_leapfrog(e, &d); }

        { psum_readback rb = { 0, 0, NULL, NULL };
          if (psum_engine_leave_buffer_get(e, "e", &rb) == 0) dead_e += rb.count; }
        { psum_readback rb = { 0, 0, NULL, NULL };
          if (psum_engine_leave_buffer_get(e, "i", &rb) == 0) dead_i += rb.count; }

        auto inject = [&](const char* grp, double T, double mass) {
            xbuf.clear(); vbuf.clear();
            double N = 5e13 * dx / weight;
            while (true) {
                double v = maxwell_x(T, mass);
                double x_ = length + dx * ((double)std::rand() / RAND_MAX);
                if (v * dt + x_ < length) { xbuf.push_back(v * dt + x_); vbuf.push_back(v); }
                N -= 1.0;
                if (!((double)std::rand() / RAND_MAX < N)) break;
            }
            if (!xbuf.empty())
                psum_engine_particles_append_n(e, grp, xbuf.size(), xbuf.data(), vbuf.data());
        };
        inject("i", Ti, mass_ar);
        inject("e", Te, mass_ele);

        size_t ne_n = 0, ni_n = 0;
        psum_engine_particles_count(e, "e", &ne_n);
        psum_engine_particles_count(e, "i", &ni_n);
        if (ne_n < 0.8 * (ne_n + dead_e) && ne_n > 2000) {
            psum_engine_particles_compress(e, "e"); dead_e = 0;
        }
        if (ni_n < 0.8 * (ni_n + dead_i) && ni_n > 2000) {
            psum_engine_particles_compress(e, "i"); dead_i = 0;
        }
    }

    psum_engine_species_save_mas(e, "e", "/tmp/opencode/sheath_e.mas");
    psum_engine_species_save_mas(e, "i", "/tmp/opencode/sheath_i.mas");
    psum_engine_field_save_mas(e, "phi", "/tmp/opencode/sheath_phi.mas");
    psum_engine_destroy(e);
    return 0;
}
