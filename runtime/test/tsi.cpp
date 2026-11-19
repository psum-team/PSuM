// benchmark/TSI on the C runtime
#include <cmath>
#include <random>
#include <vector>
#include "psum_runtime_api.h"

static double wrapL(double x, double L) {
    while (x < 0.0) x += L;
    while (x >= L) x -= L;
    return x;
}

int main() {
    psum_engine_t e = psum_engine_create();

    const double length = 2 * M_PI;
    const int cell_num = 64, ppc = 154, iters = 1000;
    const double q_m = 1, dt = 0.125, VT = 0.01, VD = 0.2;
    const int N = ppc * cell_num;
    const double Q = 1.0 * 1.0 / (q_m * (double(N) / length) * 4 * M_PI);   // omega_p = 1
    const double rho_bg = -Q * N / length;

    const double lo[1] = {0.0}, up[1] = {length};
    const int nc[1] = {cell_num};
    { psum_grid_desc d = { .name="g", .dim=1, .lower=lo, .upper=up, .node_count=nc }; psum_engine_grid_create(e, &d); }
    { psum_field_desc d = { .name="cnt", .grid="g", .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d); }
    { psum_field_desc d = { .name="rho", .grid="g", .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d); }
    { psum_field_desc d = { .name="phi", .grid="g", .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_field_create(e, &d); }
    { psum_field_desc d = { .name="hr", .grid="g", .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_host_field_create(e, &d); }
    { psum_field_desc d = { .name="hphi", .grid="g", .quantity_size=1, .scalar_kind=PSUM_SCALAR_DOUBLE }; psum_engine_host_field_create(e, &d); }

    std::mt19937 gen(123456789);
    std::uniform_real_distribution<double> R(0.0, 1.0);
    std::vector<double> pos(N), vel(N);
    for (int i = 0; i < N; i++) {
        double p = i * length / N;
        pos[i] = p;
        vel[i] = ((i % 2 == 0) ? VD : -VD)
               + VT * cos(2 * M_PI * R(gen)) * sqrt(-2 * log(R(gen)))
               + 0.1 * VT * sin(2 * M_PI * p / length * 2);
    }
    psum_engine_particles_create(e, "e", 1);
    size_t accepted = 0;
    { psum_particles_desc d = { .group="e" };
      psum_engine_particles_append_deposit(e, &d, N, pos.data(), vel.data(), "cnt", 0, 1.0, &accepted); }

    const int bc[2] = {PSUM_BC_DIRICHLET, PSUM_BC_DIRICHLET};
    { psum_poisson_desc d = { .solver_kind=PSUM_POISSON_1D, .bc_kinds=bc, .robin_center=NULL, .grid="g", .backend="native", .options=NULL, .epsilon=0.25 / M_PI };
      psum_engine_poisson_init(e, &d); }
    psum_engine_poisson_bc_value(e, PSUM_POISSON_1D, 0, 0.0);
    psum_engine_poisson_bc_value(e, PSUM_POISSON_1D, 1, 0.0);

    for (int iter = 0; iter < iters; iter++) {
        psum_engine_field_copy(e, "rho", "cnt");
        psum_engine_deposit_to_density(e, "rho", PSUM_GEOMETRY_CARTESIAN, Q);
        psum_engine_field_add(e, "rho", rho_bg);
        psum_engine_host_field_copy(e, "hr", "rho");
        psum_engine_poisson_solve_h(e, PSUM_POISSON_1D, "hr", "hphi");
        psum_engine_field_copy(e, "phi", "hphi");

        psum_engine_field_zero(e, "cnt");
        { psum_leapfrog_desc d = { .group="e", .e_field="phi", .e_id=0, .q_m=q_m, .dt=dt, .deposit_field="cnt", .deposit_weight=1.0 };
          psum_engine_leapfrog(e, &d); }

        size_t lc = 0; int ld = 0;
        psum_readback rb = { 0, 0, NULL, NULL };
        if (psum_engine_leave_buffer_get(e, "e", &rb) == 0) { lc = rb.count; ld = rb.dim; }
        if (lc > 0) {
            std::vector<double> lp(lc * ld), lv(lc * ld), wp(lc * ld), wv(lc * ld);
            { psum_readback rb2 = { 0, 0, lp.data(), lv.data() }; psum_engine_leave_buffer_get(e, "e", &rb2); }
            for (size_t k = 0; k < lc; k++) {
                wp[k * ld] = wrapL(lp[k * ld], length);
                wv[k * ld] = lv[k * ld];
            }
            size_t a = 0;
            { psum_particles_desc d = { .group="e" };
              psum_engine_particles_append_deposit(e, &d, lc, wp.data(), wv.data(), "cnt", 0, 1.0, &a); }
        }
    }

    psum_engine_group_save_mas(e, "e", "/tmp/opencode/tsi_e.mas");
    psum_engine_field_save_mas(e, "phi", "/tmp/opencode/tsi_phi.mas");
    psum_engine_destroy(e);
    return 0;
}
