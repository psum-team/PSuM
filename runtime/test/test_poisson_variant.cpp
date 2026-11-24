// Poisson probes straight against poisson.hpp — no engine,
// mirroring test_push_variant: raw grids + fields + solver slots.
// Every case has an analytic anchor; solver_kind/bc values follow
// psum_runtime_api.h (0 periodic, 1 dirichlet, 2 robin, 3 neumann).
#include <psum/psum.hpp>
#include <iostream>
#include "../src/poisson.hpp"

using namespace psum::prelude;
using namespace psum_runtime::poisson;

static int failures = 0;

static void expect(const char* what, double got, double want, double rel = 1e-6) {
    double scale = want < 0 ? -want : (want > 1 ? want : 1);
    bool ok = got - want < rel * scale && want - got < rel * scale;
    if (!ok) failures++;
    std::cout << what << ": got " << got << " want " << want << (ok ? "" : "  FAIL") << "\n";
}

static void check(int code, const char* what) {
    if (code != 0) { failures++; std::cout << "FAIL " << what << ": " << code << "\n"; }
}

// host field midpoint value (node indices, row-major grid layout)
template <typename F, typename... Idx>
static double mid_of(F& f, Idx... idx) {
    return double(f.data()[f.getGrid().n2i({int(idx)...})]);
}

int main() {
    sycl::queue q{ sycl::default_selector_v };

    {   // P1: 1d dirichlet linear ramp (phi = 0 at L, 100 at R)
        serialization::object_manager mng{"", q};
        mng.obj<grid1>("g") = grid1({0.0}, {10.0}, {64});
        host_field1<1>& rho = mng.obj<host_field1<1>>("rho") = host_field1<1>(mng.obj<grid1>("g"));
        rho.setConstant(0.0);
        host_field1<1>& phi = mng.obj<host_field1<1>>("phi") = host_field1<1>(mng.obj<grid1>("g"));
        phi.setConstant(0.0);
        const int bc[2] = {1, 1};
        slots s;
        check(init(mng, s, 0, bc, nullptr, "g", "native", nullptr, 8.854e-12), "P1 init");
        check(bc_value(mng, s, 0, 0, 0.0), "P1 bc L");
        check(bc_value(mng, s, 0, 1, 100.0), "P1 bc R");
        check(solve_h(mng, s, 0, "rho", "phi"), "P1 solve");
        expect("P1 1d linear mid", mid_of(phi, 32), 50.0);
    }
    {   // P2: 1d constant source, epsilon pass-through: eps=1 -> mid=25
        serialization::object_manager mng{"", q};
        mng.obj<grid1>("g") = grid1({0.0}, {10.0}, {64});
        host_field1<1>& rho = mng.obj<host_field1<1>>("rho") = host_field1<1>(mng.obj<grid1>("g"));
        rho.setConstant(2.0);
        host_field1<1>& phi = mng.obj<host_field1<1>>("phi") = host_field1<1>(mng.obj<grid1>("g"));
        phi.setConstant(0.0);
        const int bc[2] = {1, 1};
        slots s;
        check(init(mng, s, 0, bc, nullptr, "g", "native", nullptr, 1.0), "P2 init eps=1");
        check(bc_value(mng, s, 0, 0, 0.0), "P2 bc L");
        check(bc_value(mng, s, 0, 1, 0.0), "P2 bc R");
        check(solve_h(mng, s, 0, "rho", "phi"), "P2 solve");
        expect("P2 1d source mid", mid_of(phi, 32), 25.0);
        rho.setConstant(4.0);
        check(solve_h(mng, s, 0, "rho", "phi"), "P2 solve eps=2");
        expect("P2 rho/eps doubling mid", mid_of(phi, 32), 50.0);
    }
    {   // P3: 2d cartesian periodic y + dirichlet x, bc_value live update,
        //     re-init refused
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        host_field1<2>& rho = mng.obj<host_field1<2>>("rho") = host_field1<2>(mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        host_field1<2>& phi = mng.obj<host_field1<2>>("phi") = host_field1<2>(mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {0, 0, 1, 1};
        slots s;
        check(init(mng, s, 1, bc, nullptr, "g", "native", nullptr), "P3 init");
        check(bc_value(mng, s, 1, 2, 200.0), "P3 bc E");
        check(bc_value(mng, s, 1, 3, 0.0), "P3 bc W");
        check(solve_h(mng, s, 1, "rho", "phi"), "P3 solve");
        expect("P3 2d periodic+D mid", mid_of(phi, 8, 8), 100.0);
        check(bc_value(mng, s, 1, 3, 100.0), "P3 bc W live");
        check(solve_h(mng, s, 1, "rho", "phi"), "P3 resolve");
        expect("P3 bc_value no re-init", mid_of(phi, 8, 8), 150.0);
        expect("P3 second init refused", (double)init(mng, s, 1, bc, nullptr, "g", "native", nullptr), -2.0);
    }
    {   // P4: axisymmetric dirichlet N + periodic elsewhere, constant phi
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        host_field1<2>& rho = mng.obj<host_field1<2>>("rho") = host_field1<2>(mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        host_field1<2>& phi = mng.obj<host_field1<2>>("phi") = host_field1<2>(mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {1, 0, 0, 0};
        slots s;
        check(init(mng, s, 2, bc, nullptr, "g", "native", nullptr), "P4 init");
        check(bc_value(mng, s, 2, 0, 5.0), "P4 bc N");
        check(solve_h(mng, s, 2, "rho", "phi"), "P4 solve");
        expect("P4 axisym const mid", mid_of(phi, 8, 8), 5.0);
    }
    {   // P5: 2d cartesian gpu multigrid, device route
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        field1<2>& rho = mng.obj<field1<2>>("rho") = field1<2>(q, mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        field1<2>& phi = mng.obj<field1<2>>("phi") = field1<2>(q, mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {0, 0, 1, 1};
        slots s;
        check(init(mng, s, 1, bc, nullptr, "g", "sycl_multigrid_gpu",
                   "I 17 J 17 K 1 depth 3 max_cycles 100 pre_relax 4 post_relax 2"), "P5 init");
        check(bc_value(mng, s, 1, 2, 200.0), "P5 bc E");
        check(bc_value(mng, s, 1, 3, 0.0), "P5 bc W");
        check(solve_d(mng, s, 1, "rho", "phi"), "P5 solve_d");
        auto host = phi.getContent().to_host();
        expect("P5 2d gpu solve_d mid",
               double(host[phi.getGrid().n2i({8, 8})]), 100.0, 1e-3);
    }
    {   // P6: axisymmetric gpu, device route
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        field1<2>& rho = mng.obj<field1<2>>("rho") = field1<2>(q, mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        field1<2>& phi = mng.obj<field1<2>>("phi") = field1<2>(q, mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {1, 0, 0, 0};
        slots s;
        check(init(mng, s, 2, bc, nullptr, "g", "sycl_multigrid_gpu",
                   "I 17 J 17 K 1 depth 3 max_cycles 100 pre_relax 4 post_relax 2"), "P6 init");
        check(bc_value(mng, s, 2, 0, 5.0), "P6 bc N");
        check(solve_d(mng, s, 2, "rho", "phi"), "P6 solve_d");
        auto host = phi.getContent().to_host();
        expect("P6 axisym gpu mid", double(host[phi.getGrid().n2i({8, 8})]), 5.0, 1e-3);
    }
    {   // P7: 3d gpu, device route
        serialization::object_manager mng{"", q};
        mng.obj<grid3>("g") = grid3({0.0, 0.0, 0.0}, {10.0, 10.0, 10.0}, {8, 8, 8});
        field1<3>& rho = mng.obj<field1<3>>("rho") = field1<3>(q, mng.obj<grid3>("g"));
        rho.setConstant(0.0);
        field1<3>& phi = mng.obj<field1<3>>("phi") = field1<3>(q, mng.obj<grid3>("g"));
        phi.setConstant(0.0);
        const int bc[6] = {1, 1, 0, 0, 0, 0};
        slots s;
        check(init(mng, s, 3, bc, nullptr, "g", "sycl_multigrid_gpu",
                   "I 9 J 9 K 9 depth 2 max_cycles 100 pre_relax 4 post_relax 2"), "P7 init");
        check(bc_value(mng, s, 3, 0, 0.0), "P7 bc Xpos");
        check(bc_value(mng, s, 3, 1, 200.0), "P7 bc Xneg");
        check(solve_d(mng, s, 3, "rho", "phi"), "P7 solve_d");
        auto host = phi.getContent().to_host();
        expect("P7 3d gpu mid", double(host[phi.getGrid().n2i({4, 4, 4})]), 100.0, 1e-3);
    }
    {   // P8/P9: cpu multigrid backends, host route
        serialization::object_manager mng{"", q};
        mng.obj<grid3>("g3") = grid3({0.0, 0.0, 0.0}, {10.0, 10.0, 10.0}, {8, 8, 8});
        host_field1<3>& rho3 = mng.obj<host_field1<3>>("rho3") = host_field1<3>(mng.obj<grid3>("g3"));
        rho3.setConstant(0.0);
        host_field1<3>& phi3 = mng.obj<host_field1<3>>("phi3") = host_field1<3>(mng.obj<grid3>("g3"));
        phi3.setConstant(0.0);
        const int bc3[6] = {1, 1, 0, 0, 0, 0};
        slots s;
        check(init(mng, s, 3, bc3, nullptr, "g3", "eigen_multigrid_cpu",
                   "I 9 J 9 K 9 depth 2 max_cycles 100 pre_relax 4 post_relax 2"), "P8 init");
        check(bc_value(mng, s, 3, 0, 0.0), "P8 bc Xpos");
        check(bc_value(mng, s, 3, 1, 200.0), "P8 bc Xneg");
        check(solve_h(mng, s, 3, "rho3", "phi3"), "P8 solve_h");
        expect("P8 3d cpu-mg mid", mid_of(phi3, 4, 4, 4), 100.0, 1e-3);

        mng.obj<grid2>("g2") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        host_field1<2>& rho2 = mng.obj<host_field1<2>>("rho2") = host_field1<2>(mng.obj<grid2>("g2"));
        rho2.setConstant(0.0);
        host_field1<2>& phi2 = mng.obj<host_field1<2>>("phi2") = host_field1<2>(mng.obj<grid2>("g2"));
        phi2.setConstant(0.0);
        const int bc2[4] = {0, 0, 1, 1};
        check(init(mng, s, 1, bc2, nullptr, "g2", "eigen_multigrid_cpu",
                   "I 17 J 17 K 1 depth 3 max_cycles 100 pre_relax 4 post_relax 2"), "P9 init");
        check(bc_value(mng, s, 1, 2, 200.0), "P9 bc E");
        check(bc_value(mng, s, 1, 3, 0.0), "P9 bc W");
        check(solve_h(mng, s, 1, "rho2", "phi2"), "P9 solve_h");
        expect("P9 2d cpu-mg mid", mid_of(phi2, 8, 8), 100.0, 1e-3);
    }

    {   // P10: unknown backend auto-degrades to native and still solves
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        host_field1<2>& rho = mng.obj<host_field1<2>>("rho") = host_field1<2>(mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        host_field1<2>& phi = mng.obj<host_field1<2>>("phi") = host_field1<2>(mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {0, 0, 1, 1};
        slots s;
        check(init(mng, s, 1, bc, nullptr, "g", "no_such_backend", nullptr), "P10 init degraded");
        check(bc_value(mng, s, 1, 2, 200.0), "P10 bc E");
        check(bc_value(mng, s, 1, 3, 0.0), "P10 bc W");
        check(solve_h(mng, s, 1, "rho", "phi"), "P10 solve");
        expect("P10 native degrade mid", mid_of(phi, 8, 8), 100.0);
    }

    {   // P11: degrade + DEVICE route: solve_d round-trips through host
        serialization::object_manager mng{"", q};
        mng.obj<grid2>("g") = grid2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
        field1<2>& rho = mng.obj<field1<2>>("rho") = field1<2>(q, mng.obj<grid2>("g"));
        rho.setConstant(0.0);
        field1<2>& phi = mng.obj<field1<2>>("phi") = field1<2>(q, mng.obj<grid2>("g"));
        phi.setConstant(0.0);
        const int bc[4] = {0, 0, 1, 1};
        slots s;
        check(init(mng, s, 1, bc, nullptr, "g", "no_such_backend", nullptr), "P11 init degraded");
        check(bc_value(mng, s, 1, 2, 200.0), "P11 bc E");
        check(bc_value(mng, s, 1, 3, 0.0), "P11 bc W");
        check(solve_d(mng, s, 1, "rho", "phi"), "P11 solve_d");
        auto host = phi.getContent().to_host();
        expect("P11 device degrade mid", double(host[phi.getGrid().n2i({8, 8})]), 100.0);
    }

    if (failures) { std::cout << "test_poisson_variant: " << failures << " FAILURES\n"; return 1; }
    std::cout << "positive: 11 probes executed\n";
    return 0;
}
