#include <psum/psum.hpp>
#include <iostream>
#include "../src/particle_operation.hpp"

using namespace psum::prelude;
using namespace psum_runtime;
using namespace psum_runtime::particle_operation;
using property::position;
using property::velocity;

using P1 = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 1>>,
    tag_bind<velocity, Eigen::Vector<double, 1>>
>;
using P2 = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 2>>,
    tag_bind<velocity, Eigen::Vector<double, 2>>
>;
using P3 = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 3>>,
    tag_bind<velocity, Eigen::Vector<double, 3>>
>;

template <typename P>
using PG = particle_group<P, pos_x_nan_is_invalid>;

using acc_1D1C = node_field1D<double>::acc_type;    // 1d scalar
using acc_2D1C = node_field2D<double>::acc_type;    // 2d scalar
using acc_2D2C = node_field2D<double, 2>::acc_type; // 2d (z, r) or (x, y)
using acc_2D3C = node_field2D<double, 3>::acc_type; // 2d grid, 3 components
using acc_3D1C = node_field3D<double>::acc_type;    // 3d scalar
using acc_3D2C = node_field3D<double, 2>::acc_type; // 3d, (x, y)
using acc_3D3C = node_field3D<double, 3>::acc_type; // 3d, 3 components

using e_direct = e_strategy<eCalcDirect, cartesianMap, 1>;
using e_gradient = e_strategy<eCalcGradient, cartesianMap, 1>;
using e_auto = e_strategy<eCalcAuto, cartesianMap, 1>;
using e_axisym_direct = e_strategy<eCalcDirect, axisymmetricMap, 1>;
using e_axisym_gradient = e_strategy<eCalcGradient, axisymmetricMap, 1>;
using e_order4 = e_strategy<eCalcDirect, cartesianMap, 4>;
using b_default = b_strategy<cartesianMap, 1>;
using b_axisym = b_strategy<axisymmetricMap, 1>;
using b_order4 = b_strategy<cartesianMap, 4>;
using boris_e_direct = push_strategy<e_direct, b_default>;
using boris_e_gradient = push_strategy<e_gradient, b_default>;
using boris_axisym_both = push_strategy<e_axisym_direct, b_axisym>;
using boris_axisym_grad_both = push_strategy<e_axisym_gradient, b_axisym>;
using boris_e_gradient_b_axisym = push_strategy<e_gradient, b_axisym>;
using boris_e_axisym_direct = push_strategy<e_axisym_direct, b_default>;
using boris_order4 = push_strategy<e_order4, b_order4>;

// 1d E direct
inline constexpr auto pos1 = [](P1& p, const acc_1D1C& e) {
    particle_operation::leapfrog<e_direct>(p, e, 1.0, 0.01);
};
// 1d phi gradient
inline constexpr auto pos2 = [](P1& p, const acc_1D1C& e) {
    particle_operation::leapfrog<e_gradient>(p, e, 1.0, 0.01);
};
// 2d E direct + B on 1d grid
inline constexpr auto pos3 = [](P2& p, const acc_2D1C& e, const acc_1D1C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 2d E direct + B on 2d grid
inline constexpr auto pos4 = [](P2& p, const acc_2D1C& e, const acc_2D1C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 3d axisym E direct + axisym B QS=2
inline constexpr auto pos5 = [](P3& p, const acc_2D2C& e, const acc_2D2C& b) {
    particle_operation::boris<boris_axisym_both>(p, e, b, 1.0, 0.01);
};
// 3d axisym phi E + axisym B QS=2
inline constexpr auto pos6 = [](P3& p, const acc_2D1C& e, const acc_2D2C& b) {
    particle_operation::boris<boris_axisym_grad_both>(p, e, b, 1.0, 0.01);
};
// 3d cartesian phi E + axisym B QS=2
inline constexpr auto pos7 = [](P3& p, const acc_3D1C& e, const acc_2D2C& b) {
    particle_operation::boris<boris_e_gradient_b_axisym>(p, e, b, 1.0, 0.01);
};
// 3d cartesian phi E + axisym B QS=3
inline constexpr auto pos8 = [](P3& p, const acc_3D1C& e, const acc_2D3C& b) {
    particle_operation::boris<boris_e_gradient_b_axisym>(p, e, b, 1.0, 0.01);
};
// 3d E 2d-grid QS=3 + B 3d QS=3
inline constexpr auto pos9 = [](P3& p, const acc_2D3C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 3d E 3d QS=3 + B 3d QS=3
inline constexpr auto pos10 = [](P3& p, const acc_3D3C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 3d phi E 3d + B 3d QS=3
inline constexpr auto pos11 = [](P3& p, const acc_3D1C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_gradient>(p, e, b, 1.0, 0.01);
};
// 2d phi E + scalar B (kernel A)
inline constexpr auto pos12 = [](P2& p, const acc_2D1C& e, const acc_2D1C& b) {
    particle_operation::boris<boris_e_gradient>(p, e, b, 1.0, 0.01);
};
// 3d E on 1d grid (truncated) + B 3d QS=3
inline constexpr auto pos13 = [](P3& p, const acc_1D1C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 3d axisym E direct + cartesian B QS=3 (mixed maps)
inline constexpr auto pos14 = [](P3& p, const acc_2D2C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_axisym_direct>(p, e, b, 1.0, 0.01);
};
// 2d auto E, scalar field -> phi
inline constexpr auto pos15 = [](P2& p, const acc_2D1C& e) {
    particle_operation::leapfrog<e_auto>(p, e, 1.0, 0.01);
};
// 2d auto E, 2-component field -> direct
inline constexpr auto pos16 = [](P2& p, const acc_2D2C& e) {
    particle_operation::leapfrog<e_auto>(p, e, 1.0, 0.01);
};
// 3d partial E QS=2 (Ex,Ey) + B 3d QS=3
inline constexpr auto pos17 = [](P3& p, const acc_3D2C& e, const acc_3D3C& b) {
    particle_operation::boris<boris_e_direct>(p, e, b, 1.0, 0.01);
};
// 3d B QS=1 prefix + E 3d direct, order 4 both
inline constexpr auto pos18 = [](P3& p, const acc_3D1C& e, const acc_3D1C& b) {
    particle_operation::boris<boris_order4>(p, e, b, 1.0, 0.01);
};
// 3d storage, 1d phi grid, gradient E
inline constexpr auto pos19 = [](P3& p, const acc_1D1C& e) {
    particle_operation::leapfrog<e_gradient>(p, e, 1.0, 0.01);
};
// free flight on all three particle dims
inline constexpr auto pos20 = [](P1& p) { particle_operation::drift(p, 0.01); };
inline constexpr auto pos21 = [](P2& p) { particle_operation::drift(p, 0.01); };
inline constexpr auto pos22 = [](P3& p) { particle_operation::drift(p, 0.01); };


// ---------- one-step drivers ----------
template <typename F>
void zero(F& f) {
    f.setZero();
}
P1 mk1(double x) {
    P1 p;
    get<position>(p).x() = x;
    get<velocity>(p).x() = 0;
    return p;
}
P2 mk2(double x, double y) {
    P2 p;
    get<position>(p) = {x, y};
    get<velocity>(p) = {0, 0};
    return p;
}
P3 mk3(double x, double y, double z) {
    P3 p;
    get<position>(p) = {x, y, z};
    get<velocity>(p) = {0, 0, 0};
    return p;
}
template <typename P>
void load(PG<P>& pg, sycl::queue& q, P p) {
    device_vector<P> dv(q, {p});
    pg.insert(dv);
}
template <typename Probe, typename Group>
void drive(Probe probe, Group& pg) {
    pg.for_each([&](sycl::handler& h) {
        return [=](typename std::remove_cvref_t<Group>::value_type& p) {
            probe(p);
        };
    });
}
template <typename Probe, typename Group, typename EField>
void drive(Probe probe, Group& pg, EField& e) {
    pg.for_each([&](sycl::handler& h) {
        auto e_acc = e.get_access(h);
        return [=](typename std::remove_cvref_t<Group>::value_type& p) {
            probe(p, e_acc);
        };
    });
}
template <typename Probe, typename Group, typename EField, typename BField>
void drive(Probe probe, Group& pg, EField& e, BField& b) {
    pg.for_each([&](sycl::handler& h) {
        auto e_acc = e.get_access(h);
        auto b_acc = b.get_access(h);
        return [=](typename std::remove_cvref_t<Group>::value_type& p) {
            probe(p, e_acc, b_acc);
        };
    });
}

int main() {
    sycl::queue q{sycl::default_selector()};

    grid1D g1({0.0}, {10.0}, {16});
    grid2D gc2({0.0, 0.0}, {10.0, 10.0}, {16, 16});
    grid2D gzr({0.0, 0.0}, {10.0, 8.0}, {16, 16});
    grid3D g3({0.0, 0.0, 0.0}, {10.0, 10.0, 10.0}, {8, 8, 8});

    {   // pos1/2: 1d
        node_field1D<double> e(q, g1); zero(e);
        PG<P1> pg(q); load(pg, q, mk1(5.0));
        drive(pos1, pg, e);
        drive(pos2, pg, e);
    }
    {   // pos3/4: 2d, B on 1d and 2d grids
        node_field2D<double> e(q, gc2); zero(e);
        node_field1D<double> b1(q, g1); zero(b1);
        node_field2D<double> b2(q, gc2); zero(b2);
        PG<P2> pg(q); load(pg, q, mk2(5.0, 5.0));
        drive(pos3, pg, e, b1);
        drive(pos4, pg, e, b2);
    }
    {   // pos5/6: 3d axisym E + axisym B (z, r) = (5, 5)
        node_field2D<double, 2> e5(q, gzr); zero(e5);
        node_field2D<double> e6(q, gzr); zero(e6);
        node_field2D<double, 2> b(q, gzr); zero(b);
        PG<P3> pg(q); load(pg, q, mk3(3.0, 4.0, 5.0));
        drive(pos5, pg, e5, b);
        drive(pos6, pg, e6, b);
    }
    {   // pos7/8: 3d cartesian phi E + axisym B QS=2/3
        node_field3D<double> e(q, g3); zero(e);
        node_field2D<double, 2> b2(q, gzr); zero(b2);
        node_field2D<double, 3> b3(q, gzr); zero(b3);
        PG<P3> pg(q); load(pg, q, mk3(3.0, 4.0, 5.0));
        drive(pos7, pg, e, b2);
        drive(pos8, pg, e, b3);
    }
    {   // pos9/10/11: 3d cartesian E/B in QS=1/3 mixes
        node_field2D<double, 3> e9(q, gc2); zero(e9);
        node_field3D<double, 3> e10(q, g3); zero(e10);
        node_field3D<double> e11(q, g3); zero(e11);
        node_field3D<double, 3> b(q, g3); zero(b);
        PG<P3> pg(q); load(pg, q, mk3(5.0, 5.0, 5.0));
        drive(pos9, pg, e9, b);
        drive(pos10, pg, e10, b);
        drive(pos11, pg, e11, b);
    }
    {   // pos12: 2d phi E + scalar B (kernel A)
        node_field2D<double> e(q, gc2); zero(e);
        node_field2D<double> b(q, gc2); zero(b);
        PG<P2> pg(q); load(pg, q, mk2(5.0, 5.0));
        drive(pos12, pg, e, b);
    }
    {   // pos13/14: truncated 1d-grid E; axisym E + cartesian B
        node_field1D<double> e13(q, g1); zero(e13);
        node_field3D<double, 3> b3(q, g3); zero(b3);
        node_field2D<double, 2> e14(q, gzr); zero(e14);
        PG<P3> pg(q); load(pg, q, mk3(5.0, 3.0, 4.0));
        drive(pos13, pg, e13, b3);
        PG<P3> pg2(q); load(pg2, q, mk3(3.0, 4.0, 5.0));
        drive(pos14, pg2, e14, b3);
    }
    {   // pos15/16: 2d auto
        node_field2D<double> e15(q, gc2); zero(e15);
        node_field2D<double, 2> e16(q, gc2); zero(e16);
        PG<P2> pg(q); load(pg, q, mk2(5.0, 5.0));
        drive(pos15, pg, e15);
        drive(pos16, pg, e16);
    }
    {   // pos17/18: partial E QS=2; order 4 both
        node_field3D<double, 2> e17(q, g3); zero(e17);
        node_field3D<double, 3> b(q, g3); zero(b);
        node_field3D<double> e18(q, g3); zero(e18);
        PG<P3> pg(q); load(pg, q, mk3(5.0, 5.0, 5.0));
        drive(pos17, pg, e17, b);
        drive(pos18, pg, e18, e18);
    }
    {   // pos19: CCP shape, 3d storage + 1d phi grid
        node_field1D<double> e(q, g1); zero(e);
        PG<P3> pg(q); load(pg, q, mk3(5.0, 3.0, 4.0));
        drive(pos19, pg, e);
    }
    {   // pos20/21/22: free flight, all particle dims
        PG<P1> pg1(q); load(pg1, q, mk1(5.0));
        PG<P2> pg2(q); load(pg2, q, mk2(5.0, 5.0));
        PG<P3> pg3(q); load(pg3, q, mk3(5.0, 5.0, 5.0));
        drive(pos20, pg1);
        drive(pos21, pg2);
        drive(pos22, pg3);
    }

    std::cout << "positive: 22 probes executed\n";
    return 0;
}
