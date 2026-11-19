#ifndef PSUM_RUNTIME_PARTICLE_OPERATION_HPP
#define PSUM_RUNTIME_PARTICLE_OPERATION_HPP

// Generic particle operation kernels (push + deposit) 

#include <array>
#include <Eigen/Core>
#include <sycl/sycl.hpp>

#include "tag/property.hpp"
#include "particle_container/particle_group.hpp"
#include "field/higher_order_interp.hpp"

namespace psum_runtime {

    namespace pt = psum::property;

    // ---------- policies ----------
    struct cartesianMap {};     // default mapping
    struct axisymmetricMap {};
    struct eCalcDirect {};      // E strategy: interpolate an E field
    struct eCalcGradient {};        // E strategy: E = -grad(phi)
    struct eCalcAuto {};       // unset: adjudicated from the particle dim, E grid dim and E component count

    namespace particle_operation_detail {

        // explicit policies win; auto resolves per the rules above; if both
        // predicates are false the combination is rejected
        template <typename ECalc, int PosDim, int GridDim, int QuantitySize>
        consteval bool e_calc_is_direct() {
            if constexpr (std::is_same_v<ECalc, eCalcDirect>) return true;
            else if constexpr (std::is_same_v<ECalc, eCalcGradient>) return false;
            else return PosDim == GridDim && PosDim != 1 && QuantitySize == PosDim;
        }
        template <typename ECalc, int PosDim, int GridDim, int QuantitySize>
        consteval bool e_calc_is_gradient() {
            if constexpr (std::is_same_v<ECalc, eCalcGradient>) return true;
            else if constexpr (std::is_same_v<ECalc, eCalcDirect>) return false;
            else return PosDim == GridDim && PosDim != 1 && QuantitySize == 1;
        }

        // dimension of a position/velocity type (property-constrained Eigen
        // vectors; std::array tolerated at the interp level)
        template <typename T> struct pos_dim_of;
        // any floating scalar counts as one dimension (float / double /
        // long double, following the field scalar type)
        template <typename Scalar>
            requires std::is_floating_point_v<Scalar>
        struct pos_dim_of<Scalar> : std::integral_constant<int, 1> {};
        template <typename Scalar, std::size_t N>
        struct pos_dim_of<std::array<Scalar, N>> : std::integral_constant<int, int(N)> {};
        template <typename Scalar, int R, int C>
        struct pos_dim_of<Eigen::Matrix<Scalar, R, C>>
            : std::integral_constant<int, (R > C ? R : C)> {};
        template <typename T>
        constexpr int pos_dim = pos_dim_of<std::remove_cvref_t<T>>::value;

        template <typename P>
        decltype(auto) at(P& x, int i) { return (x[i]); }
        template <typename P>
        decltype(auto) at(const P& x, int i) { return (x[i]); }

        // writes all OutSize slots explicitly: the first InSize components
        // come from the field value, the remaining ones are set to zero
        template <int InSize, int OutSize, typename V, typename Out>
            requires (InSize >= 1 && InSize <= OutSize)
        void fill_prefix(const V& v, Out* out) {
            if constexpr (InSize == 1)
                out[0] = v;
            else
                for (int i = 0; i < InSize; i++) out[i] = v[i];
            if constexpr (InSize < OutSize)
                for (int i = InSize; i < OutSize; i++) out[i] = 0.0;
        }

        // first GridDim coordinates of x; the remaining coordinates are
        // dropped (explicit truncation, no padding needed or done here)
        template <int GridDim, typename Position> requires (GridDim <= pos_dim<Position>)
        Eigen::Vector<double, GridDim> prefix_coords(const Position& x) {
            Eigen::Vector<double, GridDim> xp;
            for (int i = 0; i < GridDim; i++) xp[i] = at(x, i);
            return xp;
        }

        // cartesian gather: interp / interp-diff at the truncated position;
        // axisymmetric gather: (z, r) coords + (z, r, t) -> Cartesian theta
        template <typename Position> requires (pos_dim<Position> == 3)
        Eigen::Vector<double, 2> axisym_coords(const Position& x) {
            return {at(x, 2), sycl::sqrt(at(x, 0) * at(x, 0) + at(x, 1) * at(x, 1))};
        }

        // writes all three out slots explicitly; InSize == 1 means the
        // value is the t component
        template <int InSize, typename V, typename Position, typename Out>
            requires ((InSize == 1 || InSize == 2 || InSize == 3) && pos_dim<Position> == 3 && InSize == pos_dim<V>)
        void axisym_to_cart(const V& v, const Position& x, Out* out3) {
            double r = sycl::sqrt(at(x, 0) * at(x, 0) + at(x, 1) * at(x, 1));
            double ct = at(x, 0) / r, st = at(x, 1) / r;
            if constexpr (InSize == 1) {
                out3[0] = -v * st;
                out3[1] = v * ct;
                out3[2] = 0.0;
            } else if constexpr (InSize == 2) {
                out3[0] = v[1] * ct;
                out3[1] = v[1] * st;
                out3[2] = v[0];
            } else if constexpr (InSize == 3) { 
                out3[0] = v[1] * ct - v[2] * st;
                out3[1] = v[1] * st + v[2] * ct;
                out3[2] = v[0];
            }
        }

        // strategy-dispatched gathers; the returned vector length follows the
        // combination: E has the particle dimension (zero-padded prefix or
        // axisym-converted), B is 3 for kernel B and 1 (the scalar Bz) on 2d
        template <typename Scalar, typename Position>
        using e_vector = Eigen::Vector<Scalar, pos_dim<Position>>;
        template <typename Scalar, typename Position>
        using b_vector = Eigen::Vector<Scalar, pos_dim<Position> == 2 ? 1 : 3>;

        template <typename EPart, typename Scalar, typename FieldAcc, typename Position>
        inline e_vector<Scalar, Position> get_E(const Position& x, const FieldAcc& acc) {
            constexpr bool as_gradient =
                e_calc_is_gradient<typename EPart::e_calc, pos_dim<Position>, FieldAcc::Dimension, FieldAcc::QuantitySize>();
            constexpr bool as_direct =
                e_calc_is_direct<typename EPart::e_calc, pos_dim<Position>, FieldAcc::Dimension, FieldAcc::QuantitySize>();
            static_assert(as_gradient || as_direct,
                          "E calc unresolved: Electric field interpolation strategy cannot be adjudicated, please specify a strategy explicitly");
            e_vector<Scalar, Position> E; // every slot written below
            if constexpr (std::is_same_v<typename EPart::e_map, axisymmetricMap>) {
                static_assert(FieldAcc::Dimension == 2, "E calc: axisymmetric map requires a 2d field");
                if constexpr (as_gradient) {
                    static_assert(FieldAcc::QuantitySize == 1, "E calc: axisymmetric gradient requires a scalar field");
                    auto g = psum::field::interp_diff_B_spline<EPart::order>(axisym_coords(x), acc);
                    Eigen::RowVector<Scalar, 2> c(-g[0], -g[1]);
                    axisym_to_cart<2>(c, x, E.data());
                } else {
                    auto v = psum::field::interp_B_spline<EPart::order>(axisym_coords(x), acc);
                    axisym_to_cart<FieldAcc::QuantitySize>(v, x, E.data());
                }
            } else {
                if constexpr (as_gradient) {
                    static_assert(FieldAcc::QuantitySize == 1, "E calc: direct gradient requires a scalar field");
                    auto g = psum::field::interp_diff_B_spline<EPart::order>(prefix_coords<FieldAcc::Dimension>(x), acc);
                    for (int i = 0; i < pos_dim<Position>; i++)
                        E[i] = i < FieldAcc::Dimension ? -g[i] : 0.0;
                } else {
                    auto v = psum::field::interp_B_spline<EPart::order>(prefix_coords<FieldAcc::Dimension>(x), acc);
                    fill_prefix<FieldAcc::QuantitySize, pos_dim<Position>>(v, E.data());
                }
            }
            return E;
        }

        template <typename BPart, typename Scalar, typename FieldAcc, typename Position>
        inline b_vector<Scalar, Position> get_B(const Position& x, const FieldAcc& acc) {
            static_assert(pos_dim<Position> == 2 || pos_dim<Position> == 3, "B calc: the boris kernels act on 2d or 3d particles");
            b_vector<Scalar, Position> B;   // every slot written below
            if constexpr (pos_dim<Position> == 2) {
                static_assert(FieldAcc::QuantitySize == 1, "B calc: 2d particles require a scalar B field");
                B[0] = psum::field::interp_B_spline<BPart::order>(prefix_coords<FieldAcc::Dimension>(x), acc);
            } else if constexpr (std::is_same_v<typename BPart::b_map, axisymmetricMap>) {
                static_assert(FieldAcc::Dimension == 2, "B calc: axisymmetric map requires a 2d field");
                auto v = psum::field::interp_B_spline<BPart::order>(axisym_coords(x), acc);
                axisym_to_cart<FieldAcc::QuantitySize>(v, x, B.data());
            } else {
                auto v = psum::field::interp_B_spline<BPart::order>(prefix_coords<FieldAcc::Dimension>(x), acc);
                fill_prefix<FieldAcc::QuantitySize, pos_dim<Position>>(v, B.data());
            }
            return B;
        }

        // ---------- kernels ----------
        // Kernel A: 2d Boris (a = q/m * dt/2); v' = v- + v- x t, v+ = v- + v' x s.
        template <typename Scalar>
        inline void boris_2d_step(Scalar& vx, Scalar& vy,
                                Scalar Ex, Scalar Ey, Scalar Bz, Scalar a) {
            Scalar vmx = vx + a * Ex, vmy = vy + a * Ey;
            Scalar t = a * Bz, s = Scalar(2) * t / (Scalar(1) + t * t);
            Scalar vpx = vmx + vmy * t, vpy = vmy - vmx * t;
            vx = vmx + vpy * s;
            vy = vmy - vpx * s;
            vx += a * Ex; vy += a * Ey;
        }
        // Kernel B: full 3d Boris in component form.
        template <typename Scalar>
        inline void boris_3d_step(Scalar& vx, Scalar& vy, Scalar& vz,
                                Scalar Ex, Scalar Ey, Scalar Ez,
                                Scalar Bx, Scalar By, Scalar Bz, Scalar a) {
            vx += a * Ex; vy += a * Ey; vz += a * Ez;
            Scalar tx = a * Bx, ty = a * By, tz = a * Bz;
            Scalar f = Scalar(2) / (Scalar(1) + tx * tx + ty * ty + tz * tz);
            Scalar sx = tx * f, sy = ty * f, sz = tz * f;
            Scalar vpx = vx + vy * tz - vz * ty;
            Scalar vpy = vy + vz * tx - vx * tz;
            Scalar vpz = vz + vx * ty - vy * tx;
            vx += vpy * sz - vpz * sy;
            vy += vpz * sx - vpx * sz;
            vz += vpx * sy - vpy * sx;
            vx += a * Ex; vy += a * Ey; vz += a * Ez;
        }
        // vector-argument adapters (kernel A reads E.xy and the scalar Bz vector)
        template <typename Vel, typename EVec, typename BVec, typename Coef>
        inline void boris_2d_step(Vel& v, const EVec& E, const BVec& B, Coef a) {
            boris_2d_step(v[0], v[1], E[0], E[1], B[0], a);
        }
        template <typename Vel, typename EVec, typename BVec, typename Coef>
        inline void boris_3d_step(Vel& v, const EVec& E, const BVec& B, Coef a) {
            boris_3d_step(v[0], v[1], v[2], E[0], E[1], E[2], B[0], B[1], B[2], a);
        }
        // x[i] += v[i] * dt for i < pos_dim; demands matching dimensions and
        // per-component addability (x[i] += v[i] * dt must be well-formed)
        template <int PosDim, typename Vel, typename Pos>
            requires (PosDim == pos_dim<Vel> &&
                    PosDim == pos_dim<Pos> &&
                    requires(const Vel& v, Pos& x) { x[0] += v[0] * double{}; })
        inline void position_update(const Vel& v, Pos& x, auto dt) {
            for (int i = 0; i < PosDim; i++)
                at(x, i) += at(v, i) * dt;
        }

    }

    // ---------- strategies ----------
    // E and B parts carry independent orders; the E part has one extra
    // (optional) policy: field-interpolation vs phi-differential.
    template <typename ECalc = eCalcAuto, typename EMap = cartesianMap, int Order = 1>
    struct e_strategy {
        using e_calc = ECalc;
        using e_map = EMap;
        static constexpr int order = Order;
    };
    template <typename BMap = cartesianMap, int Order = 1>
    struct b_strategy {
        using b_map = BMap;
        static constexpr int order = Order;
    };
    template <typename EPart = e_strategy<>, typename BPart = b_strategy<>>
    struct push_strategy {
        using e_part = EPart;
        using b_part = BPart;
    };
    using default_strategy = e_strategy<>;

    namespace particle_operation {
        // ---------- operation cores ----------
        // drift: the drift of the leapfrog split (x += v dt), the free-flight
        // sibling of leapfrog/boris below
        template <typename Particle>
        void drift(Particle& p, auto dt) {
            auto& x = psum::tag::get<pt::position>(p);
            auto& v = psum::tag::get<pt::velocity>(p);
            constexpr int PosDim = particle_operation_detail::pos_dim<decltype(x)>;
            particle_operation_detail::position_update<PosDim>(v, x, dt);
        }

        // One particle against interpolatable field accessors (the device
        // objects handed out by field.get_access(handler)).
        // Electric-only kick + drift (Euler-Cromer, the staggered leapfrog):
        // v += (q/m) E dt; x += v dt.
        template <typename EPart = default_strategy, typename Particle, typename FieldAcc>
        void leapfrog(Particle& p, const FieldAcc& e_acc, auto q_m, auto dt) {
            auto& x = psum::tag::get<pt::position>(p);
            auto& v = psum::tag::get<pt::velocity>(p);
            constexpr int PosDim = particle_operation_detail::pos_dim<decltype(x)>;
            using VelScalar = std::remove_cvref_t<decltype(v[0])>;
            auto Evec = particle_operation_detail::get_E<EPart, VelScalar>(x, e_acc);
            for (int i = 0; i < PosDim; i++)
                particle_operation_detail::at(v, i) += q_m * Evec[i] * dt;
            particle_operation_detail::position_update<PosDim>(v, x, dt);
        }

        // Boris rotation + drift; kernel selection is proven at compile time
        // (kernel A on 2d particles with the scalar Bz, kernel B otherwise).
        template <typename Strategy, typename Particle, typename FieldAcc, typename BFieldAcc>
        void boris(Particle& p, const FieldAcc& e_acc, const BFieldAcc& b_acc, auto q_m, auto dt) {
            auto& x = psum::tag::get<pt::position>(p);
            auto& v = psum::tag::get<pt::velocity>(p);
            constexpr int PosDim = particle_operation_detail::pos_dim<decltype(x)>;
            constexpr int BQuantitySize = BFieldAcc::QuantitySize;
            using VelScalar = std::remove_cvref_t<decltype(v[0])>;
            auto Evec = particle_operation_detail::get_E<typename Strategy::e_part, VelScalar>(x, e_acc);
            auto Bvec = particle_operation_detail::get_B<typename Strategy::b_part, VelScalar>(x, b_acc);
            auto a = VelScalar(q_m * dt / 2);
            if constexpr (BQuantitySize == 1 && PosDim == 2)
                particle_operation_detail::boris_2d_step(v, Evec, Bvec, a);
            else
                particle_operation_detail::boris_3d_step(v, Evec, Bvec, a);
            particle_operation_detail::position_update<PosDim>(v, x, dt);
        }


        // w lands on the nodes around x with the B-spline shape function of the
        // strategy order; accumulation semantics (add_back), no implicit zeroing
        template <typename EPart, typename Particle, typename FieldAcc, typename WeightCharge>
        void deposit(Particle& p, FieldAcc& acc, WeightCharge weight_charge) {
            static_assert(FieldAcc::QuantitySize == 1,
                          "deposit: the deposit field must be scalar");
            auto& x = psum::tag::get<pt::position>(p);
            if constexpr (std::is_same_v<typename EPart::e_map, axisymmetricMap>) {
                static_assert(FieldAcc::Dimension == 2,
                              "deposit: the axisymmetric map requires a 2d (z, r) field");
                static_assert(particle_operation_detail::pos_dim<decltype(x)> == 3,
                              "deposit: the axisymmetric map requires 3d particles");
                psum::field::add_back_B_spline<EPart::order>(particle_operation_detail::axisym_coords(x), weight_charge, acc);
            } else {
                psum::field::add_back_B_spline<EPart::order>(
                    particle_operation_detail::prefix_coords<FieldAcc::Dimension>(x), weight_charge, acc);
            }
        }

    }

}

#endif
