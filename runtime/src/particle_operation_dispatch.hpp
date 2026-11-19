#ifndef PSUM_RUNTIME_PARTICLE_OPERATION_DISPATCH_HPP
#define PSUM_RUNTIME_PARTICLE_OPERATION_DISPATCH_HPP

// Strategy-id dispatch for the particle operation family.
// The switch arms binding a runtime id to the push templates live HERE, 
// E id bits: 2 = direct (else gradient), 1 = axisymmetric (else cartesian); 
// B id bit 1 = axisymmetric.
// The orders come from the universe.

#include <psum/psum.hpp>
#include "particle_operation.hpp"

namespace psum_runtime {

namespace particle_operation_dispatch {

    using namespace psum::prelude;

    template <int Id>
    using e_calc_at = std::conditional_t<(Id & 2) != 0, eCalcDirect, eCalcGradient>;
    template <int Id>
    using e_map_at = std::conditional_t<(Id & 1) != 0, axisymmetricMap, cartesianMap>;
    template <int Id, int EOrder>
    using e_strategy_at = e_strategy<e_calc_at<Id>, e_map_at<Id>, EOrder>;
    template <int Id>
    using b_map_at = std::conditional_t<(Id & 1) != 0, axisymmetricMap, cartesianMap>;
    template <int Id, int BOrder>
    using b_strategy_at = b_strategy<b_map_at<Id>, BOrder>;

    // legality of one id against (particle dim, field dim, field QS);
    // mirrors the static_asserts of get_E/get_B in particle_operation.hpp
    template <int Id, int PosDim, int GridDim, int QuantitySize>
    consteval bool e_combo_ok() {
        if constexpr (PosDim < 1 || GridDim < 1 || QuantitySize < 1) return false;
        else if constexpr (std::is_same_v<e_map_at<Id>, axisymmetricMap>)
            return PosDim == 3 && GridDim == 2 &&
                    (std::is_same_v<e_calc_at<Id>, eCalcDirect> ? QuantitySize <= 3 : QuantitySize == 1);
        else
            return GridDim <= PosDim &&
                    (std::is_same_v<e_calc_at<Id>, eCalcDirect> ? QuantitySize <= PosDim : QuantitySize == 1);
    }
    template <int Id, int PosDim, int GridDim, int QuantitySize>
    consteval bool b_combo_ok() {
        if constexpr (PosDim == 2)
            return QuantitySize == 1 && GridDim <= 2;
        else if constexpr (PosDim == 3)
            return std::is_same_v<b_map_at<Id>, axisymmetricMap> ? GridDim == 2 : GridDim <= 3;
        else
            return false;
    }
    // deposit follows the E strategy mapping onto a scalar field
    template <int Id, int PosDim, int GridDim, int QuantitySize>
    consteval bool d_combo_ok() {
        if constexpr (QuantitySize != 1) return false;
        else if constexpr (std::is_same_v<e_map_at<Id>, axisymmetricMap>)
            return PosDim == 3 && GridDim == 2;
        else
            return GridDim <= PosDim;
    }

    template <typename Universe, int Dim>
    using particle_t = typename Universe::template particle<Dim>;

    // deposit gate, evaluated before the deposit call.
    // Cartesian + periodic universe — wrap the coordinate modulo span+del per grid dimension 
    // into the half-cell extended domain [lo-del/2, hi+del/2);
    // Otherwise (axisymmetric map, or a non-periodic universe) — test against the plain deposit 
    // grid, closed bounds; a particle outside is invalid and skips the deposit.
    // Everything the in-kernel gate needs is computed outside the kernel and passed in as the following struct.
    template <int GDim>
    struct ingrid_bounds {
        std::array<double, GDim> lo, hi, a, end, period;
    };

    template <typename Grid>
    auto bounds_of(const Grid& grid) {
        constexpr int GDim = Grid::Dimension;
        ingrid_bounds<GDim> b;
        [&]<size_t... Is>(std::index_sequence<Is...>) {
            ((b.lo[Is] = grid.template lowerBound<Is>(),
                b.hi[Is] = grid.template upperBound<Is>(),
                b.a[Is] = grid.template lowerBound<Is>() -
                        grid.template del<Is>() * 0.5,
                b.period[Is] = (grid.template upperBound<Is>() -
                                grid.template lowerBound<Is>()) +
                                grid.template del<Is>(),
                b.end[Is] = b.a[Is] + b.period[Is]),
                ...);
        }(std::make_index_sequence<GDim>{});
        return b;
    }

    // the departure log for a group: a device_vector<Particle> under
    // "<group>_leave_buffer". Every kernel that may make_invalid
    // refreshes it before launch (content of the previous launch is
    // dropped) and doomed particles are push_back'd BEFORE they are
    // invalidated.
    template <typename Universe, int Dim, typename Group>
    psum::particle_container::device_vector<typename Group::value_type>&
    leave_buffer_of(serialization::object_manager& mng, Group& grp, const char* group_name) {
        using part_t = typename Group::value_type;
        using vec_t = psum::particle_container::device_vector<typename Group::value_type>;
        auto& lv = mng.template obj<vec_t>(std::string(group_name) + "_leave_buffer");
        lv.reserve(std::max(grp.size() / 5, (size_t)1024));
        lv.clear();   // the previous launch's content is discarded
        return lv;
    }

    template <typename Universe, int Dim>
    using leave_acc_t = psum::particle_container::device_vector_acc<
        typename Universe::template group<Dim>::value_type>;

    // true = may deposit; false = was invalidated (and must be skipped).
    template <int Id, typename Universe, int Dim, int GDim>
    bool deposit_allowed(typename Universe::template group<Dim>::value_type& p,
                            const ingrid_bounds<GDim>& b,
                            const leave_acc_t<Universe, Dim>* leave) {
        constexpr bool axi = std::is_same_v<e_map_at<Id>, axisymmetricMap>;
        static_assert(!axi || GDim == 2, "deposit_allowed: the axisymmetric map is 2d");
        auto& x = psum::tag::get<pt::position>(p);
        Eigen::Vector<double, GDim> xc = [&]() {
            if constexpr (axi) return particle_operation_detail::axisym_coords(x);
            else return particle_operation_detail::prefix_coords<GDim>(x);
        }();
        bool inside = true;
        for (int i = 0; i < GDim; i++) {
            if constexpr (Universe::periodic_grid && !axi) {
                while (xc[i] < b.a[i]) xc[i] += b.period[i];
                while (xc[i] >= b.end[i]) xc[i] -= b.period[i];
                x[i] = xc[i];
            } else {
                inside = inside && xc[i] >= b.lo[i] && xc[i] <= b.hi[i];
            }
        }
        if (!inside) {
            if (leave) leave->push_back(p);   // record, then invalidate
            Universe::template group<Dim>::validator::make_invalid(p);
        }
        return inside;
    }

    // a dummy deposit field for the drift kernel when no deposit is requested; no real deposit is done with it!
    template <int Dim>
    psum::field::device_field<Dim, psum::field::var_loc::nodeCentered, double, 1>&
    dummy_dep(serialization::object_manager& mng, sycl::queue q) {
        using dep_t = psum::field::device_field<Dim, psum::field::var_loc::nodeCentered, double, 1>;
        using grid_t = simple_grid<Dim>;
        if (!mng.template contains<dep_t>("__dummy_dep")) {
            std::vector<double> lo(Dim, 0.0), hi(Dim, 1.0);
            std::vector<int> nc(Dim, 1);
            mng.template obj<grid_t>("__dummy_dep_grid") = grid_t(lo, hi, nc);
            mng.template obj<dep_t>("__dummy_dep") = dep_t(q, mng.template obj<grid_t>("__dummy_dep_grid"));
        }
        return mng.template obj<dep_t>("__dummy_dep");
    }


    // ---------- drift: one family with leapfrog/boris, no field read.
    // With a deposit field it is the shared arm shape; without one the
    // plain x += v dt kernel runs (no gate: there is no grid to test).
    template <typename Universe, int Dim, typename Group>
    int drift_one(serialization::object_manager& mng, const char* group_name,
                    Group& grp,
                    int e_id, double dt,
                    const char* deposit_name, double deposit_weight) {
        using part_t = typename Group::value_type;
        auto& lv = leave_buffer_of<Universe, Dim>(mng, grp, group_name);
        auto run = [&](auto& df, bool want_dep) {
            using DF = std::remove_cvref_t<decltype(df)>;
            constexpr int GDim = DF::Dimension;
            const auto bnd = bounds_of(df.getGrid());
            switch (e_id) {
#define PSUM_PD_DRIFT_ARM(i)                                                      \
    case i:                                                                       \
        if constexpr (d_combo_ok<i, Dim, GDim, 1>()) {                            \
            grp.for_each([&](sycl::handler& h) {                                  \
                auto acc = df.get_access(h);                                      \
                auto lv_acc = lv.get_access(h);                                   \
                return [=](part_t& p) {                                           \
                    particle_operation::drift(p, dt);                             \
                    if (want_dep &&                                               \
                        deposit_allowed<i, Universe, Dim>(p, bnd, &lv_acc))       \
                        particle_operation::deposit<e_strategy_at<i, Universe::e_order>>( \
                            p, acc, deposit_weight);                              \
                };                                                                \
            });                                                                   \
            return 0;                                                             \
        } else                                                                    \
            return -2;
                PSUM_PD_DRIFT_ARM(0) PSUM_PD_DRIFT_ARM(1)
                PSUM_PD_DRIFT_ARM(2) PSUM_PD_DRIFT_ARM(3)
#undef PSUM_PD_DRIFT_ARM
            }
            return -2;
        };
        if (deposit_name) {
            auto on_gdim = [&](auto gdim_constant) {
                constexpr int GDim = decltype(gdim_constant)::value;
                using dep_field_t = psum::field::device_field<
                    GDim, psum::field::var_loc::nodeCentered, double, 1>;
                if (!mng.template contains<dep_field_t>(deposit_name))
                    return -3;   // the named field is not this dim
                auto& df = mng.template obj<dep_field_t>(deposit_name);
                if (!df.has_grid()) return -1;
                return run(df, true);
            };
            int r = on_gdim(std::integral_constant<int, 1>{});
            if (r != -3) return r;
            r = on_gdim(std::integral_constant<int, 2>{});
            if (r != -3) return r;
            return on_gdim(std::integral_constant<int, 3>{});
        }
        auto& dum = dummy_dep<Dim>(mng, grp.get_content().get_queue());
        return run(dum, false);
    }

    // ---------- batch append + deposit: candidates are gated first —
    // survivors are deposited and appended with a single insert;
    // rejects are NOT recorded in the leave buffer (they never entered
    // the group; the caller holds the data and the accepted count)
    template <typename Universe, int Dim, typename Group, typename DField>
    int inject_batch(serialization::object_manager& mng, const char* group_name,
                        Group& grp,
                        DField& f, int e_id, size_t count,
                        const double* pos, const double* vel,
                        double weight, size_t* accepted) {
        using part_t = typename Group::value_type;
        if (!f.has_grid() || !pos || !vel) return -1;
        const auto bnd = bounds_of(f.getGrid());
        sycl::queue q = f.getQueue();
        std::vector<part_t> cand;
        cand.reserve(count);
        for (size_t k = 0; k < count; k++) {
            part_t p;
            for (int i = 0; i < Dim; i++) {
                psum::tag::get<pt::position>(p)[i] = pos[k * Dim + i];
                psum::tag::get<pt::velocity>(p)[i] = vel[k * Dim + i];
            }
            cand.push_back(std::move(p));
        }
        using vec_t = psum::particle_container::device_vector<part_t>;
        vec_t cvec(q, cand);
        vec_t keep(q, std::max<size_t>(count, 1));
        switch (e_id) {
#define PSUM_PD_INJECT_ARM(i)                                                      \
    case i:                                                                        \
        if constexpr (d_combo_ok<i, Dim, DField::Dimension, DField::QuantitySize>()) { \
            q.submit([&](sycl::handler& h) {                                       \
                auto c_acc = cvec.get_access(h);                                   \
                auto f_acc = f.get_access(h);                                      \
                auto k_acc = keep.get_access(h);                                   \
                h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {       \
                    part_t p = c_acc[idx];                      \
                    if (deposit_allowed<i, Universe, Dim>(p, bnd, nullptr)) {      \
                        k_acc.push_back(p);                                        \
                        particle_operation::deposit<e_strategy_at<i, Universe::e_order>>(        \
                            p, f_acc, weight);                                     \
                    }                                                              \
                });                                                                \
            }).wait();                                                             \
            if (accepted) *accepted = keep.size();                                 \
            grp.template insert<false>(keep);                                      \
            return 0;                                                              \
        } else                                                                     \
        return -2;
            PSUM_PD_INJECT_ARM(0) PSUM_PD_INJECT_ARM(1)
            PSUM_PD_INJECT_ARM(2) PSUM_PD_INJECT_ARM(3)
#undef PSUM_PD_INJECT_ARM
        }
        return -2;
    }

#define PSUM_PD_E_ARM(i)                                                                    \
    case i:                                                                                 \
        if constexpr (e_combo_ok<i, Dim, EField::Dimension, EField::QuantitySize>()) {      \
            using dep_field_t = psum::field::device_field<                                  \
                EField::Dimension, psum::field::var_loc::nodeCentered, double, 1>;          \
            /* runtime fuse: deposit always compiled in; entered only when a                \
               valid deposit field is passed. Without one the ingrid gate                   \
               uses the PASSED E-field grid (and nothing is deposited). */                  \
            dep_field_t* dfp = nullptr;                                                     \
            if (deposit_name) {                                                             \
                if (!mng.template contains<dep_field_t>(deposit_name))                      \
                    return -1;                                                              \
                dfp = &mng.template obj<dep_field_t>(deposit_name);                         \
                if (!dfp->has_grid()) return -1;                                            \
            } else if (!f.has_grid()) {                                                     \
                return -1;                                                                  \
            }                                                                               \
            const bool want_dep = dfp != nullptr;                                           \
            dep_field_t& dref = want_dep ? *dfp : dummy_dep<EField::Dimension>(mng, f.getQueue());    \
            const auto bnd = bounds_of(want_dep ? dref.getGrid() : f.getGrid());            \
            auto& lv = leave_buffer_of<Universe, Dim>(mng, grp, group_name);                \
            grp.for_each([&](sycl::handler& h) {                                            \
                auto acc = f.get_access(h);                                                 \
                auto d_acc = dref.get_access(h);                                            \
                auto lv_acc = lv.get_access(h);                                             \
                return [=](part_t& p) {                                  \
                    particle_operation::leapfrog<e_strategy_at<i, Universe::e_order>>(                    \
                        p, acc, q_m, dt);                                                   \
                    if (deposit_allowed<i, Universe, Dim>(p, bnd, &lv_acc) && want_dep)          \
                        particle_operation::deposit<e_strategy_at<i, Universe::e_order>>(                 \
                            p, d_acc, deposit_weight);                                      \
                };                                                                          \
            });                                                                             \
            return 0;                                                                       \
        } else                                                                              \
            return -2;


    // ---------- leapfrog: E-only kick + drift, runtime-fused deposit
    template <typename Universe, int Dim, typename Group, typename EField>
    int leapfrog_one(serialization::object_manager& mng, const char* group_name,
                        Group& grp,
                        EField& f, int e_id, double q_m, double dt,
                        const char* deposit_name, double deposit_weight) {
        using part_t = typename Group::value_type;
        switch (e_id) {
            PSUM_PD_E_ARM(0) PSUM_PD_E_ARM(1)
            PSUM_PD_E_ARM(2) PSUM_PD_E_ARM(3)
#undef PSUM_PD_E_ARM
        }
        return -2;
    }

#define PSUM_PD_D_ARM(i)                                                                    \
    case i:                                                                                 \
        if constexpr (d_combo_ok<i, Dim, DField::Dimension, DField::QuantitySize>()) {      \
            if (!f.has_grid()) return -1;                                                   \
            const auto bnd = bounds_of(f.getGrid());                                        \
            auto& lv = leave_buffer_of<Universe, Dim>(mng, grp, group_name);               \
            grp.for_each([&](sycl::handler& h) {                                            \
                auto acc = f.get_access(h);                                                 \
                auto lv_acc = lv.get_access(h);                                            \
                return [=](part_t& p) {                                  \
                    if (deposit_allowed<i, Universe, Dim>(p, bnd, &lv_acc))                      \
                        particle_operation::deposit<e_strategy_at<i, Universe::e_order>>(                 \
                            p, acc, weight_charge);                                         \
                };                                                                          \
            });                                                                             \
            return 0;                                                                       \
        } else                                                                              \
            return -2;

    // ---------- standalone deposit onto a scalar field
    template <typename Universe, int Dim, typename Group, typename DField>
    int deposit_one(serialization::object_manager& mng, const char* group_name,
                    Group& grp,
                    DField& f, int e_id, double weight_charge) {
        using part_t = typename Group::value_type;
        switch (e_id) {
            PSUM_PD_D_ARM(0) PSUM_PD_D_ARM(1)
            PSUM_PD_D_ARM(2) PSUM_PD_D_ARM(3)
#undef PSUM_PD_D_ARM
        }
        return -2;
    }

#define PSUM_PD_E2B_ARM(i)                                                                  \
    case i:                                                                                 \
        if constexpr (e_combo_ok<i, Dim, EField::Dimension, EField::QuantitySize>())        \
            return boris_b_part<i, Universe, Dim, Group>(mng, group_name, grp, ef, bf, b_id,           \
                                   q_m, dt, deposit_name, deposit_weight);                  \
        else                                                                                \
            return -2;

    template <int e_id_constant, typename Universe, int Dim, typename Group,
                typename EField, typename BField>
    int boris_b_part(serialization::object_manager& mng, const char* group_name,
                        Group& grp,
                        EField& ef, BField& bf, int b_id, double q_m, double dt,
                        const char* deposit_name, double deposit_weight);

    // ---------- boris: E + B rotation push, runtime-fused deposit
    template <typename Universe, int Dim, typename Group, typename EField, typename BField>
    int boris_one(serialization::object_manager& mng, const char* group_name,
                    Group& grp,
                    EField& ef, BField& bf, int e_id, int b_id,
                    double q_m, double dt,
                    const char* deposit_name, double deposit_weight) {
        using part_t = typename Group::value_type;
        switch (e_id) {
            PSUM_PD_E2B_ARM(0) PSUM_PD_E2B_ARM(1)
            PSUM_PD_E2B_ARM(2) PSUM_PD_E2B_ARM(3)
#undef PSUM_PD_E2B_ARM
        }
        return -2;
    }

#define PSUM_PD_B_ARM(i)                                                                    \
    case i:                                                                                 \
        if constexpr (b_combo_ok<i, Dim, BField::Dimension, BField::QuantitySize>()) {      \
            using dep_field_t = psum::field::device_field<                                  \
                EField::Dimension, psum::field::var_loc::nodeCentered, double, 1>;          \
            /* runtime fuse: as in the E arm; without a deposit field the                   \
               gate uses the E-field grid */                                                \
            dep_field_t* dfp = nullptr;                                                     \
            if (deposit_name) {                                                             \
                if (!mng.template contains<dep_field_t>(deposit_name))                      \
                    return -1;                                                              \
                dfp = &mng.template obj<dep_field_t>(deposit_name);                         \
                if (!dfp->has_grid()) return -1;                                            \
            } else if (!ef.has_grid()) {                                                    \
                return -1;                                                                  \
            }                                                                               \
            const bool want_dep = dfp != nullptr;                                           \
            dep_field_t& dref = want_dep ? *dfp                                             \
                                         : dummy_dep<EField::Dimension>(mng, ef.getQueue());\
            const auto bnd = bounds_of(want_dep ? dref.getGrid() : ef.getGrid());           \
            auto& lv = leave_buffer_of<Universe, Dim>(mng, grp, group_name);                \
            grp.for_each([&](sycl::handler& h) {                                            \
                auto e_acc = ef.get_access(h);                                              \
                auto b_acc = bf.get_access(h);                                              \
                auto d_acc = dref.get_access(h);                                            \
                auto lv_acc = lv.get_access(h);                                             \
                return [=](part_t& p) {                                  \
                    particle_operation::boris<push_strategy<                                              \
                        e_strategy_at<e_id_constant, Universe::e_order>,                    \
                        b_strategy_at<i, Universe::b_order>>>(                              \
                        p, e_acc, b_acc, q_m, dt);                                          \
                    if (deposit_allowed<e_id_constant, Universe, Dim>(p, bnd, &lv_acc) &&        \
                        want_dep)                                                           \
                        particle_operation::deposit<e_strategy_at<e_id_constant,                          \
                                                 Universe::e_order>>(                      \
                            p, d_acc, deposit_weight);                                      \
                };                                                                          \
            });                                                                             \
            return 0;                                                                       \
        } else                                                                              \
            return -2;

    template <int e_id_constant, typename Universe, int Dim, typename Group,
                typename EField, typename BField>
    int boris_b_part(serialization::object_manager& mng, const char* group_name,
                        Group& grp,
                        EField& ef, BField& bf, int b_id, double q_m, double dt,
                        const char* deposit_name, double deposit_weight) {
        using part_t = typename Group::value_type;
        switch (b_id) {
            PSUM_PD_B_ARM(0) PSUM_PD_B_ARM(1)
#undef PSUM_PD_B_ARM
        }
        return -2;
    }

} // namespace particle_operation_dispatch

} // namespace psum_runtime

#endif
