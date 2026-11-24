#ifndef PSUM_RUNTIME_FIELD_MANAGER_HPP
#define PSUM_RUNTIME_FIELD_MANAGER_HPP

#include <psum/psum.hpp>
#include "family_manager.hpp"
#include "families.hpp"
#include "grid_manager.hpp"
#include "desc/field_desc.hpp"
#include <cstddef>
#include <unordered_map>
#include <string>

namespace psum_runtime {

    using namespace psum::prelude;

    // field_manager is parameterized by the universe
    // Universe::enable_float_fields will determine if float fields are enabled.
    template <typename Universe>
    class field_manager {
    public:
        using status = manager_status_of<200>;   // the field segment
        using family = field_family<Universe>;

        // which window does a family member belong to?
        template <typename T>
        struct is_host_field : std::false_type {};
        template <int Dim, int QS, typename Scalar>
        struct is_host_field<host_field_t<Dim, QS, Scalar>> : std::true_type {};
        template <typename T>
        struct is_device_field : std::bool_constant<!is_host_field<T>::value> {};

        field_manager(serialization::object_manager& mng, sycl::queue& q, grid_manager<Universe>& grids, last_failure& last)
            : q_(q), grids_(grids), fields_(mng, &last, 200) {}

        // ---------------- query / access ----------------

        bool exists(const char* name) const { return fields_.exists(name); }

        template <typename F>
        status dispatch_field(const char* name, F&& fn) {
            return fields_.template dispatch_if<is_device_field>(name, status(st_missing), std::forward<F>(fn));
        }
        template <typename F>
        status dispatch_host_field(const char* name, F&& fn) {
            return fields_.template dispatch_if<is_host_field>(name, status(st_missing), std::forward<F>(fn));
        }

        // ---------------- descriptor interface ----------------

        bool desc_get(const char* name, psum_obj_field_desc* out) const {
            return fields_.desc_get(name, out);
        }
        status check(const psum_obj_field_desc& d) const { return status(fields_.check(d)); }

        status create(psum_obj_field_desc* d, const char* grid_name) {
            if (!d || !d->name || !grid_name) return st_bad_arg;
            psum_obj_grid_desc g = {};
            if (!grids_.desc_get(grid_name, &g)) return status(psum_code(PSUM_ERR_NOT_FOUND, PSUM_SEGMENT_GRID));   // the grid is the one missing
            if (d->dim != 0 && d->dim != g.dim) return st_bad_arg;
            d->dim = g.dim;                       // the grid decides the rest
            for (int i = 0; i < g.dim; i++) {
                d->lower[i] = g.lower[i];
                d->upper[i] = g.upper[i];
                d->node_counts[i] = g.node_count[i] + 1;
            }
            return status(fields_.create(d,
                // resolver handed to the protocol: the grid of the given dim
                [&](auto dim_constant) -> auto& {
                    constexpr int N = decltype(dim_constant)::value;
                    return grids_.template get<N>(grid_name);
                },
                q_));
        }

    private:
        sycl::queue& q_;
        // the object_manager is reached ONLY through these families
        grid_manager<Universe>& grids_;                 // grids belong to the grid manager
        family_manager<family> fields_;   // device AND host fields, one name space

    };

    // field desc  <->  the field family
    template <typename Universe>
    struct desc_family_protocol<psum_obj_field_desc, field_family<Universe>> {
        using protocol = void;
        using FM = field_manager<Universe>;

        // the descriptor's TYPE invariants ARE the family key
        static constexpr field_key key(const psum_obj_field_desc& d) {
            return field_key{d.dim, d.quantity_size, d.scalar_kind, d.on_device != 0};
        }

        template <typename Obj>
        static void extract(const Obj& f, psum_obj_field_desc& d) {
            constexpr int N = Obj::Dimension;
            auto& g = f.getGrid();
            d.dim = N;
            d.quantity_size = Obj::QuantitySize;
            d.scalar_kind = std::is_same_v<typename Obj::Scalar, float> ? 1 : 0;
            d.on_device = std::is_same_v<Obj, host_field_t<
                                   N, Obj::QuantitySize, typename Obj::Scalar>> ? 0 : 1;
            [&]<int... D>(std::integer_sequence<int, D...>) {
                ((d.node_counts[D] = g.template numCells<D>() + 1), ...);
                ((d.lower[D] = g.template lowerBound<D>()), ...);
                ((d.upper[D] = g.template upperBound<D>()), ...);
            }(std::make_integer_sequence<int, N>{});
        }

        static equality equal(const psum_obj_field_desc& real, const psum_obj_field_desc& d) {
            if (real.dim != d.dim || real.on_device != d.on_device
                || real.quantity_size != d.quantity_size || real.scalar_kind != d.scalar_kind)
                return equality::type_diff;
            for (int i = 0; i < d.dim; i++)
                if (real.node_counts[i] != d.node_counts[i] || real.lower[i] != d.lower[i]
                    || real.upper[i] != d.upper[i]) return equality::value_diff;
            return equality::same;
        }

        // A field needs its grid to be constructed.
        // grid_of(std::integral_constant<int, Dim>) must return the
        // simple_grid<Dim>& the field is to live on (it has already verified
        // that the named grid exists and has this dim). 
        template <typename Obj, typename GridOf>
        static void build(const psum_obj_field_desc& d, GridOf&& grid_of, sycl::queue& q, Obj& f) {
            constexpr int N = Obj::Dimension;
            using Scalar = typename Obj::Scalar;
            auto& g = grid_of(std::integral_constant<int, N>{});
            if constexpr (std::is_same_v<Obj, host_field_t<N, Obj::QuantitySize, Scalar>>)
                f = Obj(g);
            else
                f = Obj(q, g);
        }
    };

} // namespace psum_runtime

#endif
