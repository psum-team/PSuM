#ifndef PSUM_RUNTIME_GRID_MANAGER_HPP
#define PSUM_RUNTIME_GRID_MANAGER_HPP

#include <psum/psum.hpp>
#include "family_manager.hpp"
#include "families.hpp"
#include "desc/grid_desc.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace psum_runtime {

    using namespace psum::prelude;

    template <typename Universe>
    class grid_manager {
    public:
        using status = manager_status_of<100>;   // the grid segment
        using family = grid_family<Universe>;

        grid_manager(serialization::object_manager& mng, last_failure& last) : grids_(mng, &last, 100) {}

        // ---------------- query / access ----------------
        bool exists(const char* name) const { return grids_.exists(name); }

        // hand the grid stored under `name` to fn(grid)
        template <typename F>
        bool find(const char* name, F&& fn) const {
            return grids_.find(name, [&](auto& g, auto) { fn(g); });
        }

        // the grid of EXACTLY Dim under `name` (throws when absent; never creates)
        template <int Dim>
        simple_grid<Dim>& get(const char* name) const {
            return grids_.template get<simple_grid<Dim>>(name);
        }

        // ---------------- descriptor interface ----------------
        // TYPE + VALUE invariants of the grid under `name`; false when absent
        bool desc_get(const char* name, psum_obj_grid_desc* out) const {
            return grids_.desc_get(name, out);
        }

        // authenticity of d against what the family holds
        status check(const psum_obj_grid_desc& d) const { return status(grids_.check(d)); }

        // create the grid d describes IF none exists under its name.
        status create(psum_obj_grid_desc* d) const { return status(grids_.create(d)); }

    private:
        family_manager<family> grids_;
    };

    // grid desc  <->  the simple_grid<Dim> family
    template <typename Universe>
    struct desc_family_protocol<psum_obj_grid_desc, grid_family<Universe>> {
        using protocol = void;

        // the descriptor's TYPE invariant (dim) IS the family key
        static constexpr int key(const psum_obj_grid_desc& d) { return d.dim; }

        template <typename Obj>
        static void extract(const Obj& g, psum_obj_grid_desc& d) {
            constexpr int N = Obj::Dimension;
            d.dim = N;
            [&]<int... D>(std::integer_sequence<int, D...>) {
                ((d.node_count[D] = g.template numCells<D>()), ...);
                ((d.lower[D] = g.template lowerBound<D>()), ...);
                ((d.upper[D] = g.template upperBound<D>()), ...);
            }(std::make_integer_sequence<int, N>{});
        }

        static equality equal(const psum_obj_grid_desc& real, const psum_obj_grid_desc& d) {
            if (real.dim != d.dim) return equality::type_diff;
            for (int i = 0; i < d.dim; i++)
                if (real.node_count[i] != d.node_count[i] || real.lower[i] != d.lower[i]
                    || real.upper[i] != d.upper[i]) return equality::value_diff;
            return equality::same;
        }

        // in place: the fresh default grid is assigned the descriptor's geometry
        template <typename Obj>
        static void build(const psum_obj_grid_desc& d, Obj& g) {
            constexpr int N = Obj::Dimension;
            std::vector<double> lo(d.lower, d.lower + N), up(d.upper, d.upper + N);
            std::vector<int> n(d.node_count, d.node_count + N);
            g = Obj(lo, up, n);
        }
    };

} // namespace psum_runtime

#endif
