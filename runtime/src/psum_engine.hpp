#ifndef PSUM_RUNTIME_PSUM_ENGINE_HPP
#define PSUM_RUNTIME_PSUM_ENGINE_HPP

// psum_engine<Universe>: one engine under a certain universe configuration.
// engine = object_manager + PIC algorithm interface. 
// psum_engine is designed as the implementation layer of low-level API in psum runtime.

#include <psum/psum.hpp>
#include "psum_runtime_api.h"
#include "universe.hpp"
#include "particle_operation.hpp"
#include "particle_operation_dispatch.hpp"
#include "field_manager.hpp"
#include "group_manager.hpp"
#include "poisson.hpp"

namespace psum_runtime {

    using namespace psum::prelude;

    template <typename Universe>
    class psum_engine {
    public:
        template <int Dim>
        using particle_t = typename Universe::template particle<Dim>;
        template <int Dim>
        using group_t = typename Universe::template group<Dim>;
        template <int Dim>
        using grid_t = typename field_manager<Universe>::template grid_t<Dim>;
        template <int Dim, int QuantitySize, typename Scalar = double>
        using field_t = typename field_manager<Universe>::template field_t<Dim, QuantitySize, Scalar>;

        psum_engine() : mng_("", q_), field_manager_(mng_, q_), group_manager_(mng_, q_) {}
        serialization::object_manager& manager() { return mng_; }
        sycl::queue& queue() { return q_; }

        // ---------- field/grid ----------
        int create_grid(const psum_grid_desc& d) { return field_manager_.create_grid(d.name, d.dim, d.lower, d.upper, d.node_count); }
        int create_field(const psum_field_desc& d) { return field_manager_.create_field(d.name, d.grid, d.quantity_size, d.scalar_kind); }
        int fill_field(const char* n, double v) { return field_manager_.fill_field(n, v); }
        int field_get_node(const char* n, const int* i, double* v) { return field_manager_.field_get_node(n, i, v); }
        int field_shape(const char* n, psum_field_shape* s) { return field_manager_.field_shape(n, &s->dim, s->node_counts, &s->quantity_size, &s->scalar_kind); }
        int host_field_create(const psum_field_desc& d) { return field_manager_.host_field_create(d.name, d.grid, d.quantity_size, d.scalar_kind); }
        int host_field_fill(const char* n, double v) { return field_manager_.host_field_fill(n, v); }
        int host_field_get_node(const char* n, const int* i, double* v) { return field_manager_.host_field_get_node(n, i, v); }
        int host_field_shape(const char* n, psum_field_shape* s) { return field_manager_.host_field_shape(n, &s->dim, s->node_counts, &s->quantity_size, &s->scalar_kind); }
        int field_copy(const char* d, const char* s) { return field_manager_.field_copy(d, s); }
        int zero_field(const char* n) { return field_manager_.zero_field(n); }
        int host_zero_field(const char* n) { return field_manager_.host_zero_field(n); }
        int deposit_to_density(const char* n, int axi, double scale) { return field_manager_.deposit_to_density(n, axi, scale); }
        int host_field_copy(const char* d, const char* s) { return field_manager_.host_field_copy(d, s); }

        // ---------- poisson solve ----------
        int poisson_init(const psum_poisson_desc& p) {
            return poisson::init(mng_, poisson_slots_, p.solver_kind, p.bc_kinds, p.robin_center, p.grid, p.backend, p.options, p.epsilon);
        }
        int poisson_bc_value(int solver_kind, int boundary_index, double value) {
            return poisson::bc_value(mng_, poisson_slots_, solver_kind, boundary_index, value);
        }
        int poisson_solve_d(int solver_kind, const char* rho_name, const char* phi_name) {
            return poisson::solve_d(mng_, poisson_slots_, solver_kind, rho_name, phi_name);
        }
        int poisson_solve_h(int solver_kind, const char* rho_name, const char* phi_name) {
            return poisson::solve_h(mng_, poisson_slots_, solver_kind, rho_name, phi_name);
        }

        // ---------- particles ----------
        int create_particles(const char* name, int dim) { return group_manager_.create(name, dim); }
        int append_particle(const char* name, const double* pos, const double* vel) { return group_manager_.append(name, pos, vel); }
        int append_particles(const psum_particles_desc& d, size_t count, const double* pos, const double* vel) { return group_manager_.append_n(d.group, count, pos, vel); }
        int particles_count(const char* name, size_t* count) { return group_manager_.count(name, count); }
        int particles_size(const char* name, size_t* count, int* dim) { return group_manager_.size(name, count, dim); }
        int particles_compress(const char* name) { return group_manager_.compress(name); }
        int particles_get_all(const char* name, double* pos, double* vel) { return group_manager_.get_all_pv(name, pos, vel); }
        int particles_get(const char* name, size_t index, double* pos, double* vel) { return group_manager_.get(name, index, pos, vel); }

        // ---------- particle operations ----------

        // adds weight_charge of every particle onto the scalar field named
        // field_name (grid dim as the electric field, mapping/order per e_id)
        int deposit_step(const psum_deposit_desc& d) {
            if (!d.group || !d.field) return -1;
            return dispatch_group_field(d.group, d.field, [&](auto dc, auto& grp, auto& f) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::deposit_one<Universe, D>(mng_, d.group, grp, f, d.e_id, d.weight);
            });
        }

        // free flight: the drift operator for every particle (no strategy)
        int drift_step(const psum_drift_desc& d) {
            if (!d.group) return -1;
            return group_manager_.dispatch_group(d.group, [&](auto dc, auto& grp) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::drift_one<Universe, D>(mng_, d.group, grp, d.e_id, d.dt,
                                                             d.deposit_field, d.deposit_weight);
            });
        }

        int leapfrog_step(const psum_leapfrog_desc& d) {
            if (!d.group || !d.e_field) return -1;
            return dispatch_group_field(d.group, d.e_field, [&](auto dc, auto& grp, auto& f) {
                constexpr int Dim = decltype(dc)::value;
                return particle_operation_dispatch::leapfrog_one<Universe, Dim>(mng_, d.group, grp, f, d.e_id,
                                                             d.q_m, d.dt, d.deposit_field,
                                                             d.deposit_weight);
                (void)Dim;
            });
        }

        int boris_step(const psum_boris_desc& d) {
            if (!d.group || !d.e_field || !d.b_field) return -1;
            // the E and B fields are looked up one after the other over the
            // window; the group dim fixes the particle type
            return dispatch_group_field(d.group, d.e_field, [&](auto dc, auto& grp, auto& ef) {
                constexpr int Dim = decltype(dc)::value;
                return field_manager_.dispatch_field(d.b_field, [&](auto& bf) {
                    return particle_operation_dispatch::boris_one<Universe, Dim>(mng_, d.group, grp, ef, bf,
                                                              d.e_id, d.b_id, d.q_m, d.dt,
                                                              d.deposit_field,
                                                              d.deposit_weight);
                });
            });
        }

        // save a single object (one group / one field) into a named .mas
        // archive (replace mode)
        int group_save(const char* name, const char* path) {
            if (!name || !path) return -1;
            return group_manager_.dispatch_group(name, [&](auto dc, auto& grp) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), grp);
                return 0;
            });
        }
        int field_save(const char* name, const char* path) {
            if (!name || !path) return -1;
            return field_manager_.dispatch_field(name, [&](auto& f) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), f);
                return 0;
            });
        }
        int host_field_save(const char* name, const char* path) {
            if (!name || !path) return -1;
            return field_manager_.dispatch_host_field(name, [&](auto& f) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), f);
                return 0;
            });
        }

        int leave_buffer_get(const char* name, psum_readback* out) {
            if (!out) return -1;
            return group_manager_.leave_buffer(name, &out->count, &out->dim, out->pos, out->vel);
        }

        // runtime introspection of the engine's universe
        int universe_info(psum_universe_info* info) {
            using P1 = typename Universe::template particle<1>;
            using PosV = std::remove_cvref_t<decltype(psum::tag::get<pt::position>(std::declval<P1&>()))>;
            using VelV = std::remove_cvref_t<decltype(psum::tag::get<pt::velocity>(std::declval<P1&>()))>;
            info->dim_mask = Universe::dim_mask;
            info->pos_scalar = std::is_same_v<typename PosV::Scalar, double> ? 0 : 1;
            info->vel_scalar = std::is_same_v<typename VelV::Scalar, double> ? 0 : 1;
            info->e_order = Universe::e_order;
            info->b_order = Universe::b_order;
            info->periodic_grid = Universe::periodic_grid ? 1 : 0;
            info->enable_float_fields = Universe::enable_float_fields ? 1 : 0;
            return 0;
        }

        int field_add(const char* n, double v) { return field_manager_.add_field(n, v); }
        int field_add_field(const char* d, const char* s) { return field_manager_.add_field_from(d, s); }
        int host_field_add_field(const char* d, const char* s) { return field_manager_.host_add_field_from(d, s); }
        int host_field_add(const char* n, double v) { return field_manager_.host_add_field(n, v); }

        // batch append + deposit
        int particles_append_deposit(const psum_particles_desc& d, size_t count, const double* pos, const double* vel, const char* field_name, int e_id, double weight, size_t* accepted) {
            if (!d.group || !field_name) return -1;
            return dispatch_group_field(d.group, field_name, [&](auto dc, auto& grp, auto& f) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::inject_batch<Universe, D>(
                    mng_, d.group, grp, f, e_id, count, pos, vel, weight, accepted);
            });
        }

    private:
        sycl::queue q_{sycl::default_selector()};
        serialization::object_manager mng_;
        field_manager<Universe> field_manager_;
        group_manager<Universe> group_manager_;
        poisson::slots poisson_slots_;

        template <typename F>
        int dispatch_group_field(const char* group_name, const char* field_name, F&& fn) {
            return group_manager_.dispatch_group(group_name, [&](auto dc, auto& grp) {
                return field_manager_.dispatch_field(field_name, [&](auto& f) {
                    return fn(dc, grp, f);
                });
            });
        }

    };

} // namespace psum_runtime

#endif
