#ifndef PSUM_RUNTIME_PSUM_ENGINE_HPP
#define PSUM_RUNTIME_PSUM_ENGINE_HPP

// psum_engine<Universe>: one engine under a certain universe configuration.
// engine = object_manager + PIC algorithm interface. 
// psum_engine is designed as the implementation layer of low-level API in psum runtime.

#include <psum/psum.hpp>
#include <cstdio>
#include <string>
#include <type_traits>
#include "psum_runtime_api.h"
#include "last_failure.hpp"
#include "universe.hpp"
#include "particle_operation.hpp"
#include "particle_operation_dispatch.hpp"
#include "grid_manager.hpp"
#include "field_manager.hpp"
#include "field_operation.hpp"
#include "species_manager.hpp"
#include "poisson.hpp"

namespace psum_runtime {

    using namespace psum::prelude;
    namespace pt = psum::property;   // this header stands on its own

    template <typename Universe>
    class psum_engine {
    public:

        psum_engine()
            : mng_("", q_), grid_manager_(mng_, last_), field_manager_(mng_, q_, grid_manager_, last_),
              field_operation_(field_manager_), species_manager_(mng_, q_, last_) {}
        // Every port of the C interface ends here. A failure that no manager
        // wrote down on its way out is entered as the bare code, one that was
        // written down keeps its entry. A success changes nothing.
        int checked(int code) {
            if (code && (described_ == last_.writes || last_.status != code)) last_.write(code, "");
            described_ = last_.writes;
            return code;
        }
        // the most recent failure of this engine, in more than its code
        int last_error(psum_error* out) const {
            if (!out) return PSUM_ERR_ARG;
            *out = psum_error{};
            if (last_.status == 0) return PSUM_OK;
            out->status = last_.status;
            out->segment = psum_status_segment(last_.status);
            snprintf(out->name, sizeof out->name, "%s", last_.name.c_str());
            describe(last_, out->message, sizeof out->message);
            return PSUM_OK;
        }

        // The object-descriptor conventions live in the managers: each owns a
        // family and its descriptor protocol. The engine names the manager and
        // passes the extras; the codes need no translating.
        template <typename Desc>
        int exists(const char* name) {
            if (!name) return PSUM_ERR_ARG;
            if constexpr (std::is_same_v<Desc, psum_obj_grid_desc>)
                return grid_manager_.exists(name) ? 1 : 0;
            else if constexpr (std::is_same_v<Desc, psum_obj_field_desc>)
                return field_manager_.exists(name) ? 1 : 0;
            else
                return species_manager_.exists(name) ? 1 : 0;
        }
        serialization::object_manager& manager() { return mng_; }
        sycl::queue& queue() { return q_; }

        // ---------- grid / field ----------
        int grid_desc_get(const char* name, psum_obj_grid_desc* out) {
            if (!name || !out) return PSUM_ERR_ARG;
            return grid_manager_.desc_get(name, out) ? PSUM_OK : missing(PSUM_SEGMENT_GRID, name);
        }
        int create_grid(psum_obj_grid_desc& d) {
            if (!d.name || d.dim < 1 || d.dim > 3) return PSUM_ERR_ARG;
            return grid_manager_.create(&d);
        }
        int create_field(psum_obj_field_desc& d, const char* grid_name) {
            return field_manager_.create(&d, grid_name);
        }
        // create from name + necessary parameters, RETURN an authentic desc
        int create_field_from_params(const char* name, const char* grid_name,
                                     int quantity_size, int scalar_kind, int on_device,
                                     psum_obj_field_desc* out) {
            if (!name || !grid_name || !out) return PSUM_ERR_ARG;
            psum_obj_field_desc d = {};
            d.name = name;
            d.on_device = on_device;
            d.quantity_size = quantity_size;
            d.scalar_kind = scalar_kind;
            int r = create_field(d, grid_name);
            if (r == PSUM_OK || psum_status_base(r) == PSUM_ERR_EXISTS) *out = d;
            return r;
        }
        int field_desc_get(const char* name, psum_obj_field_desc* out) {
            if (!name || !out) return PSUM_ERR_ARG;
            return field_manager_.desc_get(name, out) ? PSUM_OK : missing(PSUM_SEGMENT_FIELD, name);
        }
        int fill_field(const char* n, double v) { return field_operation_.fill_field(n, v); }
        int field_get_node(const char* n, const int* i, double* v) { return field_operation_.field_get_node(n, i, v); }
        int field_shape(const char* n, psum_field_shape* s) { return field_operation_.field_shape(n, &s->dim, s->node_counts, &s->quantity_size, &s->scalar_kind); }
        int host_field_fill(const char* n, double v) { return field_operation_.host_field_fill(n, v); }
        int host_field_get_node(const char* n, const int* i, double* v) { return field_operation_.host_field_get_node(n, i, v); }
        int host_field_shape(const char* n, psum_field_shape* s) { return field_operation_.host_field_shape(n, &s->dim, s->node_counts, &s->quantity_size, &s->scalar_kind); }
        int field_copy(const char* d, const char* s) { return field_operation_.field_copy(d, s); }
        int zero_field(const char* n) { return field_operation_.zero_field(n); }
        int host_zero_field(const char* n) { return field_operation_.host_zero_field(n); }
        int deposit_to_density(const char* n, int axi, double scale) { return field_operation_.deposit_to_density(n, axi, scale); }
        int host_field_copy(const char* d, const char* s) { return field_operation_.host_field_copy(d, s); }

        // ---------- poisson solve ----------
        int poisson_init(const psum_opt_poisson_desc& p) {
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
        int create_particles(psum_obj_species_desc& d) {
            if (!d.name) return PSUM_ERR_ARG;
            return species_manager_.create(&d);
        }
        int species_desc_get(const char* name, psum_obj_species_desc* out) {
            if (!name || !out) return PSUM_ERR_ARG;
            return species_manager_.desc_get(name, out) ? PSUM_OK : missing(PSUM_SEGMENT_SPECIES, name);
        }
        int append_particle(const char* name, const double* pos, const double* vel) { return species_manager_.append(name, pos, vel); }
        int append_particles(const char* name, size_t count, const double* pos, const double* vel) { return species_manager_.append_n(name, count, pos, vel); }
        int particles_count(const char* name, size_t* count) { return species_manager_.count(name, count); }
        int particles_size(const char* name, size_t* count, int* dim) { return species_manager_.size(name, count, dim); }
        int particles_compress(const char* name) { return species_manager_.compress(name); }
        int particles_get_all(const char* name, double* pos, double* vel) { return species_manager_.get_all_pv(name, pos, vel); }
        int particles_get(const char* name, size_t index, double* pos, double* vel) { return species_manager_.get(name, index, pos, vel); }

        // ---------- particle operations ----------

        // adds weight_charge of every particle onto the scalar field named
        // field_name (grid dim as the electric field, mapping/order per e_id)
        int deposit_step(const psum_opt_deposit_desc& d) {
            if (!d.species || !d.field) return PSUM_ERR_ARG;
            return dispatch_species_field(d.species, d.field, [&](auto dc, auto& grp, auto& f) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::deposit_one<Universe, D>(mng_, d.species, grp, f, d.e_id, d.weight);
            });
        }

        // free flight: the drift operator for every particle (no strategy)
        int drift_step(const psum_opt_drift_desc& d) {
            if (!d.species) return PSUM_ERR_ARG;
            return species_manager_.dispatch_species(d.species, [&](auto dc, auto& grp) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::drift_one<Universe, D>(mng_, d.species, grp, d.e_id, d.dt,
                                                             d.deposit_field, d.deposit_weight);
            });
        }

        int leapfrog_step(const psum_opt_leapfrog_desc& d) {
            if (!d.species || !d.e_field) return PSUM_ERR_ARG;
            return dispatch_species_field(d.species, d.e_field, [&](auto dc, auto& grp, auto& f) {
                constexpr int Dim = decltype(dc)::value;
                return particle_operation_dispatch::leapfrog_one<Universe, Dim>(mng_, d.species, grp, f, d.e_id,
                                                             d.q_m, d.dt, d.deposit_field,
                                                             d.deposit_weight);
                (void)Dim;
            });
        }

        int boris_step(const psum_opt_boris_desc& d) {
            if (!d.species || !d.e_field || !d.b_field) return PSUM_ERR_ARG;
            // the E and B fields are looked up one after the other over the
            // window; the species dim fixes the particle type
            return dispatch_species_field(d.species, d.e_field, [&](auto dc, auto& grp, auto& ef) {
                constexpr int Dim = decltype(dc)::value;
                return field_manager_.dispatch_field(d.b_field, [&](auto& bf) {
                    return particle_operation_dispatch::boris_one<Universe, Dim>(mng_, d.species, grp, ef, bf,
                                                              d.e_id, d.b_id, d.q_m, d.dt,
                                                              d.deposit_field,
                                                              d.deposit_weight);
                });
            });
        }

        // save a single object (one species / one field) into a named .mas
        // archive (replace mode)
        int species_save(const char* name, const char* path) {
            if (!name || !path) return PSUM_ERR_ARG;
            return species_manager_.dispatch_species(name, [&](auto dc, auto& grp) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), grp);
                return 0;
            });
        }
        int field_save(const char* name, const char* path) {
            if (!name || !path) return PSUM_ERR_ARG;
            return field_manager_.dispatch_field(name, [&](auto& f) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), f);
                return 0;
            });
        }
        int host_field_save(const char* name, const char* path) {
            if (!name || !path) return PSUM_ERR_ARG;
            return field_manager_.dispatch_host_field(name, [&](auto& f) {
                serialization::mas_file fp(path, serialization::mas_file::replaceMode);
                save(fp, std::string(name), f);
                return 0;
            });
        }

        int leave_buffer_get(const char* name, psum_readback* out) {
            if (!out) return PSUM_ERR_ARG;
            return species_manager_.leave_buffer(name, &out->count, &out->dim, out->pos, out->vel);
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
            info->auto_wrap = Universe::auto_wrap ? 1 : 0;
            info->enable_float_fields = Universe::enable_float_fields ? 1 : 0;
            return 0;
        }

        int field_add(const char* n, double v) { return field_operation_.add_field(n, v); }
        int field_add_field(const char* d, const char* s) { return field_operation_.add_field_from(d, s); }
        int host_field_add_field(const char* d, const char* s) { return field_operation_.host_add_field_from(d, s); }
        int host_field_add(const char* n, double v) { return field_operation_.host_add_field(n, v); }

        // batch append + deposit
        int particles_append_deposit(const char* name, size_t count, const double* pos, const double* vel, const char* field_name, int e_id, double weight, size_t* accepted) {
            if (!name || !field_name) return PSUM_ERR_ARG;
            return dispatch_species_field(name, field_name, [&](auto dc, auto& grp, auto& f) {
                constexpr int D = decltype(dc)::value;
                return particle_operation_dispatch::inject_batch<Universe, D>(
                    mng_, name, grp, f, e_id, count, pos, vel, weight, accepted);
            });
        }

    private:
        sycl::queue q_{sycl::default_selector()};
        serialization::object_manager mng_;
        last_failure last_;                    // the most recent failure; the managers write into it too
        unsigned described_ = 0;               // last_.writes when checked() last looked
        grid_manager<Universe> grid_manager_;          // declared before field_manager_: that one refers to it
        field_manager<Universe> field_manager_;
        field_operation<Universe> field_operation_;   // numerical work on fields
        species_manager<Universe> species_manager_;
        poisson::slots poisson_slots_;

        // NOT_FOUND, reported by the family of `segment` and about `name`
        int missing(int segment, const char* name) {
            const int code = psum_code(PSUM_ERR_NOT_FOUND, segment);
            last_.write(code, name);
            return code;
        }
        // "field 'E': no such object" for a family's verdict; the name of the
        // code, and the object if one is known, for any other
        static void describe(const last_failure& e, char* out, size_t n) {
            const int segment = psum_status_segment(e.status);
            std::string text = psum_status_name(e.status);
            if (segment != PSUM_SEGMENT_NONE) {
                text = psum_segment_name(segment);
                if (!e.name.empty()) text += " '" + e.name + "'";
                if (*meaning(e.status)) text += std::string(": ") + meaning(e.status);
            } else if (!e.name.empty()) {
                text += " '" + e.name + "'";
            }
            snprintf(out, n, "%s", text.c_str());
        }
        // a family's verdict in words; "" for a code that is not one of them
        static const char* meaning(int status) {
            switch (psum_status_base(status)) {
                case PSUM_ERR_ARG: return "invalid argument";
                case PSUM_ERR_NOT_FOUND: return "no such object";
                case PSUM_ERR_EXISTS: return "an identical object exists";
                case PSUM_ERR_UNSUPPORTED: return "no such type in this build";
                case PSUM_ERR_OVERFLOW: return "capacity exceeded";
                case PSUM_ERR_CONFLICT_TYPE: return "an object with other TYPE invariants exists";
                case PSUM_ERR_CONFLICT_VALUE: return "an object with other VALUE invariants exists";
                default: return "";
            }
        }

        template <typename F>
        int dispatch_species_field(const char* species_name, const char* field_name, F&& fn) {
            return species_manager_.dispatch_species(species_name, [&](auto dc, auto& grp) {
                return field_manager_.dispatch_field(field_name, [&](auto& f) {
                    return fn(dc, grp, f);
                });
            });
        }

    };

} // namespace psum_runtime

#endif
