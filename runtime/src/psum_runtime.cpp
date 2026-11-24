#include "psum_runtime_api.h"
#include "psum_engine.hpp"

using psum_runtime::psum_engine;
using psum_runtime::configured_universe;
using engine_t = psum_engine<configured_universe>;

struct psum_engine_handle { engine_t engine; };

extern "C" {

psum_engine_t psum_engine_create(void) {
    return new psum_engine_handle{};
}

int psum_engine_destroy(psum_engine_t engine) {
    if (!engine) return PSUM_ERR_ARG;
    delete engine;
    return 0;
}

#define PSUM_ENG engine->engine

/* out = the most recent failure of this engine; all zeroes when none failed */
int psum_engine_last_error(psum_engine_t engine, psum_error* out) {
    if (!engine || !out) return PSUM_ERR_ARG;
    return PSUM_ENG.last_error(out);
}

int psum_engine_grid_create(psum_engine_t engine, psum_obj_grid_desc* grid) {
    if (!engine) return PSUM_ERR_ARG;
    if (!grid) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.create_grid(*grid));
}

int psum_engine_field_create(psum_engine_t engine, psum_obj_field_desc* field, const char* grid_name) {
    if (!engine) return PSUM_ERR_ARG;
    if (!field) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.create_field(*field, grid_name));
}

int psum_engine_field_fill(psum_engine_t engine, const char* name, double value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.fill_field(name, value));
}

int psum_engine_field_zero(psum_engine_t engine, const char* name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.zero_field(name));
}

int psum_engine_host_field_fill(psum_engine_t engine, const char* name, double value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_field_fill(name, value));
}

int psum_engine_host_field_zero(psum_engine_t engine, const char* name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_zero_field(name));
}

int psum_engine_deposit_to_density(psum_engine_t engine, const char* name, int geometry, double scale) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.deposit_to_density(name, geometry == PSUM_GEOMETRY_AXISYMMETRIC, scale));
}

int psum_engine_species_save_mas(psum_engine_t engine, const char* species, const char* file) {
    if (!engine) return PSUM_ERR_ARG;
    if (!species || !file) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.species_save(species, file));
}

int psum_engine_field_save_mas(psum_engine_t engine, const char* field, const char* file) {
    if (!engine) return PSUM_ERR_ARG;
    if (!field || !file) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.field_save(field, file));
}

int psum_engine_host_field_save_mas(psum_engine_t engine, const char* field, const char* file) {
    if (!engine) return PSUM_ERR_ARG;
    if (!field || !file) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.host_field_save(field, file));
}

int psum_engine_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.field_copy(dst_name, src_name));
}

int psum_engine_host_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_field_copy(dst_name, src_name));
}

int psum_engine_field_get_node(psum_engine_t engine, const char* name,
                               const int* node_indices, double* value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.field_get_node(name, node_indices, value));
}

int psum_engine_host_field_get_node(psum_engine_t engine, const char* name,
                                    const int* node_indices, double* value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_field_get_node(name, node_indices, value));
}

int psum_engine_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name || !shape) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.field_shape(name, shape));
}

int psum_engine_host_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name || !shape) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.host_field_shape(name, shape));
}

int psum_engine_particles_create(psum_engine_t engine, psum_obj_species_desc* species) {
    if (!engine) return PSUM_ERR_ARG;
    if (!species || !species->name) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.create_particles(*species));
}

int psum_engine_grid_desc_get(psum_engine_t engine, const char* name, psum_obj_grid_desc* out) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name || !out) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.grid_desc_get(name, out));
}

int psum_engine_field_create_on_grid(psum_engine_t engine,
                                     const char* field_name, const char* grid_name,
                                     int quantity_size, int scalar_kind, int on_device,
                                     psum_obj_field_desc* out) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.create_field_from_params(field_name, grid_name, quantity_size,
                                                             scalar_kind, on_device, out));
}

int psum_engine_species_desc_get(psum_engine_t engine, const char* name, psum_obj_species_desc* out) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name || !out) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.species_desc_get(name, out));
}

int psum_engine_field_desc_get(psum_engine_t engine, const char* name, psum_obj_field_desc* out) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name || !out) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.field_desc_get(name, out));
}

int psum_engine_particles_append(psum_engine_t engine, const char* name,
                                 const double* pos, const double* vel) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.append_particle(name, pos, vel));
}

int psum_engine_particles_append_deposit(psum_engine_t engine, const char* name,
                                         size_t count, const double* pos, const double* vel,
                                         const char* deposit_field, int e_id, double weight,
                                         size_t* accepted) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.particles_append_deposit(name, count, pos, vel,
                                             deposit_field, e_id, weight, accepted));
}

int psum_engine_particles_append_n(psum_engine_t engine, const char* name,
                                   size_t count, const double* pos, const double* vel) {
    if (!engine) return PSUM_ERR_ARG;
    if (!name) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.append_particles(name, count, pos, vel));
}

int psum_engine_particles_count(psum_engine_t engine, const char* name, size_t* count) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.particles_count(name, count));
}

int psum_engine_particles_size(psum_engine_t engine, const char* name, size_t* count, int* dim) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.particles_size(name, count, dim));
}

int psum_engine_particles_compress(psum_engine_t engine, const char* name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.particles_compress(name));
}

int psum_engine_particles_get(psum_engine_t engine, const char* name, size_t index,
                              double* pos, double* vel) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.particles_get(name, index, pos, vel));
}

int psum_engine_particles_get_all(psum_engine_t engine, const char* name,
                                  double* pos, double* vel) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.particles_get_all(name, pos, vel));
}

int psum_engine_leave_buffer_get(psum_engine_t engine, const char* species, psum_readback* out) {
    if (!engine) return PSUM_ERR_ARG;
    if (!species || !out) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.leave_buffer_get(species, out));
}

int psum_engine_universe_info(psum_engine_t engine, psum_universe_info* info) {
    if (!engine) return PSUM_ERR_ARG;
    if (!info) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.universe_info(info));
}

int psum_engine_field_add(psum_engine_t engine, const char* name, double value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.field_add(name, value));
}

int psum_engine_host_field_add(psum_engine_t engine, const char* name, double value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_field_add(name, value));
}

int psum_engine_field_add_field(psum_engine_t engine, const char* dst, const char* src) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.field_add_field(dst, src));
}

int psum_engine_host_field_add_field(psum_engine_t engine, const char* dst, const char* src) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.host_field_add_field(dst, src));
}

int psum_engine_deposit(psum_engine_t engine, const psum_opt_deposit_desc* deposit) {
    if (!engine) return PSUM_ERR_ARG;
    if (!deposit) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.deposit_step(*deposit));
}

int psum_engine_drift(psum_engine_t engine, const psum_opt_drift_desc* drift) {
    if (!engine) return PSUM_ERR_ARG;
    if (!drift) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.drift_step(*drift));
}

int psum_engine_leapfrog(psum_engine_t engine, const psum_opt_leapfrog_desc* push) {
    if (!engine) return PSUM_ERR_ARG;
    if (!push) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.leapfrog_step(*push));
}

int psum_engine_boris(psum_engine_t engine, const psum_opt_boris_desc* boris) {
    if (!engine) return PSUM_ERR_ARG;
    if (!boris) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.boris_step(*boris));
}

int psum_engine_poisson_init(psum_engine_t engine, const psum_opt_poisson_desc* poisson) {
    if (!engine) return PSUM_ERR_ARG;
    if (!poisson) return PSUM_ENG.checked(PSUM_ERR_ARG);
    return PSUM_ENG.checked(PSUM_ENG.poisson_init(*poisson));
}

int psum_engine_poisson_bc_value(psum_engine_t engine, int solver_kind,
                                 int boundary_index, double value) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.poisson_bc_value(solver_kind, boundary_index, value));
}

int psum_engine_poisson_solve_d(psum_engine_t engine, int solver_kind,
                                const char* rho_name, const char* phi_name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.poisson_solve_d(solver_kind, rho_name, phi_name));
}

int psum_engine_poisson_solve_h(psum_engine_t engine, int solver_kind,
                                const char* rho_name, const char* phi_name) {
    if (!engine) return PSUM_ERR_ARG;
    return PSUM_ENG.checked(PSUM_ENG.poisson_solve_h(solver_kind, rho_name, phi_name));
}

} // extern "C"
