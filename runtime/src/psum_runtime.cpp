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
    if (!engine) return -1;
    delete engine;
    return 0;
}

#define PSUM_ENG engine->engine

int psum_engine_grid_create(psum_engine_t engine, const psum_grid_desc* grid) {
    if (!engine || !grid) return -1;
    return PSUM_ENG.create_grid(*grid);
}

int psum_engine_field_create(psum_engine_t engine, const psum_field_desc* field) {
    if (!engine || !field) return -1;
    return PSUM_ENG.create_field(*field);
}

int psum_engine_field_fill(psum_engine_t engine, const char* name, double value) {
    if (!engine) return -1;
    return PSUM_ENG.fill_field(name, value);
}

int psum_engine_field_zero(psum_engine_t engine, const char* name) {
    if (!engine) return -1;
    return PSUM_ENG.zero_field(name);
}

int psum_engine_host_field_create(psum_engine_t engine, const psum_field_desc* field) {
    if (!engine || !field) return -1;
    return PSUM_ENG.host_field_create(*field);
}

int psum_engine_host_field_fill(psum_engine_t engine, const char* name, double value) {
    if (!engine) return -1;
    return PSUM_ENG.host_field_fill(name, value);
}

int psum_engine_host_field_zero(psum_engine_t engine, const char* name) {
    if (!engine) return -1;
    return PSUM_ENG.host_zero_field(name);
}

int psum_engine_deposit_to_density(psum_engine_t engine, const char* name, int geometry, double scale) {
    if (!engine) return -1;
    return PSUM_ENG.deposit_to_density(name, geometry == PSUM_GEOMETRY_AXISYMMETRIC, scale);
}

int psum_engine_group_save_mas(psum_engine_t engine, const char* group, const char* file) {
    if (!engine || !group || !file) return -1;
    return PSUM_ENG.group_save(group, file);
}

int psum_engine_field_save_mas(psum_engine_t engine, const char* field, const char* file) {
    if (!engine || !field || !file) return -1;
    return PSUM_ENG.field_save(field, file);
}

int psum_engine_host_field_save_mas(psum_engine_t engine, const char* field, const char* file) {
    if (!engine || !field || !file) return -1;
    return PSUM_ENG.host_field_save(field, file);
}

int psum_engine_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name) {
    if (!engine) return -1;
    return PSUM_ENG.field_copy(dst_name, src_name);
}

int psum_engine_host_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name) {
    if (!engine) return -1;
    return PSUM_ENG.host_field_copy(dst_name, src_name);
}

int psum_engine_field_get_node(psum_engine_t engine, const char* name,
                               const int* node_indices, double* value) {
    if (!engine) return -1;
    return PSUM_ENG.field_get_node(name, node_indices, value);
}

int psum_engine_host_field_get_node(psum_engine_t engine, const char* name,
                                    const int* node_indices, double* value) {
    if (!engine) return -1;
    return PSUM_ENG.host_field_get_node(name, node_indices, value);
}

int psum_engine_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape) {
    if (!engine || !name || !shape) return -1;
    return PSUM_ENG.field_shape(name, shape);
}

int psum_engine_host_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape) {
    if (!engine || !name || !shape) return -1;
    return PSUM_ENG.host_field_shape(name, shape);
}

int psum_engine_particles_create(psum_engine_t engine, const char* name, int dim) {
    if (!engine) return -1;
    return PSUM_ENG.create_particles(name, dim);
}

int psum_engine_particles_append(psum_engine_t engine, const char* name,
                                 const double* pos, const double* vel) {
    if (!engine) return -1;
    return PSUM_ENG.append_particle(name, pos, vel);
}

int psum_engine_particles_append_deposit(psum_engine_t engine, const psum_particles_desc* batch,
                                         size_t count, const double* pos, const double* vel,
                                         const char* deposit_field, int e_id, double weight,
                                         size_t* accepted) {
    if (!engine || !batch) return -1;
    return PSUM_ENG.particles_append_deposit(*batch, count, pos, vel,
                                             deposit_field, e_id, weight, accepted);
}

int psum_engine_particles_append_n(psum_engine_t engine, const psum_particles_desc* batch,
                                   size_t count, const double* pos, const double* vel) {
    if (!engine || !batch) return -1;
    return PSUM_ENG.append_particles(*batch, count, pos, vel);
}

int psum_engine_particles_count(psum_engine_t engine, const char* name, size_t* count) {
    if (!engine) return -1;
    return PSUM_ENG.particles_count(name, count);
}

int psum_engine_particles_size(psum_engine_t engine, const char* name, size_t* count, int* dim) {
    if (!engine) return -1;
    return PSUM_ENG.particles_size(name, count, dim);
}

int psum_engine_particles_compress(psum_engine_t engine, const char* name) {
    if (!engine) return -1;
    return PSUM_ENG.particles_compress(name);
}

int psum_engine_particles_get(psum_engine_t engine, const char* name, size_t index,
                              double* pos, double* vel) {
    if (!engine) return -1;
    return PSUM_ENG.particles_get(name, index, pos, vel);
}

int psum_engine_particles_get_all(psum_engine_t engine, const char* name,
                                  double* pos, double* vel) {
    if (!engine) return -1;
    return PSUM_ENG.particles_get_all(name, pos, vel);
}

int psum_engine_leave_buffer_get(psum_engine_t engine, const char* group, psum_readback* out) {
    if (!engine || !group || !out) return -1;
    return PSUM_ENG.leave_buffer_get(group, out);
}

int psum_engine_universe_info(psum_engine_t engine, psum_universe_info* info) {
    if (!engine || !info) return -1;
    return PSUM_ENG.universe_info(info);
}

int psum_engine_field_add(psum_engine_t engine, const char* name, double value) {
    if (!engine) return -1;
    return PSUM_ENG.field_add(name, value);
}

int psum_engine_host_field_add(psum_engine_t engine, const char* name, double value) {
    if (!engine) return -1;
    return PSUM_ENG.host_field_add(name, value);
}

int psum_engine_field_add_field(psum_engine_t engine, const char* dst, const char* src) {
    if (!engine) return -1;
    return PSUM_ENG.field_add_field(dst, src);
}

int psum_engine_host_field_add_field(psum_engine_t engine, const char* dst, const char* src) {
    if (!engine) return -1;
    return PSUM_ENG.host_field_add_field(dst, src);
}

int psum_engine_deposit(psum_engine_t engine, const psum_deposit_desc* deposit) {
    if (!engine || !deposit) return -1;
    return PSUM_ENG.deposit_step(*deposit);
}

int psum_engine_drift(psum_engine_t engine, const psum_drift_desc* drift) {
    if (!engine || !drift) return -1;
    return PSUM_ENG.drift_step(*drift);
}

int psum_engine_leapfrog(psum_engine_t engine, const psum_leapfrog_desc* push) {
    if (!engine || !push) return -1;
    return PSUM_ENG.leapfrog_step(*push);
}

int psum_engine_boris(psum_engine_t engine, const psum_boris_desc* boris) {
    if (!engine || !boris) return -1;
    return PSUM_ENG.boris_step(*boris);
}

int psum_engine_poisson_init(psum_engine_t engine, const psum_poisson_desc* poisson) {
    if (!engine || !poisson) return -1;
    return PSUM_ENG.poisson_init(*poisson);
}

int psum_engine_poisson_bc_value(psum_engine_t engine, int solver_kind,
                                 int boundary_index, double value) {
    if (!engine) return -1;
    return PSUM_ENG.poisson_bc_value(solver_kind, boundary_index, value);
}

int psum_engine_poisson_solve_d(psum_engine_t engine, int solver_kind,
                                const char* rho_name, const char* phi_name) {
    if (!engine) return -1;
    return PSUM_ENG.poisson_solve_d(solver_kind, rho_name, phi_name);
}

int psum_engine_poisson_solve_h(psum_engine_t engine, int solver_kind,
                                const char* rho_name, const char* phi_name) {
    if (!engine) return -1;
    return PSUM_ENG.poisson_solve_h(solver_kind, rho_name, phi_name);
}

} // extern "C"
