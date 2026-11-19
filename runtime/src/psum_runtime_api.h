#ifndef PSUM_RUNTIME_API_H
#define PSUM_RUNTIME_API_H

#include <stddef.h>   /* size_t */

/* C interface of the PSuM runtime. Including object-manager, I/O and PIC algorithm calls.
 * The universe (a series of compile configuration, see universe.hpp) will determine the ability closure but will not effect the runtime API. */

typedef struct psum_engine_handle* psum_engine_t;

/* runtime universe introspection (out): scalars as 0=double / 1=float, the switches as 0/1 */
typedef struct psum_universe_info {
    int dim_mask;
    int pos_scalar;
    int vel_scalar;
    int e_order;
    int b_order;
    int periodic_grid;
    int enable_float_fields;
} psum_universe_info;

#ifdef __cplusplus
extern "C" {
#endif

int psum_engine_universe_info(psum_engine_t engine, psum_universe_info* info);

/* E strategy ids: bit 2 = direct (else gradient), bit 1 = axisymmetric (else cartesian). */
enum { PSUM_E_GRADIENT_CARTESIAN = 0, PSUM_E_GRADIENT_AXISYM = 1, PSUM_E_DIRECT_CARTESIAN = 2, PSUM_E_DIRECT_AXISYM = 3 };
/* B strategy ids: bit 1 = axisymmetric (else cartesian). */
enum { PSUM_B_CARTESIAN = 0, PSUM_B_AXISYM = 1 };

/* solver kinds: one slot per dimension; the 2d slot is cartesian or axisymmetric */
enum { PSUM_POISSON_1D = 0, PSUM_POISSON_2D_CARTESIAN, PSUM_POISSON_2D_AXISYMMETRIC, PSUM_POISSON_3D };
/* per-edge condition kinds; periodic edges stay unset inside the solvers */
enum { PSUM_BC_PERIODIC = 0, PSUM_BC_DIRICHLET = 1, PSUM_BC_ROBIN = 2, PSUM_BC_NEUMANN = 3 };

/* geometry kinds for the density conversion */
enum { PSUM_GEOMETRY_CARTESIAN = 0, PSUM_GEOMETRY_AXISYMMETRIC = 1 };

/* field scalar kinds */
enum { PSUM_SCALAR_DOUBLE = 0, PSUM_SCALAR_FLOAT = 1 };

psum_engine_t psum_engine_create(void);

// ---------- descriptor structs -------------------

typedef struct psum_grid_desc {
    const char* name;
    int dim;                     /* 1..3 */
    const double* lower;         /* dim doubles */
    const double* upper;         /* dim doubles */
    const int* node_count;       /* dim ints, CELLS per axis */
} psum_grid_desc;

typedef struct psum_field_desc {
    const char* name;
    const char* grid;
    int quantity_size;           /* 1..3 (1d: 1 only) */
    int scalar_kind;             /* PSUM_SCALAR_DOUBLE / _FLOAT (float needs a universe built with PSUM_ENABLE_FLOAT_FIELDS) */
} psum_field_desc;

typedef struct psum_particles_desc {
    const char* group;
} psum_particles_desc;

/* one leapfrog step (E-only kick + drift), runtime-fused deposit */
typedef struct psum_leapfrog_desc {
    const char* group;
    const char* e_field;
    int e_id;                    /* strategy id, see PSUM_E_* */
    double q_m;
    double dt;
    const char* deposit_field;   /* NULL: no deposit (gate uses e_field grid) */
    double deposit_weight;
} psum_leapfrog_desc;

/* one boris step (E + B rotation), runtime-fused deposit */
typedef struct psum_boris_desc {
    const char* group;
    const char* e_field;
    const char* b_field;
    int e_id;
    int b_id;
    double q_m;
    double dt;
    const char* deposit_field;   /* NULL: no deposit (gate uses e_field grid) */
    double deposit_weight;
} psum_boris_desc;

/* free flight (x += v dt); deposit (and its gate) only with a field */
typedef struct psum_drift_desc {
    const char* group;
    double dt;
    int e_id;                    /* deposit mapping */
    const char* deposit_field;   /* NULL: plain drift, no gate */
    double deposit_weight;
} psum_drift_desc;

/* standalone deposit of a whole group onto a scalar field */
typedef struct psum_deposit_desc {
    const char* group;
    const char* field;
    int e_id;
    double weight;
} psum_deposit_desc;

/* read-back arrays; pass pos/vel NULL to query shapes only */
typedef struct psum_readback {
    size_t count;                /* filled by the runtime */
    int dim;                     /* filled by the runtime */
    double* pos;                 /* dim*count doubles, or NULL */
    double* vel;                 /* dim*count doubles, or NULL */
} psum_readback;

/* a poisson slot setup */
typedef struct psum_poisson_desc {
    int solver_kind;             /* PSUM_POISSON_1D / 2D_CARTESIAN / ... */
    const int* bc_kinds;         /* 2*dim entries, PSUM_BC_* */
    const double* robin_center;  /* dim doubles, Robin only */
    const char* grid;
    const char* backend;         /* NULL/"" = native */
    const char* options;         /* NULL/"" = defaults */
    double epsilon;
} psum_poisson_desc;

/* shape of one field, filled by the runtime */
typedef struct psum_field_shape {
    int dim;
    int node_counts[3];          /* NODES per axis */
    int quantity_size;
    int scalar_kind;
} psum_field_shape;


int psum_engine_destroy(psum_engine_t engine);

int psum_engine_grid_create(psum_engine_t engine, const psum_grid_desc* grid);

/* scalar_kind: PSUM_SCALAR_DOUBLE or PSUM_SCALAR_FLOAT */
int psum_engine_field_create(psum_engine_t engine, const psum_field_desc* field);

int psum_engine_field_fill(psum_engine_t engine, const char* name, double value);

int psum_engine_particles_create(psum_engine_t engine, const char* name, int dim);

/* pos/vel point to `dim` doubles each (dim as given at creation). */
int psum_engine_particles_append(psum_engine_t engine, const char* name, const double* pos, const double* vel);

/* batch append: pos/vel each hold dim*count doubles, one upload + insert */
int psum_engine_particles_append_n(psum_engine_t engine, const psum_particles_desc* batch,
                                   size_t count, const double* pos, const double* vel);

/* batch append + deposit in one shot */
int psum_engine_particles_append_deposit(psum_engine_t engine, const psum_particles_desc* batch,
                                         size_t count, const double* pos, const double* vel,
                                         const char* deposit_field, int e_id, double weight,
                                         size_t* accepted);

int psum_engine_particles_count(psum_engine_t engine, const char* name, size_t* count);

/* compacts the group: lazily deleted (invalid) slots are removed */
int psum_engine_particles_compress(psum_engine_t engine, const char* name);

int psum_engine_leave_buffer_get(psum_engine_t engine, const char* group, psum_readback* out);

/* shape queries so callers can size buffers exactly */
int psum_engine_particles_size(psum_engine_t engine, const char* name, size_t* count, int* dim);
/* node_counts needs room for 3 ints (only `dim` written) */
int psum_engine_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape);
int psum_engine_host_field_shape(psum_engine_t engine, const char* name, psum_field_shape* shape);

int psum_engine_particles_get(psum_engine_t engine, const char* name, size_t index, double* pos, double* vel);

/* bulk read-back: pos/vel each hold dim*valid_count doubles (size them
 * with psum_particles_size); valid particles packed in order */
int psum_engine_particles_get_all(psum_engine_t engine, const char* name, double* pos, double* vel);

/* in-place += value on every node (background charge & friends) */
int psum_engine_field_add(psum_engine_t engine, const char* name, double value);
int psum_engine_host_field_add(psum_engine_t engine, const char* name, double value);
/* dst += src element-wise (matching dims/QS) */
int psum_engine_field_add_field(psum_engine_t engine, const char* dst, const char* src);
int psum_engine_host_field_add_field(psum_engine_t engine, const char* dst, const char* src);

int psum_engine_poisson_init(psum_engine_t engine, const psum_poisson_desc* poisson);

int psum_engine_poisson_bc_value(psum_engine_t engine, int solver_kind, int boundary_index, double value);

/* solves rho -> phi; device flavor takes device scalar fields (data
 * pointers straight into the solver: needs a device-capable backend) */
int psum_engine_poisson_solve_d(psum_engine_t engine, int solver_kind, const char* rho_name, const char* phi_name);

/* host flavor takes HOST scalar fields (needs a host backend) */
int psum_engine_poisson_solve_h(psum_engine_t engine, int solver_kind, const char* rho_name, const char* phi_name);

/* host scalar fields (QS=1) for the host solve flavor */
/* host fields mirror the device window (qs per dimension, scalar kind) */
int psum_engine_host_field_create(psum_engine_t engine, const psum_field_desc* field);
int psum_engine_host_field_fill(psum_engine_t engine, const char* name, double value);
int psum_engine_host_field_get_node(psum_engine_t engine, const char* name, const int* node_indices, double* value);

/* drift: the free-flight operator (x += v dt) for every particle */
int psum_engine_drift(psum_engine_t engine, const psum_drift_desc* drift);

/* one leapfrog step (E-only kick + drift) for every particle in chosen group. */
int psum_engine_leapfrog(psum_engine_t engine, const psum_leapfrog_desc* push);

/* one Boris step (rotation + drift); deposit parameters as above */
int psum_engine_boris(psum_engine_t engine, const psum_boris_desc* boris);

/* adds weight x charge of every particle onto the scalar field; mapping and order from e_strategy_id). */
int psum_engine_deposit(psum_engine_t engine, const psum_deposit_desc* deposit);

/* one node value of a field; indices has one entry per grid dimension, range-checked */
int psum_engine_field_get_node(psum_engine_t engine, const char* name, const int* node_indices, double* value);

/* zeroes a field (device and host variants) */
int psum_engine_field_zero(psum_engine_t engine, const char* name);
int psum_engine_host_field_zero(psum_engine_t engine, const char* name);

/* deposited charge -> density: v *= scale / cell measure */
int psum_engine_deposit_to_density(psum_engine_t engine, const char* name, int geometry, double scale);

/* same grid dim, same QS, same side: copies with double<->float conversion
 * (either direction; identical scalars also fine) */
int psum_engine_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name);
int psum_engine_host_field_copy(psum_engine_t engine, const char* dst_name, const char* src_name);

/* save one object (group / device field / host field) into a named .mas
 * archive (replace mode) */
int psum_engine_group_save_mas(psum_engine_t engine, const char* group, const char* file);
int psum_engine_field_save_mas(psum_engine_t engine, const char* field, const char* file);
int psum_engine_host_field_save_mas(psum_engine_t engine, const char* field, const char* file);

#ifdef __cplusplus
}
#endif

#endif /* PSUM_RUNTIME_API_H */
