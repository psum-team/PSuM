#ifndef PSUM_RUNTIME_FIELD_DESC_HPP
#define PSUM_RUNTIME_FIELD_DESC_HPP

/* psum_obj_field_desc: object descriptor of a field. */
typedef struct psum_obj_field_desc {
    const char* name;
    /* TYPE invariants -> field_t<Dim,QS,Scalar> or host_field_t<Dim,QS,Scalar> */
    int dim;                     /* 1..3 */
    int on_device;               /* 1: device field / 0: host field */
    int quantity_size;           /* 1..3 (1d: 1 only) */
    int scalar_kind;             /* PSUM_SCALAR_DOUBLE / _FLOAT */
    /* VALUE invariants: geometry (inlined from the grid, no name ref) */
    double lower[3];
    double upper[3];
    int node_counts[3];          /* nodes per axis (cells + 1) */
} psum_obj_field_desc;

#endif /* PSUM_RUNTIME_FIELD_DESC_HPP */
