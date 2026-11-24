#ifndef PSUM_RUNTIME_GRID_DESC_HPP
#define PSUM_RUNTIME_GRID_DESC_HPP

/* psum_obj_grid_desc: object descriptor of a grid. */
typedef struct psum_obj_grid_desc {
    const char* name;
    /* TYPE invariant -> simple_grid<Dim> */
    int dim;                     /* 1..3 */
    /* VALUE invariants */
    double lower[3];             /* read-only setting, dim doubles */
    double upper[3];             /* read-only setting, dim doubles */
    int node_count[3];           /* read-only setting, CELLS per axis */
} psum_obj_grid_desc;

#endif /* PSUM_RUNTIME_GRID_DESC_HPP */
