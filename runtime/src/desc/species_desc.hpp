#ifndef PSUM_RUNTIME_SPECIES_DESC_HPP
#define PSUM_RUNTIME_SPECIES_DESC_HPP

/* psum_obj_species_desc: object descriptor of a species. */
typedef struct psum_obj_species_desc {
    const char* name;
    /* TYPE invariant */
    int dim;                     /* 1..3 */
    /* VALUE invariants */
    int has_global_mass;
    double global_mass;
    int has_global_charge;
    double global_charge;
    int has_global_weight;
    double global_weight;
} psum_obj_species_desc;

#endif /* PSUM_RUNTIME_SPECIES_DESC_HPP */
