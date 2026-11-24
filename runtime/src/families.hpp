#ifndef PSUM_RUNTIME_FAMILIES_HPP
#define PSUM_RUNTIME_FAMILIES_HPP

// The families the managers manage, each a keyed type list (family.hpp). A
// manager takes its family from here instead of defining it in place.

#include <psum/psum.hpp>
#include "family.hpp"
#include "species.hpp"

namespace psum_runtime {

    using namespace psum::prelude;

    // ---- the grid family ----
    //
    // Each entry is keyed by the grid's one TYPE invariant, its dimension.
    template <typename Universe>
    struct grid_family : type_list<keyed_type<1, simple_grid<1>>,
                                      keyed_type<2, simple_grid<2>>,
                                      keyed_type<3, simple_grid<3>>> {
        using base_list = type_list<keyed_type<1, simple_grid<1>>,
                                      keyed_type<2, simple_grid<2>>,
                                      keyed_type<3, simple_grid<3>>>;
    };

    // ---- the field family ------------------------------------------------
    //
    // A field's TYPE invariants are exactly four values, spelled out here
    // once. They ARE the key that selects the concrete C++ type:
    //
    //     dim            1..3
    //     quantity_size  1..3, and 1 for a 1d field
    //     scalar_kind    0 = double, 1 = float (float needs the universe switch)
    //     on_device      device field or host field

    template <int Dim, int QuantitySize, typename Scalar = double>
    using field_t = psum::field::device_field<Dim, psum::field::var_loc::nodeCentered, Scalar, QuantitySize>;
    template <int Dim, int QuantitySize, typename Scalar = double>
    using host_field_t = host_field<Dim, psum::field::var_loc::nodeCentered, Scalar, QuantitySize>;

    struct field_key {
        int dim;
        int quantity_size;
        int scalar_kind;
        bool on_device;
        bool operator==(const field_key&) const = default;
    };

    // one entry of the family: a key, and the type it selects
    template <int Dim, int QS, int ScalarKind, bool OnDevice>
    using field_of = keyed_type<
        field_key{Dim, QS, ScalarKind, OnDevice},
        std::conditional_t<OnDevice,
            field_t<Dim, QS, std::conditional_t<ScalarKind == 0, double, float>>,
            host_field_t<Dim, QS, std::conditional_t<ScalarKind == 0, double, float>>>>;

    // the seven (dim, qs) pairs that exist — 1d carries QS=1 only — for both
    // windows, in one scalar. This IS the family; it is written out rather
    // than generated so that it can be read.
    template <int ScalarKind>
    using fields_of_scalar = type_list<
        field_of<1, 1, ScalarKind, true>,  field_of<1, 1, ScalarKind, false>,

        field_of<2, 1, ScalarKind, true>,  field_of<2, 1, ScalarKind, false>,
        field_of<2, 2, ScalarKind, true>,  field_of<2, 2, ScalarKind, false>,
        field_of<2, 3, ScalarKind, true>,  field_of<2, 3, ScalarKind, false>,

        field_of<3, 1, ScalarKind, true>,  field_of<3, 1, ScalarKind, false>,
        field_of<3, 2, ScalarKind, true>,  field_of<3, 2, ScalarKind, false>,
        field_of<3, 3, ScalarKind, true>,  field_of<3, 3, ScalarKind, false>>;

    // double fields always; float fields only when the universe allows them
    template <typename Universe>
    struct field_family : type_list_cat<fields_of_scalar<0>,
                                         std::conditional_t<Universe::enable_float_fields, fields_of_scalar<1>, type_list<>>> {
        using base_list = typename type_list_cat<fields_of_scalar<0>,
            std::conditional_t<Universe::enable_float_fields, fields_of_scalar<1>, type_list<>>>::type;
    };

    // ---- the species family ----
    //
    // Each entry is keyed by the species' one TYPE invariant, its dimension.
    // Only the ENABLED dims are listed, so a disabled dim simply has no key.
    template <typename Universe, int D>
    using species_dim_entry = std::conditional_t<Universe::has_dim(D),
                                                 type_list<keyed_type<D, species<Universe, D>>>, type_list<>>;
    template <typename Universe>
    struct species_family : type_list_cat<typename type_list_cat<species_dim_entry<Universe, 1>,
                                                                  species_dim_entry<Universe, 2>>::type,
                                          species_dim_entry<Universe, 3>>::type {
        using base_list = typename type_list_cat<
            typename type_list_cat<species_dim_entry<Universe, 1>, species_dim_entry<Universe, 2>>::type,
            species_dim_entry<Universe, 3>>::type;
    };

} // namespace psum_runtime

#endif
