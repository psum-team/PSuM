#ifndef PSUM_RUNTIME_UNIVERSE_HPP
#define PSUM_RUNTIME_UNIVERSE_HPP

// The universe: compile-time parameters of one runtime flavor.
//   PosScalar / VelScalar   position and velocity scalars (may differ)
//   EOrder / BOrder         interpolation orders in push kernels
//   DimMask                 which particle dims are supported (1d/2d/3d)
//   AutoWrap                whether particle positions wrap around automatically
//   EnableFloatFields       whether float fields are supported (default: no)
//   ExtraBindings           particle property bindings beyond pos/vel
// The library-wide defaults come from compile definitions.

#include <psum/psum.hpp>

namespace psum_runtime {

    using namespace psum::prelude;
    namespace pt = psum::property;

    template <typename PosScalar, typename VelScalar, int EOrder, int BOrder, int DimMask, bool AutoWrap, bool EnableFloatFields,
              typename... ExtraBindings>
    struct psum_universe {
        static constexpr int e_order = EOrder;
        static constexpr int b_order = BOrder;
        static constexpr bool auto_wrap = AutoWrap;
        // enable-float-fields: FIELDS only. When off (default) the field create/dispatch layers carry double fields only.
        static constexpr bool enable_float_fields = EnableFloatFields;
        // dim mask: which particle/grid dims this build supports.
        // 8 (default) = all of 1d/2d/3d; 1..7 = bitmask (bit0 1d, bit1 2d, bit2 3d). 
        // Unsupported dims are never instantiated and make compile times shorter.
        static constexpr int dim_mask = DimMask;
        static constexpr bool has_dim(int d) {
            return DimMask >= 8 || ((DimMask >> (d - 1)) & 1) != 0;
        }

        template <int Dim>
        using particle = psum::tag::tagged_struct<
            psum::tag::tag_bind<pt::position, Eigen::Vector<PosScalar, Dim>>,
            psum::tag::tag_bind<pt::velocity, Eigen::Vector<VelScalar, Dim>>,
            ExtraBindings...>;
        
    };

    // Use compile-time definitions to configure the universe.
    // If none are defined, the defaults are used.
    #ifndef PSUM_POS_SCALAR
    #define PSUM_POS_SCALAR double
    #endif
    #ifndef PSUM_VEL_SCALAR
    #define PSUM_VEL_SCALAR double
    #endif
    #ifndef PSUM_E_ORDER
    #define PSUM_E_ORDER 1
    #endif
    #ifndef PSUM_B_ORDER
    #define PSUM_B_ORDER 1
    #endif
    #ifndef PSUM_DIM_MASK
    #define PSUM_DIM_MASK 8
    #endif
    #ifndef PSUM_AUTO_WRAP
    #define PSUM_AUTO_WRAP 0
    #endif
    #ifndef PSUM_ENABLE_FLOAT_FIELDS
    #define PSUM_ENABLE_FLOAT_FIELDS 0
    #endif

    // configured_universe is the universe used for psum_engine and the runtime API. It is configured at compile time via the PSUM_* macros above.
    using configured_universe =
        psum_universe<PSUM_POS_SCALAR, PSUM_VEL_SCALAR, PSUM_E_ORDER, PSUM_B_ORDER,
                      PSUM_DIM_MASK, PSUM_AUTO_WRAP != 0,
                      PSUM_ENABLE_FLOAT_FIELDS != 0>;
    static_assert(PSUM_DIM_MASK >= 1 && PSUM_DIM_MASK <= 8,
                  "PSUM_DIM_MASK must be in 1..8 (8 = all dims, 1..7 bitmask)");

}

#endif
