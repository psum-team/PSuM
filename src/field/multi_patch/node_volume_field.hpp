#ifndef PSUM_FIELD_MULTI_PATCH_NODE_VOLUME_FIELD_HPP
#define PSUM_FIELD_MULTI_PATCH_NODE_VOLUME_FIELD_HPP

#include "duplicate_map.hpp"
#include "device_multi_patch_field.hpp"

namespace psum {

namespace field {

namespace multi_patch {

    template <int Dim>
    device_multi_patch_field<Dim, var_loc::nodeCentered, double>
    build_node_volume_field(multi_patch_grid<Dim>&& mpg_rv) {
        const auto& mpg = mpg_rv;

        auto dm = build_duplicate_map(mpg);
        auto S = build_merge_matrix(dm);

        size_t n_index = dm.n_index;

        Eigen::VectorXd patch_vol(n_index);
        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            size_t offset = mpg.get_patch_node_offsets()[p];
            const auto& cell_num = pg.get_cell_num();
            const auto& deltas = pg.get_deltas();
            size_t patch_nodes = pg.contentSize(var_loc::nodeCentered);

            for (size_t i = 0; i < patch_nodes; i++) {
                auto ni = pg.i2n(i);
                double vol = 1.0;
                for (int d = 0; d < Dim; d++) {
                    if (ni.indices[d] == 0 || ni.indices[d] == cell_num[d])
                        vol *= deltas[d] * 0.5;
                    else
                        vol *= deltas[d];
                }
                patch_vol[offset + i] = vol;
            }
        }

        Eigen::VectorXd merged_vol = S * patch_vol;

        std::vector<double> host_vol(n_index);
        for (size_t i = 0; i < n_index; i++)
            host_vol[i] = merged_vol(i);

        device_multi_patch_field<Dim, var_loc::nodeCentered, double> field(std::move(mpg_rv));
        field.copy(host_vol);

        return field;
    }

}

}

}

#endif
