#ifndef PSUM_FIELD_MULTI_PATCH_DUPLICATE_MAP_HPP
#define PSUM_FIELD_MULTI_PATCH_DUPLICATE_MAP_HPP

#include "multi_patch_grid.hpp"
#include <Eigen/Sparse>
#include <map>
#include <array>
#include <cmath>

namespace psum {

namespace field {

namespace multi_patch {

    template <int Dim>
    struct duplicate_map {
        size_t n_index;
        size_t n_dof;
        std::vector<size_t> index_to_dof;
        Eigen::SparseMatrix<double> P;
    };

    template <int Dim>
    duplicate_map<Dim> build_duplicate_map(const multi_patch_grid<Dim>& mpg) {
        duplicate_map<Dim> result;

        size_t total_nodes = mpg.contentSize(var_loc::nodeCentered);
        result.n_index = total_nodes;

        const auto& lower = mpg.get_lower_bounds();
        const auto& patch_spans = mpg.get_patch_spans();

        std::array<double, Dim> min_delta;
        for (int d = 0; d < Dim; d++)
            min_delta[d] = patch_spans[d];

        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            const auto& deltas = pg.get_deltas();
            for (int d = 0; d < Dim; d++) {
                if (deltas[d] < min_delta[d])
                    min_delta[d] = deltas[d];
            }
        }

        using Key = std::array<int, Dim>;

        struct key_less {
            bool operator()(const Key& a, const Key& b) const {
                for (int d = 0; d < Dim; d++) {
                    if (a[d] < b[d]) return true;
                    if (a[d] > b[d]) return false;
                }
                return false;
            }
        };

        std::map<Key, size_t, key_less> key_to_dof;
        std::vector<size_t>& index_to_dof = result.index_to_dof;
        index_to_dof.resize(total_nodes);

        size_t next_dof = 0;

        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            size_t patch_nodes = pg.contentSize(var_loc::nodeCentered);
            size_t offset = mpg.get_patch_node_offsets()[p];

            for (size_t i = 0; i < patch_nodes; i++) {
                auto ni = pg.i2n(i);
                auto pos = pg.nodePosition(ni);

                Key key;
                for (int d = 0; d < Dim; d++)
                    key[d] = static_cast<int>(std::llround((pos[d] - lower[d]) / min_delta[d]));

                auto [it, inserted] = key_to_dof.try_emplace(key, next_dof);
                if (inserted)
                    next_dof++;

                index_to_dof[offset + i] = it->second;
            }
        }

        result.n_dof = next_dof;

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(total_nodes);
        for (size_t i = 0; i < total_nodes; i++)
            triplets.emplace_back(static_cast<long long>(i),
                                  static_cast<long long>(index_to_dof[i]),
                                  1.0);

        Eigen::SparseMatrix<double> P(total_nodes, next_dof);
        P.setFromTriplets(triplets.begin(), triplets.end());

        result.P = std::move(P);

        return result;
    }

    inline Eigen::SparseMatrix<double> build_merge_matrix(const Eigen::SparseMatrix<double>& P) {
        return Eigen::SparseMatrix<double>(P * P.transpose());
    }

    template <int Dim>
    Eigen::SparseMatrix<double> build_merge_matrix(const duplicate_map<Dim>& dm) {
        return build_merge_matrix(dm.P);
    }

    template <int Dim>
    Eigen::SparseMatrix<double> build_gather_matrix(const duplicate_map<Dim>& dm) {
        std::vector<size_t> dof_to_index(dm.n_dof, 0);
        for (size_t i = 0; i < dm.n_index; i++) {
            size_t dof = dm.index_to_dof[i];
            if (dof < dof_to_index.size()) dof_to_index[dof] = i;
        }

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(dm.n_dof);
        for (size_t dof = 0; dof < dm.n_dof; dof++)
            triplets.emplace_back(static_cast<long long>(dof),
                                  static_cast<long long>(dof_to_index[dof]),
                                  1.0);

        Eigen::SparseMatrix<double> G(dm.n_dof, dm.n_index);
        G.setFromTriplets(triplets.begin(), triplets.end());
        return G;
    }

}

}

}

#endif
