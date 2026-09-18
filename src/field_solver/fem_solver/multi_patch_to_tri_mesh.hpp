#ifndef PSUM_FIELD_SOLVER_FEM_MULTI_PATCH_TO_TRI_MESH_HPP
#define PSUM_FIELD_SOLVER_FEM_MULTI_PATCH_TO_TRI_MESH_HPP

#include "../../field/multi_patch/multi_patch_grid.hpp"
#include "../../field/multi_patch/duplicate_map.hpp"
#include "tri_mesh.hpp"
#include <vector>
#include <array>

namespace psum {

namespace field_solver {

namespace fem_solver {

using node_idx2 = field::simple_grid<2>::node_index;
using cell_idx2 = field::simple_grid<2>::cell_index;

struct mpg_tri_mesh_result {
    tri_mesh mesh;
    field::multi_patch::duplicate_map<2> dm;
};

inline mpg_tri_mesh_result multi_patch_to_tri_mesh(const field::multi_patch::multi_patch_grid<2>& mpg) {
    field::multi_patch::duplicate_map<2> dm = field::multi_patch::build_duplicate_map(mpg);
    const auto& i2d = dm.index_to_dof;
    size_t n_dof = dm.n_dof;

    const auto& node_offsets = mpg.get_patch_node_offsets();
    const auto& res_exp = mpg.get_resolution_exponents();
    const auto& coarse = mpg.coarse_grid();
    int coarse_nx = coarse.template numCells<0>();
    int coarse_ny = coarse.template numCells<1>();

    for (int pj = 0; pj < coarse_ny; pj++) {
        for (int pi = 0; pi < coarse_nx; pi++) {
            int my_exp = res_exp[coarse.c2i(cell_idx2{std::array<int,2>{pi, pj}})];
            if (pi + 1 < coarse_nx) {
                int nb_exp = res_exp[coarse.c2i(cell_idx2{std::array<int,2>{pi + 1, pj}})];
                if (std::abs(nb_exp - my_exp) > 1)
                    throw std::runtime_error("multi_patch_to_tri_mesh: adjacent patches must differ in resolution exponent by at most 1");
            }
            if (pj + 1 < coarse_ny) {
                int nb_exp = res_exp[coarse.c2i(cell_idx2{std::array<int,2>{pi, pj + 1}})];
                if (std::abs(nb_exp - my_exp) > 1)
                    throw std::runtime_error("multi_patch_to_tri_mesh: adjacent patches must differ in resolution exponent by at most 1");
            }
        }
    }

    std::vector<Eigen::Vector2d> nodes(n_dof);
    for (size_t p = 0; p < mpg.patches_count(); p++) {
        auto pg = mpg.patch_grid(p);
        int n = pg.template numCells<0>();
        for (int j = 0; j <= n; j++) {
            for (int i = 0; i <= n; i++) {
                node_idx2 ni{std::array<int,2>{i, j}};
                size_t global_index = node_offsets[p] + pg.n2i(ni);
                size_t dof = i2d[global_index];
                auto pos = pg.nodePosition(ni);
                nodes[dof] = Eigen::Vector2d(pos[0], pos[1]);
            }
        }
    }

    auto get_neighbor_exp = [&](int pi, int pj) -> int {
        if (pi < 0 || pi >= coarse_nx || pj < 0 || pj >= coarse_ny)
            return -1;
        return res_exp[coarse.c2i(cell_idx2{std::array<int,2>{pi, pj}})];
    };

    std::vector<std::array<size_t, 3>> elements;

    for (size_t p = 0; p < mpg.patches_count(); p++) {
        auto pg = mpg.patch_grid(p);
        int n = pg.template numCells<0>();
        int my_exp = res_exp[p];

        auto cidx = coarse.i2c(p);
        int pi = cidx.indices[0];
        int pj = cidx.indices[1];

        bool right_finer  = get_neighbor_exp(pi + 1, pj) == my_exp + 1;
        bool top_finer    = get_neighbor_exp(pi, pj + 1) == my_exp + 1;
        bool left_finer   = get_neighbor_exp(pi - 1, pj) == my_exp + 1;
        bool bottom_finer = get_neighbor_exp(pi, pj - 1) == my_exp + 1;

        auto get_dof = [&](int i, int j) -> size_t {
            node_idx2 ni{std::array<int,2>{i, j}};
            return i2d[node_offsets[p] + pg.n2i(ni)];
        };

        auto get_right_mid_dof = [&](int cj) -> size_t {
            size_t rpid = coarse.c2i(cell_idx2{std::array<int,2>{pi + 1, pj}});
            auto rpg = mpg.patch_grid(rpid);
            node_idx2 ni{std::array<int,2>{0, 2 * cj + 1}};
            return i2d[node_offsets[rpid] + rpg.n2i(ni)];
        };

        auto get_top_mid_dof = [&](int ci) -> size_t {
            size_t tpid = coarse.c2i(cell_idx2{std::array<int,2>{pi, pj + 1}});
            auto tpg = mpg.patch_grid(tpid);
            node_idx2 ni{std::array<int,2>{2 * ci + 1, 0}};
            return i2d[node_offsets[tpid] + tpg.n2i(ni)];
        };

        auto get_left_mid_dof = [&](int cj) -> size_t {
            size_t lpid = coarse.c2i(cell_idx2{std::array<int,2>{pi - 1, pj}});
            auto lpg = mpg.patch_grid(lpid);
            int rn = lpg.template numCells<0>();
            node_idx2 ni{std::array<int,2>{rn, 2 * cj + 1}};
            return i2d[node_offsets[lpid] + lpg.n2i(ni)];
        };

        auto get_bottom_mid_dof = [&](int ci) -> size_t {
            size_t bpid = coarse.c2i(cell_idx2{std::array<int,2>{pi, pj - 1}});
            auto bpg = mpg.patch_grid(bpid);
            int rn = bpg.template numCells<1>();
            node_idx2 ni{std::array<int,2>{2 * ci + 1, rn}};
            return i2d[node_offsets[bpid] + bpg.n2i(ni)];
        };

        for (int cj = 0; cj < n; cj++) {
            for (int ci = 0; ci < n; ci++) {
                size_t a = get_dof(ci, cj);
                size_t b = get_dof(ci + 1, cj);
                size_t c = get_dof(ci + 1, cj + 1);
                size_t d = get_dof(ci, cj + 1);

                bool rh = (ci == n - 1) && right_finer;
                bool th = (cj == n - 1) && top_finer;
                bool lh = (ci == 0) && left_finer;
                bool bh = (cj == 0) && bottom_finer;

                int nh = (rh ? 1 : 0) + (th ? 1 : 0) + (lh ? 1 : 0) + (bh ? 1 : 0);

                if (nh == 0) {
                    elements.push_back({a, b, d});
                    elements.push_back({b, c, d});
                } else if (nh == 1) {
                    if (rh) {
                        size_t m = get_right_mid_dof(cj);
                        elements.push_back({a, b, m});
                        elements.push_back({a, m, d});
                        elements.push_back({d, m, c});
                    } else if (th) {
                        size_t m = get_top_mid_dof(ci);
                        elements.push_back({a, b, m});
                        elements.push_back({a, m, d});
                        elements.push_back({b, c, m});
                    } else if (lh) {
                        size_t m = get_left_mid_dof(cj);
                        elements.push_back({a, b, m});
                        elements.push_back({m, b, c});
                        elements.push_back({m, c, d});
                    } else {
                        size_t m = get_bottom_mid_dof(ci);
                        elements.push_back({a, m, d});
                        elements.push_back({m, b, d});
                        elements.push_back({b, c, d});
                    }
                } else {
                    if (rh && th) {
                        size_t mr = get_right_mid_dof(cj);
                        size_t mt = get_top_mid_dof(ci);
                        elements.push_back({a, b, mr});
                        elements.push_back({a, mr, mt});
                        elements.push_back({a, mt, d});
                        elements.push_back({mr, c, mt});
                    } else if (rh && bh) {
                        size_t mr = get_right_mid_dof(cj);
                        size_t mb = get_bottom_mid_dof(ci);
                        elements.push_back({a, mb, d});
                        elements.push_back({mb, mr, d});
                        elements.push_back({mb, b, mr});
                        elements.push_back({mr, c, d});
                    } else if (lh && th) {
                        size_t ml = get_left_mid_dof(cj);
                        size_t mt = get_top_mid_dof(ci);
                        elements.push_back({a, b, ml});
                        elements.push_back({b, mt, ml});
                        elements.push_back({b, c, mt});
                        elements.push_back({ml, mt, d});
                    } else {
                        size_t ml = get_left_mid_dof(cj);
                        size_t mb = get_bottom_mid_dof(ci);
                        elements.push_back({a, mb, ml});
                        elements.push_back({mb, b, c});
                        elements.push_back({ml, mb, c});
                        elements.push_back({ml, c, d});
                    }
                }
            }
        }
    }

    return mpg_tri_mesh_result{tri_mesh(nodes, elements), std::move(dm)};
}

// to do: device version
inline void scatter_to_mpg(
    const field::multi_patch::duplicate_map<2>& dm,
    const double* mesh_data,
    double* mpg_data)
{
    for (size_t i = 0; i < dm.n_index; i++)
        mpg_data[i] = mesh_data[dm.index_to_dof[i]];
}

inline void gather_from_mpg(
    const field::multi_patch::duplicate_map<2>& dm,
    const double* mpg_data,
    double* mesh_data)
{
    for (size_t i = 0; i < dm.n_index; i++)
        mesh_data[dm.index_to_dof[i]] = mpg_data[i];
}

}

}

}

#endif
