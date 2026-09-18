#ifndef PSUM_PARTICLE_BOUNDARY_GEOMETRY_HPP
#define PSUM_PARTICLE_BOUNDARY_GEOMETRY_HPP

#include <sycl/sycl.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace psum {

namespace particle_boundary {

namespace geometry {

    constexpr double finite_small = 1e-20;

    struct triangle_data {
        double p1x, p1y, p1z;
        double p2x, p2y, p2z;
        double p3x, p3y, p3z;
        double nx, ny, nz;
    };

    // grid cooordinates
    inline std::tuple<bool, bool, int, int> trace_to_idxs_fast(int I, int J, int K, double x1, double y1, double z1, double x2, double y2, double z2) {
        bool inside_1 = (x1 >= 0 && x1 < I && y1 >= 0 && y1 < J && z1 >= 0 && z1 < K);
        bool inside_2 = (x2 >= 0 && x2 < I && y2 >= 0 && y2 < J && z2 >= 0 && z2 < K);
        int i_1 = sycl::floor(x1);
        int j_1 = sycl::floor(y1);
        int k_1 = sycl::floor(z1);
        int idx_1 = (i_1 * J + j_1) * K + k_1;
        int i_2 = sycl::floor(x2);
        int j_2 = sycl::floor(y2);
        int k_2 = sycl::floor(z2);
        int idx_2 = (i_2 * J + j_2) * K + k_2;
        return {inside_1, inside_2, idx_1, idx_2};
    }

    // grid cooordinates
    struct trace_to_idxs_state {
        int I, J, K;
        double p1x, p1y, p1z;
        double dirx, diry, dirz;
        int I_here, J_here, K_here;
        double t;
        int k_x, k_y, k_z;
        int b_x, b_y, b_z;
        int last_idx;
        bool finished;
    };

    inline int trace_to_idxs_begin(int I, int J, int K, double x1, double y1, double z1, double x2, double y2, double z2, trace_to_idxs_state& state) {
        state.I = I;
        state.J = J;
        state.K = K;
        state.p1x = x1;
        state.p1y = y1;
        state.p1z = z1;
        state.dirx = x2 - x1;
        state.diry = y2 - y1;
        state.dirz = z2 - z1;
        state.t = 0;
        state.finished = false;

        state.k_x = 1; state.b_x = 0;
        state.k_y = 1; state.b_y = 0;
        state.k_z = 1; state.b_z = 0;

        if(state.dirx < 0) {
            state.k_x = -1; state.b_x = I - 1;
            state.p1x = I - state.p1x; state.dirx *= -1;
        }
        if(state.diry < 0) {
            state.k_y = -1; state.b_y = J - 1;
            state.p1y = J - state.p1y; state.diry *= -1;
        }
        if(state.dirz < 0) {
            state.k_z = -1; state.b_z = K - 1;
            state.p1z = K - state.p1z; state.dirz *= -1;
        }

        if(state.dirx == 0) state.dirx = finite_small;
        if(state.diry == 0) state.diry = finite_small;
        if(state.dirz == 0) state.dirz = finite_small;

        double t_vec0 = -state.p1x / state.dirx;
        double t_vec1 = -state.p1y / state.diry;
        double t_vec2 = -state.p1z / state.dirz;
        
        double max_t = t_vec0;
        if(t_vec1 > max_t) max_t = t_vec1;
        if(t_vec2 > max_t) max_t = t_vec2;

        if(max_t >= 1) {
            state.finished = true;
            return -1;
        }
        if(max_t < 0) state.t = 0;
        else state.t = max_t;

        state.p1x += state.t * state.dirx;
        state.p1y += state.t * state.diry;
        state.p1z += state.t * state.dirz;

        state.I_here = sycl::floor(state.p1x);
        state.J_here = sycl::floor(state.p1y);
        state.K_here = sycl::floor(state.p1z);

        if (state.I_here >= I || state.J_here >= J || state.K_here >= K) {
            state.finished = true;
            return -1;
        }

        if(max_t > 0) {
            if(state.t == t_vec0) state.I_here = 0;
            if(state.t == t_vec1) state.J_here = 0;
            if(state.t == t_vec2) state.K_here = 0;
        }

        int newIdx = (state.k_x * state.I_here + state.b_x) * J * K + 
                     (state.k_y * state.J_here + state.b_y) * K + 
                     (state.k_z * state.K_here + state.b_z);
        state.last_idx = newIdx;
        return newIdx;
    }

    inline bool trace_to_idxs_valid(const trace_to_idxs_state& state) {
        return !state.finished;
    }

    inline int trace_to_idxs_next(trace_to_idxs_state& state) {
        if (state.finished) {
            return -1;
        }

        double t_vec0 = (state.I_here + 1 - state.p1x) / state.dirx;
        double t_vec1 = (state.J_here + 1 - state.p1y) / state.diry;
        double t_vec2 = (state.K_here + 1 - state.p1z) / state.dirz;

        if(t_vec0 < t_vec1) {
            if(t_vec0 < t_vec2) {
                state.t += t_vec0;
                state.p1x += t_vec0 * state.dirx;
                state.p1y += t_vec0 * state.diry;
                state.p1z += t_vec0 * state.dirz;
                state.I_here++;
            } else {
                state.t += t_vec2;
                state.p1x += t_vec2 * state.dirx;
                state.p1y += t_vec2 * state.diry;
                state.p1z += t_vec2 * state.dirz;
                state.K_here++;
            }
        } else {
            if(t_vec1 < t_vec2) {
                state.t += t_vec1;
                state.p1x += t_vec1 * state.dirx;
                state.p1y += t_vec1 * state.diry;
                state.p1z += t_vec1 * state.dirz;
                state.J_here++;
            } else {
                state.t += t_vec2;
                state.p1x += t_vec2 * state.dirx;
                state.p1y += t_vec2 * state.diry;
                state.p1z += t_vec2 * state.dirz;
                state.K_here++;
            }
        }

        if(!(state.t < 1)) {
            state.finished = true;
            return -1;
        }

        if(state.I_here >= state.I || state.J_here >= state.J || state.K_here >= state.K) {
            state.finished = true;
            return -1;
        }

        int newIdx = (state.k_x * state.I_here + state.b_x) * state.J * state.K +
                     (state.k_y * state.J_here + state.b_y) * state.K +
                     (state.k_z * state.K_here + state.b_z);

        if(newIdx != state.last_idx) {
            state.last_idx = newIdx;
            return newIdx;
        }
        return trace_to_idxs_next(state);
    }

    inline bool line_seg_tri_intersect_test(
        double x1, double y1, double z1,
        double x2, double y2, double z2,
        double x3, double y3, double z3,
        double p1x, double p1y, double p1z,
        double p2x, double p2y, double p2z,
        double& lambda
    ) {
        double edge1x = x2 - x1;
        double edge1y = y2 - y1;
        double edge1z = z2 - z1;

        double edge2x = x3 - x1;
        double edge2y = y3 - y1;
        double edge2z = z3 - z1;

        double dirx = p2x - p1x;
        double diry = p2y - p1y;
        double dirz = p2z - p1z;

        double h_x = diry * edge2z - dirz * edge2y;
        double h_y = dirz * edge2x - dirx * edge2z;
        double h_z = dirx * edge2y - diry * edge2x;

        double a = edge1x * h_x + edge1y * h_y + edge1z * h_z;

        if(sycl::fabs(a) < finite_small) {
            lambda = 0.0;
            return false;
        }

        double f = 1.0 / a;

        double ox = p1x - x1;
        double oy = p1y - y1;
        double oz = p1z - z1;

        double u = f * (ox * h_x + oy * h_y + oz * h_z);

        if(u < 0.0 || u > 1.0) {
            lambda = 0.0;
            return false;
        }

        double q_x = oy * edge1z - oz * edge1y;
        double q_y = oz * edge1x - ox * edge1z;
        double q_z = ox * edge1y - oy * edge1x;

        double v = f * (dirx * q_x + diry * q_y + dirz * q_z);

        if(v < 0.0 || u + v > 1.0) {
            lambda = 0.0;
            return false;
        }

        lambda = f * (edge2x * q_x + edge2y * q_y + edge2z * q_z);

        return (lambda > 0.0 && lambda <= 1.0);
    }

    inline bool line_seg_box_overlap_test(
        double box_x0, double box_y0, double box_z0,
        double box_x1, double box_y1, double box_z1,
        double p1x, double p1y, double p1z,
        double p2x, double p2y, double p2z
    ) {
        double dx = box_x1 - box_x0;
        double dy = box_y1 - box_y0;
        double dz = box_z1 - box_z0;

        double nx1 = (p1x - box_x0) / dx;
        double ny1 = (p1y - box_y0) / dy;
        double nz1 = (p1z - box_z0) / dz;
        double nx2 = (p2x - box_x0) / dx;
        double ny2 = (p2y - box_y0) / dy;
        double nz2 = (p2z - box_z0) / dz;

        trace_to_idxs_state state;
        int first_idx = trace_to_idxs_begin(1, 1, 1, nx1, ny1, nz1, nx2, ny2, nz2, state);
        return (first_idx >= 0);
    }

    inline bool box_tri_intersect_test(
        double x1, double y1, double z1,
        double x2, double y2, double z2,
        double x3, double y3, double z3,
        double box_x0, double box_y0, double box_z0,
        double box_x1, double box_y1, double box_z1
    ) {
        bool ans = false;
        double xs[2] = {box_x0, box_x1};
        double ys[2] = {box_y0, box_y1};
        double zs[2] = {box_z0, box_z1};
        double temp_lambda;

        for (int i = 0; i < 4; i++) {
            if (line_seg_tri_intersect_test(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                          xs[0], ys[i / 2], zs[i % 2],
                                          xs[1], ys[i / 2], zs[i % 2], temp_lambda)) {
                return true;
            }
        }

        for (int i = 0; i < 4; i++) {
            if (line_seg_tri_intersect_test(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                          xs[i / 2], ys[0], zs[i % 2],
                                          xs[i / 2], ys[1], zs[i % 2], temp_lambda)) {
                return true;
            }
        }

        for (int i = 0; i < 4; i++) {
            if (line_seg_tri_intersect_test(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                          xs[i / 2], ys[i % 2], zs[0],
                                          xs[i / 2], ys[i % 2], zs[1], temp_lambda)) {
                return true;
            }
        }

        if (line_seg_box_overlap_test(box_x0, box_y0, box_z0, box_x1, box_y1, box_z1, x1, y1, z1, x2, y2, z2)) {
            return true;
        }
        if (line_seg_box_overlap_test(box_x0, box_y0, box_z0, box_x1, box_y1, box_z1, x1, y1, z1, x3, y3, z3)) {
            return true;
        }
        if (line_seg_box_overlap_test(box_x0, box_y0, box_z0, box_x1, box_y1, box_z1, x3, y3, z3, x2, y2, z2)) {
            return true;
        }

        return false;
    }

    inline bool box_tri_overlap_test(
        double x1, double y1, double z1,
        double x2, double y2, double z2,
        double x3, double y3, double z3,
        double box_x0, double box_y0, double box_z0,
        double box_x1, double box_y1, double box_z1
    ) {
        auto contain = [&](double x, double y, double z) -> bool {
            return (x > box_x0 && x < box_x1) && (y > box_y0 && y < box_y1) && (z > box_z0 && z < box_z1);
        };

        if (contain(x1, y1, z1) || contain(x2, y2, z2) || contain(x3, y3, z3)) {
            return true;
        }

        return box_tri_intersect_test(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                   box_x0, box_y0, box_z0, box_x1, box_y1, box_z1);
    }

    inline void compute_udf(int I, int J, int K, const std::vector<bool>& occupancy, std::vector<float>& udf, int udf_k = 6, int max_loop = 200) {

        if (occupancy.size() != I * J * K) {
            throw std::runtime_error("Invalid occupancy size");
        }

        // simple case: all cells are not occupied
        bool full_empty = true;
        for (int i = 0; i < I * J * K; ++i) {
            if (occupancy[i]) {
                full_empty = false;
                break;
            }
        }
        if (full_empty) {
            udf.resize(I * J * K);
            int max_udf = std::min({I, J, K}) - 1;
            for (int i = 0; i < I * J * K; ++i) {
                udf[i] = static_cast<float>(max_udf);
            }
            return;
        }

        udf.resize(I * J * K);

#pragma omp parallel for
        for (int ci = 0; ci < I; ++ci) {
            for (int cj = 0; cj < J; ++cj) {
                for (int ck = 0; ck < K; ++ck) {
                    int cell_idx = ci * J * K + cj * K + ck;
                    if (occupancy[cell_idx]) {
                        udf[cell_idx] = 0.0f;
                    } else {
                        float min_dist = 1e30f;
                        for (int di = -udf_k; di <= udf_k; ++di) {
                            for (int dj = -udf_k; dj <= udf_k; ++dj) {
                                for (int dk = -udf_k; dk <= udf_k; ++dk) {
                                    int ni = ci + di;
                                    int nj = cj + dj;
                                    int nk = ck + dk;
                                    if (ni >= 0 && ni < I && nj >= 0 && nj < J && nk >= 0 && nk < K) {
                                        int neighbor_idx = ni * J * K + nj * K + nk;
                                        if (occupancy[neighbor_idx]) {
                                            int lb_i = std::max(abs(di)-1, 0);
                                            int lb_j = std::max(abs(dj)-1, 0);
                                            int lb_k = std::max(abs(dk)-1, 0);

                                            float lb_dist = sycl::sqrt(
                                                static_cast<float>(lb_i * lb_i + lb_j * lb_j + lb_k * lb_k)
                                            );

                                            if (lb_dist < min_dist) {
                                                min_dist = lb_dist;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        udf[cell_idx] = (min_dist < 1e29f) ? min_dist : static_cast<float>(udf_k - 1);
                        udf[cell_idx] = std::min(udf[cell_idx], static_cast<float>(udf_k - 1));
                    }
                }
            }
        }

        // increase udf value for the empty zones
        // like this:
        // 2 2 2        2 2 2
        // 2 2 2  -->   2 3 2
        // 2 2 2        2 2 2
        std::vector<float> new_udf;
        new_udf.resize(I * J * K);
        int loop_count = 0;
        while (true) {
            bool updated = false;

#pragma omp parallel for
            for (int ci = 0; ci < I; ++ci) {
                for (int cj = 0; cj < J; ++cj) {
                    for (int ck = 0; ck < K; ++ck) {
                        int cell_idx = ci * J * K + cj * K + ck;
                        new_udf[cell_idx] = udf[cell_idx];
                        if (udf[cell_idx] > 0.0f) {
                            float current = udf[cell_idx];
                            bool can_increase = true;
                            int count_valid = 0;
                            for (int di = -1; di <= 1; ++di) {
                                for (int dj = -1; dj <= 1; ++dj) {
                                    for (int dk = -1; dk <= 1; ++dk) {
                                        if (di == 0 && dj == 0 && dk == 0) continue;
                                        int ni = ci + di;
                                        int nj = cj + dj;
                                        int nk = ck + dk;
                                        if (ni >= 0 && ni < I && nj >= 0 && nj < J && nk >= 0 && nk < K) {
                                            int neighbor_idx = ni * J * K + nj * K + nk;
                                            count_valid++;
                                            if (udf[neighbor_idx] < current) {
                                                can_increase = false;
                                            }
                                        }
                                    }
                                }
                            }
                            if (can_increase && count_valid > 0) {
                                new_udf[cell_idx] = current + 1.0f;
                                updated = true;
                            }
                        }
                    }
                }
            }
            std::swap(udf, new_udf);
            if (!updated) break;
            loop_count++;
            if (loop_count > max_loop) break;
        }
    }

}

}

}

#endif
