#include <psum/field.hpp>
#include <field/multi_patch/duplicate_map.hpp>
#include <field/multi_patch/node_volume_field.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace psum::field;
using namespace multi_patch;

sycl::queue& get_queue() {
    static sycl::queue q;
    return q;
}

void test_uniform_density_2d() {
    std::cout << "--- Test: Uniform density via add_back + merge + divide by volume (2D) ---" << std::endl;

    auto& q = get_queue();

    std::vector<int> exponents(64);
    for (int iy = 0; iy < 8; iy++) {
        for (int ix = 0; ix < 8; ix++) {
            int ring = std::min({ix, 7 - ix, iy, 7 - iy});
            exponents[iy * 8 + ix] = 2 + std::min(ring, 1);
        }
    }

    using Field = device_multi_patch_field<2, var_loc::nodeCentered, double>;

    Field vol_field(
        multi_patch_grid<2>(
            q,
            {0.0, 0.0},
            {8.0, 8.0},
            {8, 8},
            exponents
        )
    );
    const auto& mpg = vol_field.getGrid();

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    {
        Eigen::VectorXd patch_vol(dm.n_index);
        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            size_t offset = mpg.get_patch_node_offsets()[p];
            const auto& cell_num = pg.get_cell_num();
            const auto& deltas = pg.get_deltas();
            size_t patch_nodes = pg.contentSize(var_loc::nodeCentered);

            for (size_t i = 0; i < patch_nodes; i++) {
                auto ni = pg.i2n(i);
                double v = 1.0;
                for (int d = 0; d < 2; d++) {
                    if (ni.indices[d] == 0 || ni.indices[d] == cell_num[d])
                        v *= deltas[d] * 0.5;
                    else
                        v *= deltas[d];
                }
                patch_vol[offset + i] = v;
            }
        }

        Eigen::VectorXd merged_vol = S * patch_vol;
        std::vector<double> host_vol(dm.n_index);
        for (size_t i = 0; i < dm.n_index; i++)
            host_vol[i] = merged_vol(i);
        vol_field.copy(host_vol);
    }

    Field density_field(
        multi_patch_grid<2>(
            q,
            {0.0, 0.0},
            {8.0, 8.0},
            {8, 8},
            exponents
        )
    );
    density_field.setZero();

    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist_x(0.0, 8.0);
    std::uniform_real_distribution<double> dist_y(0.0, 8.0);

    int n_particles = 80000000;
    std::vector<Eigen::RowVector<double, 2>> positions;
    positions.reserve(n_particles);
    while ((int)positions.size() < n_particles) {
        double x = dist_x(rng);
        double y = dist_y(rng);
        Eigen::RowVector<double, 2> pos(x, y);
        if (density_field.getGrid().inGrid(pos)) {
            positions.push_back(pos);
        }
    }

    size_t n_in = positions.size();
    std::cout << "  particles in grid: " << n_in << std::endl;

    device_array<Eigen::RowVector<double, 2>> pos_dev(q, positions);

    q.submit([&](sycl::handler& h) {
        auto acc = density_field.get_access(h);
        auto p_acc = pos_dev.get_access(h);
        h.parallel_for(sycl::range<1>(positions.size()), [=](sycl::id<1> idx) {
            double w = 1.0;
            add_back(p_acc[idx], w, acc);
        });
    }).wait();

    auto density_host = density_field.getContent().to_host();

    Eigen::VectorXd density_vec(dm.n_index);
    for (size_t i = 0; i < dm.n_index; i++)
        density_vec(i) = density_host[i];
    Eigen::VectorXd synced = S * density_vec;

    auto vol_host = vol_field.getContent().to_host();

    std::vector<double> result_host(dm.n_index);
    for (size_t i = 0; i < dm.n_index; i++)
        result_host[i] = synced(i) / vol_host[i];

    density_field.copy(result_host);
    density_field.plot("uniform_density_2d.plt", "density", 0.0);
    std::cout << "  -> uniform_density_2d.plt" << std::endl;

    double expected_density = (double)n_in / 64.0;
    double l1_sum = 0.0, l2_sum = 0.0, linf = 0.0;
    double l1_ref = 0.0;
    for (size_t i = 0; i < dm.n_index; i++) {
        double exact_val = expected_density;
        double diff = std::abs(result_host[i] - exact_val);
        l1_sum += diff * vol_host[i];
        l2_sum += diff * diff * vol_host[i];
        if (diff > linf) linf = diff;
        l1_ref += std::abs(exact_val) * vol_host[i];
    }
    std::cout << "  expected density = " << expected_density << std::endl;
    std::cout << "  L1 relative error = " << std::scientific << std::setprecision(4)
              << l1_sum / l1_ref << std::endl;
    std::cout << "  Linf error        = " << linf << std::endl;
}

int main() {
    test_uniform_density_2d();
    return 0;
}
