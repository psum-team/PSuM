#include <psum/field.hpp>
#include <timer.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <string>

using namespace psum::field;
using namespace multi_patch;

sycl::queue& get_queue() {
    static sycl::queue q;
    return q;
}

void test_index_conversion() {
    std::cout << "--- Test: Index conversion ---" << std::endl;

    multi_patch_grid<2> mpg(
        get_queue(),
        {0.0, 0.0},
        {2.0, 2.0},
        {2, 2},
        std::vector<int>{2, 3, 4, 3}
    );

    auto approx = [](double a, double b) {
        return std::abs(a - b) < 1e-10;
    };

    for (size_t p = 0; p < mpg.patches_count(); p++) {
        auto pg = mpg.patch_grid(p);
        size_t cell_count = pg.contentSize(var_loc::cellCentered);

        for (size_t i = 0; i < cell_count; i++) {
            auto pos = pg.cellCenter(pg.i2c(i));
            auto pci = mpg.nC(pos);
            assert(pci.patch_id == p);

            size_t global_i = mpg.c2i(pci);
            auto pci_back = mpg.i2c(global_i);
            assert(pci_back.patch_id == pci.patch_id);
            assert(pci_back.indices == pci.indices);

            auto pos_back = pg.cellCenter(typename simple_grid<2>::cell_index{pci.indices});
            assert(approx(pos_back[0], pos[0]));
            assert(approx(pos_back[1], pos[1]));
        }

        size_t node_count = pg.contentSize(var_loc::nodeCentered);
        for (size_t i = 0; i < node_count; i++) {
            auto n = pg.i2n(i);
            bool on_upper = false;
            for (int d = 0; d < 2; d++) {
                if (n.indices[d] == pg.numCells<0>()) on_upper = true;
            }
            if (on_upper) continue;

            auto pos = pg.nodePosition(n);
            auto pni = mpg.nN(pos);
            assert(pni.patch_id == p);

            size_t global_i = mpg.n2i(pni);
            auto pni_back = mpg.i2n(global_i);
            assert(pni_back.patch_id == pni.patch_id);
            assert(pni_back.indices == pni.indices);
        }
    }

    assert(mpg.c2i({0, {0, 0}}) == 0);
    assert(mpg.c2i({0, {3, 3}}) == 15);
    assert(mpg.c2i({1, {0, 0}}) == 16);
    assert(mpg.c2i({1, {7, 7}}) == 79);
    assert(mpg.c2i({2, {0, 0}}) == 80);

    auto idx = mpg.i2c(0);
    assert(idx.patch_id == 0 && idx.indices[0] == 0 && idx.indices[1] == 0);
    idx = mpg.i2c(15);
    assert(idx.patch_id == 0);
    idx = mpg.i2c(16);
    assert(idx.patch_id == 1);
    idx = mpg.i2c(80);
    assert(idx.patch_id == 2);

    {
        Eigen::RowVector<double, 2> pos(0.5, 0.5);
        auto pci = mpg.nC(pos);
        assert(pci.patch_id == 0);
        assert(pci.indices[0] == 2);
        assert(pci.indices[1] == 2);

        size_t gi = mpg.nearest<grid_element::cell>(pos);
        assert(gi == mpg.c2i(pci));
    }

    {
        Eigen::RowVector<double, 2> pos(1.5, 0.5);
        auto pci = mpg.nC(pos);
        assert(pci.patch_id == 2);
    }

    {
        Eigen::RowVector<double, 2> pos(0.25, 0.25);
        auto pni = mpg.cN(pos);
        assert(pni.patch_id == 0);
    }

    {
        multi_patch_grid<1> mpg1d(
            get_queue(),
            {0.0},
            {3.0},
            {3},
            std::vector<int>{2, 3, 4}
        );
        Eigen::RowVector<double, 1> pos(0.5);
        auto pci = mpg1d.nC(pos);
        assert(pci.patch_id == 0);
        assert(pci.indices[0] == 2);

        pos[0] = 1.5;
        pci = mpg1d.nC(pos);
        assert(pci.patch_id == 1);

        pos[0] = 2.5;
        pci = mpg1d.nC(pos);
        assert(pci.patch_id == 2);
    }

    std::cout << "✅ Index conversion passed." << std::endl;
}

void test_interp_device() {
    std::cout << "--- Test: Device interpolation ---" << std::endl;

    auto& q = get_queue();
    auto approx = [](double a, double b) { return std::abs(a - b) < 1e-10; };

    using Field = device_multi_patch_field<2, var_loc::nodeCentered, double>;

    Field field(
        multi_patch_grid<2>(
            q,
            {0.0, 0.0},
            {2.0, 2.0},
            {2, 2},
            std::vector<int>{3, 3, 3, 3}
        )
    );

    const auto& mpg = field.getGrid();

    std::vector<double> host_data(mpg.contentSize(var_loc::nodeCentered), 0.0);
    for (size_t p = 0; p < mpg.patches_count(); p++) {
        auto pg = mpg.patch_grid(p);
        size_t off = mpg.get_patch_node_offsets()[p];
        for (size_t i = 0; i < pg.contentSize(var_loc::nodeCentered); i++) {
            auto pos = pg.nodePosition(pg.i2n(i));
            host_data[off + i] = pos[0] + pos[1];
        }
    }
    field.copy(host_data);

    {
        std::vector<double> res_vec(1, -999.0);
        device_array<double> res_dev(q, res_vec);

        field.for_each([&](sycl::handler& h) {
            auto acc = field.get_access(h);
            auto res_acc = res_dev.get_access(h);
            return [=](size_t idx, double& val) {
                if (idx == 0) {
                    Eigen::RowVector<double, 2> pos(0.5, 0.5);
                    res_acc[0] = interp(pos, acc);
                }
            };
        });

        res_vec = res_dev.to_host();
        assert(approx(res_vec[0], 1.0));
    }

    {
        std::vector<double> res_vec(1, -999.0);
        device_array<double> res_dev(q, res_vec);

        field.for_each([&](sycl::handler& h) {
            auto acc = field.get_access(h);
            auto res_acc = res_dev.get_access(h);
            return [=](size_t idx, double& val) {
                if (idx == 0) {
                    Eigen::RowVector<double, 2> pos(1.5, 0.5);
                    res_acc[0] = interp(pos, acc);
                }
            };
        });

        res_vec = res_dev.to_host();
        assert(approx(res_vec[0], 2.0));
    }

    std::cout << "  interp passed." << std::endl;

    {
        std::vector<double> res_vec(2, -999.0);
        device_array<double> res_dev(q, res_vec);

        field.for_each([&](sycl::handler& h) {
            auto acc = field.get_access(h);
            auto res_acc = res_dev.get_access(h);
            return [=](size_t idx, double& val) {
                if (idx == 0) {
                    Eigen::RowVector<double, 2> pos(0.5, 0.5);
                    auto grad = interp_diff(pos, acc);
                    res_acc[0] = grad[0];
                    res_acc[1] = grad[1];
                }
            };
        });

        res_vec = res_dev.to_host();
        assert(approx(res_vec[0], 1.0));
        assert(approx(res_vec[1], 1.0));
    }

    std::cout << "  interp_diff passed." << std::endl;

    {
        Field field2(
            multi_patch_grid<2>(
                q,
                {0.0, 0.0},
                {2.0, 2.0},
                {2, 2},
                std::vector<int>{3, 3, 3, 3}
            )
        );
        field2.setZero();

        field2.for_each([&](sycl::handler& h) {
            auto acc = field2.get_access(h);
            return [=](size_t idx, double& val) {
                if (idx == 0) {
                    Eigen::RowVector<double, 2> pos(0.5, 0.5);
                    double w = 7.0;
                    add_back(pos, w, acc);
                }
            };
        });

        auto result2 = field2.getContent().to_host();
        double total = 0;
        for (auto v : result2) total += v;
        assert(approx(total, 7.0));
    }

    std::cout << "  add_back passed." << std::endl;
    std::cout << "✅ Device interpolation tests passed." << std::endl;
}

void test_interp_accuracy() {
    std::cout << "--- Test: Interpolation accuracy ---" << std::endl;

    auto& q = get_queue();

    auto analytic = [](double x, double y) -> double {
        return std::sin(M_PI * x / 2.0) * std::sin(M_PI * y / 2.0);
    };

    int fine_res = 200;

    simple_grid<2> fine_grid(
        Eigen::Vector<double, 2>(0.005, 0.005),
        Eigen::Vector<double, 2>(1.995, 1.995),
        std::vector<int>{fine_res, fine_res}
    );
    size_t n_nodes = fine_grid.contentSize(var_loc::nodeCentered);

    std::vector<Eigen::RowVector<double, 2>> fine_pos(n_nodes);
    for (size_t i = 0; i < n_nodes; i++) {
        auto ni = fine_grid.i2n(i);
        fine_pos[i] = fine_grid.nodePosition(ni);
    }
    device_array<Eigen::RowVector<double, 2>> fine_pos_dev(q, fine_pos);

    auto run_case = [&](const char* label, const std::vector<int>& exponents) -> double {
        using Field = device_multi_patch_field<2, var_loc::nodeCentered, double>;

        Field field(
            multi_patch_grid<2>(
                q,
                {0.0, 0.0},
                {2.0, 2.0},
                {2, 2},
                exponents
            )
        );

        const auto& mpg = field.getGrid();
        size_t mpg_nodes = mpg.contentSize(var_loc::nodeCentered);
        std::vector<double> host_nodes(mpg_nodes);
        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            size_t off = mpg.get_patch_node_offsets()[p];
            for (size_t i = 0; i < pg.contentSize(var_loc::nodeCentered); i++) {
                auto ni = pg.i2n(i);
                auto pos = pg.nodePosition(ni);
                host_nodes[off + i] = analytic(pos[0], pos[1]);
            }
        }
        field.copy(host_nodes);

        device_array<double> err_dev(q, std::vector<double>(n_nodes, 0.0));

        q.submit([&](sycl::handler& h) {
            auto acc = field.get_access(h);
            auto p_acc = fine_pos_dev.get_access(h);
            auto e_acc = err_dev.get_access(h);
            h.parallel_for(sycl::range<1>(n_nodes), [=](sycl::id<1> idx) {
                double val = interp(p_acc[idx], acc);
                double exact = sycl::sin(M_PI * p_acc[idx][0] / 2.0)
                             * sycl::sin(M_PI * p_acc[idx][1] / 2.0);
                e_acc[idx] = val - exact;
            });
        }).wait();

        auto err_host = err_dev.to_host();
        double max_abs_err = 0;
        for (auto v : err_host)
            if (std::abs(v) > max_abs_err) max_abs_err = std::abs(v);

        std::string fname = std::string("interp_") + label + "_error.plt";
        {
            std::ofstream fp(fname);
            fp << std::setprecision(10);
            fine_grid.plot<double, 1>(fp, err_host.data(), 0.0, var_loc::nodeCentered);
        }

        std::cout << "  [" << label << "] exponents={";
        for (size_t i = 0; i < exponents.size(); i++) {
            if (i > 0) std::cout << ",";
            std::cout << exponents[i] << "(" << (1 << exponents[i]) << "x" << (1 << exponents[i]) << ")";
        }
        std::cout << "}, max |err| = " << std::scientific << max_abs_err
                  << "  -> " << fname << std::endl;

        return max_abs_err;
    };

    double err_uniform = run_case("uniform", {4, 4, 4, 4});
    double err_nonuniform = run_case("nonuniform", {3, 4, 3, 5});
    double err_extreme = run_case("extreme", {2, 6, 2, 6});

    assert(err_uniform < 0.005);
    assert(err_nonuniform < 0.01);
    assert(err_extreme < 0.05);

    std::cout << "✅ Interpolation accuracy test passed." << std::endl;
}

void test_add_back_gaussian() {
    std::cout << "--- Test: add_back performance with Gaussian particles ---" << std::endl;

    auto& q = get_queue();

    using Field = device_multi_patch_field<2, var_loc::nodeCentered, double>;

    Field field(
        multi_patch_grid<2>(
            q,
            {0.0, 0.0},
            {2.0, 2.0},
            {2, 2},
            std::vector<int>{3, 5, 3, 6}
        )
    );
    const auto& mpg = field.getGrid();

    std::mt19937 rng(42);
    std::normal_distribution<double> dist(1.0, 0.3);

    int n_max = 10000000;
    std::vector<Eigen::RowVector<double, 2>> all_positions;
    all_positions.reserve(n_max);
    while ((int)all_positions.size() < n_max) {
        double x = dist(rng);
        double y = dist(rng);
        Eigen::RowVector<double, 2> pos(x, y);
        if (mpg.inGrid(pos)) {
            all_positions.push_back(pos);
        }
    }

    int bench_id = 0;

    auto bench = [&](int n, bool is_warmup) {
        std::vector<Eigen::RowVector<double, 2>> positions(
            all_positions.begin(), all_positions.begin() + n
        );

        bench_id++;
        std::string key_upload = "upload_" + std::to_string(bench_id);
        std::string key_kernel = "add_back_" + std::to_string(bench_id);

        Tic(key_upload)
        device_array<Eigen::RowVector<double, 2>> pos_dev(q, positions);
        Toc_(key_upload)

        field.setZero();

        const int n_repeat = is_warmup ? 1 : 5;
        Tic(key_kernel)
        for (int r = 0; r < n_repeat; r++) {
            q.submit([&](sycl::handler& h) {
                auto acc = field.get_access(h);
                auto p_acc = pos_dev.get_access(h);
                h.parallel_for(sycl::range<1>(positions.size()), [=](sycl::id<1> idx) {
                    double w = 1.0;
                    add_back(p_acc[idx], w, acc);
                });
            }).wait();
        }
        Toc_(key_kernel)

        if (!is_warmup) {
            double ms_upload = TimeUsed(key_upload) * 1000.0;
            double ms_kernel = TimeUsed(key_kernel) * 1000.0 / n_repeat;
            std::cout << "  n=" << std::setw(12) << n
                      << "  upload=" << std::setw(10) << std::fixed << std::setprecision(2) << ms_upload << " ms"
                      << "  add_back=" << std::setw(10) << ms_kernel << " ms"
                      << "  (" << std::setw(10) << std::scientific << std::setprecision(2)
                      << n / (ms_kernel * 1e-3) << " particles/s)"
                      << std::endl;
        }
    };

    std::cout << "  warmup..." << std::endl;
    bench(n_max, true);
    std::cout << "  warmup done." << std::endl;

    std::cout << "  benchmark (add_back averaged over 5 runs):" << std::endl;
    bench(100000, false);
    bench(500000, false);
    bench(1000000, false);
    bench(5000000, false);
    bench(10000000, false);

    field.setZero();
    device_array<Eigen::RowVector<double, 2>> pos_dev(q, all_positions);
    q.submit([&](sycl::handler& h) {
        auto acc = field.get_access(h);
        auto p_acc = pos_dev.get_access(h);
        h.parallel_for(sycl::range<1>(all_positions.size()), [=](sycl::id<1> idx) {
            double w = 1.0;
            add_back(p_acc[idx], w, acc);
        });
    }).wait();
    field.plot("test_add_back_gaussian.plt", "density", 0.0);
    std::cout << "  -> test_add_back_gaussian.plt" << std::endl;
    std::cout << "✅ add_back Gaussian test passed." << std::endl;
}

int main() {
    test_index_conversion();
    test_interp_device();
    test_interp_accuracy();
    test_add_back_gaussian();
    return 0;
}
