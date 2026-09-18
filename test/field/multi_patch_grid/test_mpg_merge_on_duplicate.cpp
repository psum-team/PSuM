#include <psum/field.hpp>
#include <field/multi_patch/duplicate_map.hpp>
#include <field/multi_patch/node_volume_field.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <vector>
#include <iomanip>

using namespace psum::field;
using namespace multi_patch;

sycl::queue& get_queue() {
    static sycl::queue q;
    return q;
}

void test_1d_basic() {
    std::cout << "--- Test: 1D basic sync ---" << std::endl;

    multi_patch_grid<1> mpg(
        get_queue(),
        {0.0},
        {3.0},
        {3},
        std::vector<int>{2, 3, 2}
    );

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    assert((long long)dm.n_index == dm.P.rows());
    assert((long long)dm.n_dof == dm.P.cols());
    assert((long long)dm.n_index == dm.P.nonZeros());

    size_t expected_nodes = 0;
    for (size_t p = 0; p < mpg.patches_count(); p++) {
        int cpd = 1 << mpg.get_resolution_exponents()[p];
        expected_nodes += (cpd + 1);
    }
    assert(dm.n_index == expected_nodes);

    assert((long long)dm.n_index == S.rows());
    assert((long long)dm.n_index == S.cols());

    assert(dm.n_dof < dm.n_index);

    std::cout << "  n_index=" << dm.n_index << "  n_dof=" << dm.n_dof
              << "  S nonzeros=" << S.nonZeros() << std::endl;

    size_t boundary_pairs = (dm.n_index - dm.n_dof);
    assert(boundary_pairs == 2);

    for (size_t i = 0; i < dm.n_index; i++) {
        assert(S.coeff(i, i) == 1.0);
    }

    size_t off_diag_count = 0;
    for (int k = 0; k < S.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            if (it.row() != it.col()) {
                assert(it.value() == 1.0);
                assert(dm.index_to_dof[it.row()] == dm.index_to_dof[it.col()]);
                off_diag_count++;
            }
        }
    }
    assert(off_diag_count == 4);

    std::cout << "  boundary_pairs=" << boundary_pairs
              << "  off_diagonal_entries=" << off_diag_count << std::endl;
    std::cout << "  1D basic passed." << std::endl;
}

void test_2d_dof_coordinate_consistency() {
    std::cout << "--- Test: 2D DOF coordinate consistency ---" << std::endl;

    auto approx = [](double a, double b) { return std::abs(a - b) < 1e-10; };

    multi_patch_grid<2> mpg(
        get_queue(),
        {0.0, 0.0},
        {2.0, 2.0},
        {2, 2},
        std::vector<int>{3, 5, 3, 6}
    );

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    std::cout << "  n_index=" << dm.n_index << "  n_dof=" << dm.n_dof << std::endl;

    assert((long long)dm.n_index == S.rows());
    assert(dm.n_dof < dm.n_index);

    std::vector<std::vector<size_t>> dof_to_indices(dm.n_dof);
    for (size_t i = 0; i < dm.n_index; i++)
        dof_to_indices[dm.index_to_dof[i]].push_back(i);

    size_t shared_count = 0;
    for (size_t d = 0; d < dm.n_dof; d++) {
        if (dof_to_indices[d].size() > 1)
            shared_count++;

        auto pos0 = mpg.template position<grid_element::node>(dof_to_indices[d][0]);
        for (size_t k = 1; k < dof_to_indices[d].size(); k++) {
            auto posK = mpg.template position<grid_element::node>(dof_to_indices[d][k]);
            for (int dim = 0; dim < 2; dim++)
                assert(approx(pos0[dim], posK[dim]));
        }
    }

    for (size_t d1 = 0; d1 < dm.n_dof; d1++) {
        for (size_t d2 = d1 + 1; d2 < dm.n_dof; d2++) {
            auto p1 = mpg.template position<grid_element::node>(dof_to_indices[d1][0]);
            auto p2 = mpg.template position<grid_element::node>(dof_to_indices[d2][0]);
            bool should_differ = false;
            for (int dim = 0; dim < 2; dim++) {
                if (!approx(p1[dim], p2[dim])) {
                    should_differ = true;
                    break;
                }
            }
            assert(should_differ);
        }
    }

    std::cout << "  shared DOFs (boundary nodes with >1 index): " << shared_count << std::endl;
    std::cout << "  2D DOF coordinate consistency passed." << std::endl;
}

void test_2d_uniform() {
    std::cout << "--- Test: 2D uniform sync ---" << std::endl;

    multi_patch_grid<2> mpg(
        get_queue(),
        {0.0, 0.0},
        {2.0, 2.0},
        {2, 2},
        std::vector<int>{3, 3, 3, 3}
    );

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    size_t nodes_per_patch = 9 * 9;
    assert(dm.n_index == 4 * nodes_per_patch);

    size_t expected_dof = 17 * 17;
    assert(dm.n_dof == expected_dof);

    std::cout << "  n_index=" << dm.n_index << "  n_dof=" << dm.n_dof << std::endl;
    std::cout << "  2D uniform passed." << std::endl;
}

void test_apply_sync() {
    std::cout << "--- Test: Apply S to field (2D) ---" << std::endl;

    auto approx = [](double a, double b) { return std::abs(a - b) < 1e-10; };

    multi_patch_grid<2> mpg(
        get_queue(),
        {0.0, 0.0},
        {2.0, 2.0},
        {2, 2},
        std::vector<int>{3, 3, 3, 3}
    );

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    Eigen::VectorXd field = Eigen::VectorXd::Zero(dm.n_index);
    for (size_t i = 0; i < dm.n_index; i++) {
        auto pos = mpg.template position<grid_element::node>(i);
        field(i) = pos[0] + pos[1];
    }

    Eigen::VectorXd synced = S * field;

    std::vector<double> count(dm.n_index, 0.0);
    for (size_t i = 0; i < dm.n_index; i++) {
        for (size_t j = 0; j < dm.n_index; j++) {
            if (S.coeff(i, j) == 1.0) {
                count[i] += field(j);
            }
        }
    }

    for (size_t i = 0; i < dm.n_index; i++)
        assert(approx(synced(i), count[i]));

    std::vector<size_t> dof_count(dm.n_dof, 0);
    for (size_t i = 0; i < dm.n_index; i++)
        dof_count[dm.index_to_dof[i]]++;

    for (size_t i = 0; i < dm.n_index; i++) {
        size_t dof = dm.index_to_dof[i];
        double expected_val = 0;
        for (size_t j = 0; j < dm.n_index; j++) {
            if (dm.index_to_dof[j] == dof)
                expected_val += field(j);
        }
        assert(approx(synced(i), expected_val));
    }

    std::cout << "  Apply S passed." << std::endl;
}

void test_3d_large() {
    std::cout << "--- Test: 3D large grid sync ---" << std::endl;

    auto approx = [](double a, double b) { return std::abs(a - b) < 1e-10; };

    std::vector<int> exponents;
    for (int i = 0; i < 512; i++)
        exponents.push_back(3 + (i % 3));

    multi_patch_grid<3> mpg(
        get_queue(),
        {0.0, 0.0, 0.0},
        {8.0, 8.0, 8.0},
        {8, 8, 8},
        exponents
    );

    size_t n_index = mpg.contentSize(var_loc::nodeCentered);
    std::cout << "  total nodes: " << n_index << std::endl;
    assert(n_index > 1000000);

    auto dm = build_duplicate_map(mpg);
    auto S = build_merge_matrix(dm);

    assert(dm.n_index == n_index);
    std::cout << "  n_index=" << dm.n_index << "  n_dof=" << dm.n_dof
              << "  S rows=" << S.rows() << "  S nnz=" << S.nonZeros() << std::endl;

    assert((long long)dm.n_index == S.rows());
    assert(dm.n_dof < dm.n_index);

    std::vector<std::vector<size_t>> dof_to_indices(dm.n_dof);
    for (size_t i = 0; i < dm.n_index; i++)
        dof_to_indices[dm.index_to_dof[i]].push_back(i);

    size_t shared_count = 0;
    size_t max_sharing = 0;
    for (size_t d = 0; d < dm.n_dof; d++) {
        if (dof_to_indices[d].size() > 1)
            shared_count++;
        if (dof_to_indices[d].size() > max_sharing)
            max_sharing = dof_to_indices[d].size();
    }
    std::cout << "  shared DOFs: " << shared_count
              << "  max sharing: " << max_sharing << std::endl;

    size_t check_limit = std::min(dm.n_dof, (size_t)50000);
    for (size_t d = 0; d < check_limit; d++) {
        auto pos0 = mpg.template position<grid_element::node>(dof_to_indices[d][0]);
        for (size_t k = 1; k < dof_to_indices[d].size(); k++) {
            auto posK = mpg.template position<grid_element::node>(dof_to_indices[d][k]);
            for (int dim = 0; dim < 3; dim++)
                assert(approx(pos0[dim], posK[dim]));
        }
    }

    size_t distinct_pairs = 0;
    size_t max_check = std::min(dm.n_dof, (size_t)5000);
    for (size_t d1 = 0; d1 < max_check; d1++) {
        for (size_t d2 = d1 + 1; d2 < max_check; d2++) {
            auto p1 = mpg.template position<grid_element::node>(dof_to_indices[d1][0]);
            auto p2 = mpg.template position<grid_element::node>(dof_to_indices[d2][0]);
            bool should_differ = false;
            for (int dim = 0; dim < 3; dim++) {
                if (!approx(p1[dim], p2[dim])) {
                    should_differ = true;
                    break;
                }
            }
            assert(should_differ);
            distinct_pairs++;
        }
    }
    std::cout << "  checked " << distinct_pairs << " distinct DOF pairs." << std::endl;

    for (size_t i = 0; i < dm.n_index; i++) {
        assert(S.coeff(i, i) == 1.0);
    }

    size_t off_diag = 0;
    for (int k = 0; k < S.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
            if (it.row() != it.col()) {
                assert(it.value() == 1.0);
                assert(dm.index_to_dof[it.row()] == dm.index_to_dof[it.col()]);
                off_diag++;
            }
        }
    }
    size_t expected_off_diag = 0;
    for (size_t d = 0; d < dm.n_dof; d++) {
        size_t k = dof_to_indices[d].size();
        expected_off_diag += k * (k - 1);
    }
    std::cout << "  off-diagonal entries: " << off_diag << std::endl;
    assert(off_diag == expected_off_diag);

    {
        Eigen::VectorXd field(dm.n_index);
        for (size_t i = 0; i < dm.n_index; i++) {
            auto pos = mpg.template position<grid_element::node>(i);
            field(i) = pos[0] + 2.0 * pos[1] + 3.0 * pos[2];
        }
        Eigen::VectorXd synced = S * field;

        for (size_t i = 0; i < dm.n_index; i++) {
            size_t dof = dm.index_to_dof[i];
            double expected = 0;
            for (size_t j : dof_to_indices[dof])
                expected += field(j);
            assert(approx(synced(i), expected));
        }
    }

    std::cout << "  3D large sync passed." << std::endl;
}

void test_node_volume_2d() {
    std::cout << "--- Test: Node volume 2D ---" << std::endl;

    std::vector<int> exponents(64);
    for (int iy = 0; iy < 8; iy++) {
        for (int ix = 0; ix < 8; ix++) {
            int ring = std::min({ix, 7 - ix, iy, 7 - iy});
            exponents[iy * 8 + ix] = 2 + std::min(ring, 3);
        }
    }

    auto field = build_node_volume_field(
        multi_patch_grid<2>(
            get_queue(),
            {0.0, 0.0},
            {8.0, 8.0},
            {8, 8},
            exponents
        )
    );

    field.plot("node_volume_2d.plt", "volume", 0.0);
    std::cout << "  -> node_volume_2d.plt" << std::endl;
}

void test_node_volume_3d() {
    std::cout << "--- Test: Node volume 3D ---" << std::endl;

    std::vector<int> exponents(27, 3);
    for (int iz = 1; iz <= 1; iz++)
        for (int iy = 1; iy <= 1; iy++)
            for (int ix = 1; ix <= 1; ix++)
                exponents[iz * 9 + iy * 3 + ix] = 5;

    auto field = build_node_volume_field(
        multi_patch_grid<3>(
            get_queue(),
            {0.0, 0.0, 0.0},
            {3.0, 3.0, 3.0},
            {3, 3, 3},
            exponents
        )
    );

    field.plot("node_volume_3d.plt", "volume", 0.0);
    std::cout << "  -> node_volume_3d.plt" << std::endl;

    auto vol_host = field.getContent().to_host();
    for (size_t i = 0; i < field.size(); i++) {
        assert(vol_host[i] > 0);
    }

    std::cout << "  3D node volume passed." << std::endl;
}

int main() {
    test_1d_basic();
    test_2d_dof_coordinate_consistency();
    test_2d_uniform();
    test_apply_sync();
    test_3d_large();
    test_node_volume_2d();
    test_node_volume_3d();
    std::cout << "\nAll sync matrix tests passed." << std::endl;
    return 0;
}
