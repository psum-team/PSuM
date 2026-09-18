#include <psum/field.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace psum::field;
using namespace multi_patch;

sycl::queue& get_queue() {
    static sycl::queue q;
    return q;
}

void test_construction_and_geometry() {
    std::cout << "--- Test: Construction, geometry and adjacency ---" << std::endl;

    multi_patch_grid<2> mpg(
        get_queue(),
        {0.0, 0.0},
        {2.0, 2.0},
        {2, 2},
        std::vector<int>{2, 3, 4, 5}
    );

    assert(mpg.patches_count() == 4);
    assert(mpg.patch_grid(0).numCells<0>() == 4);
    assert(mpg.patch_grid(1).numCells<0>() == 8);
    assert(mpg.patch_grid(2).numCells<0>() == 16);
    assert(mpg.patch_grid(3).numCells<0>() == 32);

    assert(mpg.patch_content_size(0, var_loc::cellCentered) == 16);
    assert(mpg.patch_content_size(1, var_loc::cellCentered) == 64);
    assert(mpg.patch_content_size(2, var_loc::cellCentered) == 256);
    assert(mpg.patch_content_size(3, var_loc::cellCentered) == 1024);

    assert(mpg.contentSize(var_loc::cellCentered) == 16 + 64 + 256 + 1024);
    assert(mpg.contentSize(var_loc::cellCentered) == 1360);

    assert(mpg.patch_offset(0) == 0);
    assert(mpg.patch_offset(1) == 16);
    assert(mpg.patch_offset(2) == 80);
    assert(mpg.patch_offset(3) == 336);

    auto approx = [](double a, double b) {
        return std::abs(a - b) < 1e-10;
    };

    for (size_t p = 0; p < mpg.patches_count(); p++) {
        auto pg = mpg.patch_grid(p);
        auto cell_idx = mpg.coarse_grid().i2c(p);

        for (int d = 0; d < 2; d++) {
            double expected_lower = mpg.get_lower_bounds()[d] + cell_idx.indices[d] * mpg.get_patch_spans()[d];
            double expected_upper = expected_lower + mpg.get_patch_spans()[d];
            assert(approx(pg.get_lower_bounds()[d], expected_lower));
            assert(approx(pg.get_upper_bounds()[d], expected_upper));
            assert(pg.get_cell_num()[d] == pg.numCells<0>());
        }
    }

    {
        auto pg0 = mpg.patch_grid(0);
        auto pg1 = mpg.patch_grid(1);
        auto pg2 = mpg.patch_grid(2);

        assert(approx(pg0.get_upper_bounds()[1], pg1.get_lower_bounds()[1]));
        assert(approx(pg0.get_lower_bounds()[0], pg1.get_lower_bounds()[0]));
        assert(approx(pg0.get_upper_bounds()[0], pg1.get_upper_bounds()[0]));

        assert(approx(pg0.get_upper_bounds()[0], pg2.get_lower_bounds()[0]));
        assert(approx(pg0.get_lower_bounds()[1], pg2.get_lower_bounds()[1]));
        assert(approx(pg0.get_upper_bounds()[1], pg2.get_upper_bounds()[1]));

        for (size_t p = 0; p < mpg.patches_count(); p++) {
            auto pg = mpg.patch_grid(p);
            for (int d = 0; d < 2; d++) {
                assert(pg.get_lower_bounds()[d] < pg.get_upper_bounds()[d]);
            }
        }
    }

    {
        multi_patch_grid<1> mpg1d(
            get_queue(),
            {0.0},
            {3.0},
            {3},
            std::vector<int>{2, 3, 4}
        );
        auto pg0 = mpg1d.patch_grid(0);
        auto pg1 = mpg1d.patch_grid(1);
        auto pg2 = mpg1d.patch_grid(2);

        assert(approx(pg0.get_upper_bounds()[0], pg1.get_lower_bounds()[0]));
        assert(approx(pg1.get_upper_bounds()[0], pg2.get_lower_bounds()[0]));
        assert(approx(pg0.get_lower_bounds()[0], 0.0));
        assert(approx(pg2.get_upper_bounds()[0], 3.0));
    }

    std::cout << "✅ Construction, geometry and adjacency passed." << std::endl;
}

int main() {
    test_construction_and_geometry();
    return 0;
}
