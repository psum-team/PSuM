#include <iostream>
#include <cmath>
#include <algorithm>
#include <random>
#include <map>
#include "../../src/field_solver/fixed_boundary.hpp"
#include "../../src/timer.hpp"

using namespace psum::field_solver;

void test_boundary_constant_value() {
    std::cout << "Test: fixed_boundary with constant value" << std::endl;

    fixed_boundary_2d b1;
    std::unordered_map<size_t, std::array<double, 2>> elements = {
        {10, {0.0, 0.0}},
        {20, {1.0, 1.0}}
    };
    b1.set_related_elements(elements);
    b1 = 5.0;

    std::vector<std::pair<size_t, double>> results;
    b1.evaluate(results);

    bool correct = (results.size() == 2) &&
                    (std::abs(results[0].second - 5.0) < 1e-10) &&
                    (std::abs(results[1].second - 5.0) < 1e-10);

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_function_value() {
    std::cout << "Test: fixed_boundary with function value" << std::endl;

    fixed_boundary_2d b1;
    std::unordered_map<size_t, std::array<double, 2>> elements = {
        {10, {0.0, 0.0}},
        {20, {1.0, 1.0}}
    };
    b1.set_related_elements(elements);
    b1 = [](const std::array<double, 2>& pos) { return pos[0] + pos[1]; };

    std::vector<std::pair<size_t, double>> results;
    b1.evaluate(results);
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    bool correct = (results.size() == 2) &&
                    (results[0].first == 10 && std::abs(results[0].second - 0.0) < 1e-10) &&
                    (results[1].first == 20 && std::abs(results[1].second - 2.0) < 1e-10);

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_list_overlap() {
    std::cout << "Test: fixed_boundary_list overlap handling" << std::endl;

    fixed_boundary_2d b1, b2;
    b1.set_related_elements({{1, {0,0}}, {2, {0,0}}});
    b1 = 1.0;

    b2.set_related_elements({{2, {1,1}}, {3, {1,1}}});
    b2 = 2.0;

    fixed_boundary_list<2> b_list;
    b_list.set_related_boundaries({b1, b2});

    std::vector<std::pair<size_t, double>> result_v;
    b_list.evaluate(result_v);

    // b1 has idx 1 (val 1.0), b2 has idx 2 and 3 (val 2.0)
    // So we expect: idx 1 with val 1.0, idx 2 with val 2.0, idx 3 with val 2.0
    bool correct = (result_v.size() == 3);

    if (correct) {
        // Build a map for easy lookup
        std::unordered_map<size_t, double> val_map;
        for (auto& [idx, val] : result_v) {
            val_map[idx] = val;
        }
        correct = (std::abs(val_map[1] - 1.0) < 1e-10) &&
                  (std::abs(val_map[2] - 2.0) < 1e-10) &&
                  (std::abs(val_map[3] - 2.0) < 1e-10);
    }

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_erase_at() {
    std::cout << "Test: fixed_boundary erase_at" << std::endl;

    fixed_boundary_2d b1;
    b1.set_related_elements({{100, {0,0}}, {101, {0,0}}, {102, {0,0}}});
    b1 = 10.0;

    std::vector<size_t> erase_list = {100, 102};
    b1.erase_at(erase_list);

    std::vector<std::pair<size_t, double>> results;
    b1.evaluate(results);

    bool correct = (results.size() == 1) &&
                    (results[0].first == 101) &&
                    (std::abs(results[0].second - 10.0) < 1e-10);

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_list_erase_at() {
    std::cout << "Test: fixed_boundary_list erase_at" << std::endl;

    fixed_boundary_2d b1, b2;
    b1.set_related_elements({{0, {0,0}}, {1, {0,0}}});
    b1 = 1.0;

    b2.set_related_elements({{2, {1,1}}, {3, {1,1}}});
    b2 = 2.0;

    fixed_boundary_list<2> b_list;
    b_list.set_related_boundaries({b1, b2});

    b_list.erase_at({0, 2});

    bool correct = (b_list.elements_size() == 2) &&
                    (b_list.element_exists(1)) &&
                    (b_list.element_exists(3)) &&
                    (!b_list.element_exists(0)) &&
                    (!b_list.element_exists(2));

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_exception() {
    std::cout << "Test: fixed_boundary exception handling" << std::endl;

    fixed_boundary_2d b_err;
    b_err.set_related_elements({{1, {0,0}}});

    bool caught = false;
    try {
        std::vector<std::pair<size_t, double>> res;
        b_err.evaluate(res);
    } catch (const std::runtime_error& e) {
        caught = true;
    }

    if (caught)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_boundary_3d() {
    std::cout << "Test: fixed_boundary_3d" << std::endl;

    fixed_boundary_3d b1;
    std::unordered_map<size_t, std::array<double, 3>> elements = {
        {0, {0.0, 0.0, 0.0}},
        {1, {1.0, 1.0, 1.0}}
    };
    b1.set_related_elements(elements);
    b1 = 2.718;

    std::vector<std::pair<size_t, double>> results;
    b1.evaluate(results);

    bool correct = (results.size() == 2);
    for (auto& [idx, val] : results) {
        if (std::abs(val - 2.718) > 1e-10) {
            correct = false;
            break;
        }
    }

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_stress_test() {
    std::cout << "Test: Stress test with 10,000 points and overlaps" << std::endl;

    const int NUM_POINTS = 10000;
    const int NUM_GROUPS = 5;

    std::unordered_map<size_t, double> ground_truth;
    fixed_boundary_3d boundaries[5];

    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> val_dist(0.0, 100.0);

    size_t current_idx = 0;
    for (int i = 0; i < NUM_GROUPS; ++i) {
        std::unordered_map<size_t, std::array<double, 3>> pts;

        for (int j = 0; j < 2000; ++j) {
            size_t idx = current_idx++;
            double val = (double)i * 100.0;
            pts[idx] = {0, 0, 0};
            ground_truth[idx] = val;
        }

        boundaries[i] = (double)i * 100.0;
        boundaries[i].set_related_elements(pts);
    }

    fixed_boundary_list<3> list;
    list.set_related_boundaries({boundaries[0], boundaries[1], boundaries[2], boundaries[3], boundaries[4]});

    std::vector<std::pair<size_t, double>> results;
    list.evaluate(results);

    bool correct = (results.size() == ground_truth.size());
    for (const auto& res : results) {
        if (std::abs(ground_truth[res.first] - res.second) > 1e-10) {
            correct = false;
            break;
        }
    }

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

void test_performance_large_scale() {
    std::cout << "Test: Performance test with 500,000 points" << std::endl;

    const int SCALE = 500000;
    const int NUM_B = 5;
    fixed_boundary_3d bs[5];

    std::unordered_map<size_t, double> inspector;

    for (int i = 0; i < NUM_B; ++i) {
        std::unordered_map<size_t, std::array<double, 3>> pts;
        for (size_t j = 0; j < SCALE / 3; ++j) {
            size_t idx = j + (i * 50000);
            pts[idx] = {0, 0, 0};
            inspector[idx] = (double)i * 10.0;
        }
        bs[i] = (double)i * 10.0;
        bs[i].set_related_elements(pts);
    }

    double expected_sum = 0;
    for (auto const& [idx, val] : inspector) expected_sum += val;

    Tic("performance_test")
    fixed_boundary_list<3> list;
    list.set_related_boundaries({bs[0], bs[1], bs[2], bs[3], bs[4]});
    std::vector<std::pair<size_t, double>> results;
    list.evaluate(results);
    Toc

    double actual_sum = 0;
    for (auto& p : results) actual_sum += p.second;

    bool correct = (std::abs(expected_sum - actual_sum) < 1e-6) &&
                   (results.size() == inspector.size());

    std::cout << "  Elapsed time: " << TimeUsed("performance_test")*1000 << " ms" << std::endl;
    std::cout << "  Points processed: " << results.size() << std::endl;

    if (correct)
        std::cout << "✅ PASSED" << std::endl;
    else
        std::cout << "❌ FAILED" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Testing fixed_boundary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    test_boundary_constant_value();
    test_boundary_function_value();
    test_boundary_list_overlap();
    test_boundary_erase_at();
    test_boundary_list_erase_at();
    test_boundary_exception();
    test_boundary_3d();
    test_stress_test();
    test_performance_large_scale();

    std::cout << "========================================" << std::endl;
    std::cout << "All tests completed" << std::endl;
    std::cout << "========================================" << std::endl;

    PrintTimer;

    return 0;
}
