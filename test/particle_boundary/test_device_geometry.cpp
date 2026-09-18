#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <sycl/sycl.hpp>
#include "../../src/timer.hpp"
#include <psum/particle_boundary.hpp>
#include "../../src/random/rander.hpp"

using namespace std;
using namespace psum::random;
using namespace psum::particle_boundary::geometry;

void test_trace_to_idxs_on_device() {
    cout << "==========================================" << endl;
    cout << "check: trace_to_idxs on device" << endl;

    sycl::queue q;
    cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << endl;

    const int num_tests = 1000000;
    vector<int> host_results(num_tests, -1);
    vector<double> host_points(num_tests * 6);

    rander seed;
    auto R = [&seed]() { return 8 * seed() - 2; };

    for (int i = 0; i < num_tests * 6; i++) {
        host_points[i] = R();
    }

    int* device_results = sycl::malloc_device<int>(num_tests, q);
    double* device_points = sycl::malloc_device<double>(num_tests * 6, q);

    q.memcpy(device_points, host_points.data(), host_points.size() * sizeof(double)).wait();

    int n_loop = 100;

    Tic("trace_to_idxs")
    for (int i = 0; i < n_loop; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tests), [=](sycl::id<1> idx) {
                int i = idx[0];
                int I = 4, J = 3, K = 2;
                double x1 = device_points[i * 6 + 0];
                double y1 = device_points[i * 6 + 1];
                double z1 = device_points[i * 6 + 2];
                double x2 = device_points[i * 6 + 3];
                double y2 = device_points[i * 6 + 4];
                double z2 = device_points[i * 6 + 5];

                trace_to_idxs_state state;
                int first_idx = trace_to_idxs_begin(I, J, K, x1, y1, z1, x2, y2, z2, state);
                int count = 0;
                if (first_idx >= 0) {
                    count++;
                    for (int idx = first_idx; trace_to_idxs_valid(state); idx = trace_to_idxs_next(state)) {
                        count++;
                    }
                }
                device_results[i] = count;
            });
        }).wait();
    }
    Toc

    q.memcpy(host_results.data(), device_results, num_tests * sizeof(int)).wait();

    int correct = 0;
    for (int i = 0; i < num_tests; i++) {
        int I = 4, J = 3, K = 2;
        double x1 = host_points[i * 6 + 0];
        double y1 = host_points[i * 6 + 1];
        double z1 = host_points[i * 6 + 2];
        double x2 = host_points[i * 6 + 3];
        double y2 = host_points[i * 6 + 4];
        double z2 = host_points[i * 6 + 5];

        trace_to_idxs_state state;
        int first_idx = trace_to_idxs_begin(I, J, K, x1, y1, z1, x2, y2, z2, state);
        int host_count = 0;
        if (first_idx >= 0) {
            host_count++;
            for (int idx = first_idx; trace_to_idxs_valid(state); idx = trace_to_idxs_next(state)) {
                host_count++;
            }
        }

        if (host_results[i] == host_count) {
            correct++;
        } else {
            cout << "Mismatch at test " << i << ": device=" << host_results[i] << ", host=" << host_count << endl;
        }
    }

    cout << "trace_to_idxs device test: " << correct << "/" << num_tests << " passed" << endl;
    if (correct == num_tests) {
        cout << "✅ All trace_to_idxs device tests passed!" << endl;
    } else {
        cout << "❌ Some trace_to_idxs device tests failed!" << endl;
    }

    cout << num_tests * n_loop / TimeUsed("trace_to_idxs") << " traces2idxs per second" << endl;

    sycl::free(device_results, q);
    sycl::free(device_points, q);
}

void test_line_seg_tri_intersect_on_device() {
    cout << "==========================================" << endl;
    cout << "check: line_seg_tri_intersect_test on device" << endl;

    sycl::queue q;

    const int num_tests = 1000000;
    vector<double> host_results(num_tests, 0);
    vector<double> host_tri_points(num_tests * 9);
    vector<double> host_seg_points(num_tests * 6);

    rander seed;
    auto R = [&seed]() { return seed() * 2.0 - 1.0; };

    for (int i = 0; i < num_tests; i++) {
        for (int j = 0; j < 9; j++) {
            host_tri_points[i * 9 + j] = R();
        }
        for (int j = 0; j < 6; j++) {
            host_seg_points[i * 6 + j] = R();
        }
    }

    double* device_results = sycl::malloc_device<double>(num_tests, q);
    double* device_tri_points = sycl::malloc_device<double>(num_tests * 9, q);
    double* device_seg_points = sycl::malloc_device<double>(num_tests * 6, q);

    q.memcpy(device_tri_points, host_tri_points.data(), host_tri_points.size() * sizeof(double)).wait();
    q.memcpy(device_seg_points, host_seg_points.data(), host_seg_points.size() * sizeof(double)).wait();

    int n_loop = 100;

    Tic("line_seg_tri_intersect")
    for (int i = 0; i < n_loop; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tests), [=](sycl::id<1> idx) {
                int i = idx[0];
                double lambda;
                bool result = line_seg_tri_intersect_test(
                    device_tri_points[i * 9 + 0], device_tri_points[i * 9 + 1], device_tri_points[i * 9 + 2],
                    device_tri_points[i * 9 + 3], device_tri_points[i * 9 + 4], device_tri_points[i * 9 + 5],
                    device_tri_points[i * 9 + 6], device_tri_points[i * 9 + 7], device_tri_points[i * 9 + 8],
                    device_seg_points[i * 6 + 0], device_seg_points[i * 6 + 1], device_seg_points[i * 6 + 2],
                    device_seg_points[i * 6 + 3], device_seg_points[i * 6 + 4], device_seg_points[i * 6 + 5],
                    lambda
                );
                device_results[i] = lambda;
            });
        }).wait();
    }
    Toc

    q.memcpy(host_results.data(), device_results, num_tests * sizeof(double)).wait();

    int correct = 0;
    for (int i = 0; i < num_tests; i++) {
        double lambda;
        bool host_result = line_seg_tri_intersect_test(
            host_tri_points[i * 9 + 0], host_tri_points[i * 9 + 1], host_tri_points[i * 9 + 2],
            host_tri_points[i * 9 + 3], host_tri_points[i * 9 + 4], host_tri_points[i * 9 + 5],
            host_tri_points[i * 9 + 6], host_tri_points[i * 9 + 7], host_tri_points[i * 9 + 8],
            host_seg_points[i * 6 + 0], host_seg_points[i * 6 + 1], host_seg_points[i * 6 + 2],
            host_seg_points[i * 6 + 3], host_seg_points[i * 6 + 4], host_seg_points[i * 6 + 5],
            lambda
        );

        if (abs(host_results[i] - lambda) < 1e-10) {
            correct++;
        } else {
            cout << "Mismatch at test " << i << ": device=" << host_results[i] << ", host=" << lambda << endl;
        }
    }

    cout << "line_seg_tri_intersect device test: " << correct << "/" << num_tests << " passed" << endl;
    if (correct == num_tests) {
        cout << "✅ All line_seg_tri_intersect device tests passed!" << endl;
    } else {
        cout << "❌ Some line_seg_tri_intersect device tests failed!" << endl;
    }

    cout << num_tests * n_loop / TimeUsed("line_seg_tri_intersect") << " ray-tri collisions per second" << endl;

    sycl::free(device_results, q);
    sycl::free(device_tri_points, q);
    sycl::free(device_seg_points, q);
}

void test_box_tri_overlap_on_device() {
    cout << "==========================================" << endl;
    cout << "check: box_tri_overlap_test on device" << endl;

    sycl::queue q;

    const int num_tests = 1000000;
    vector<int> host_results(num_tests, 0);
    vector<double> host_tri_points(num_tests * 9);

    rander seed;
    auto R = [&seed]() { return 5 * seed() - 2; };

    for (int i = 0; i < num_tests; i++) {
        for (int j = 0; j < 9; j++) {
            host_tri_points[i * 9 + j] = R();
        }
    }

    int* device_results = sycl::malloc_device<int>(num_tests, q);
    double* device_tri_points = sycl::malloc_device<double>(num_tests * 9, q);

    q.memcpy(device_tri_points, host_tri_points.data(), host_tri_points.size() * sizeof(double)).wait();

    int n_loop = 100;

    Tic("box_tri_overlap")
    for (int i = 0; i < n_loop; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tests), [=](sycl::id<1> idx) {
                int i = idx[0];
                bool result = box_tri_overlap_test(
                    device_tri_points[i * 9 + 0], device_tri_points[i * 9 + 1], device_tri_points[i * 9 + 2],
                    device_tri_points[i * 9 + 3], device_tri_points[i * 9 + 4], device_tri_points[i * 9 + 5],
                    device_tri_points[i * 9 + 6], device_tri_points[i * 9 + 7], device_tri_points[i * 9 + 8],
                    0, 0, 0, 1, 1, 1
                );
                device_results[i] = result ?1 : 0;
            });
        }).wait();
    }
    Toc

    q.memcpy(host_results.data(), device_results, num_tests * sizeof(int)).wait();

    int correct = 0;
    for (int i = 0; i < num_tests; i++) {
        bool host_result = box_tri_overlap_test(
            host_tri_points[i * 9 + 0], host_tri_points[i * 9 + 1], host_tri_points[i * 9 + 2],
            host_tri_points[i * 9 + 3], host_tri_points[i * 9 + 4], host_tri_points[i * 9 + 5],
            host_tri_points[i * 9 + 6], host_tri_points[i * 9 + 7], host_tri_points[i * 9 + 8],
            0, 0, 0, 1, 1, 1
        );

        if (host_results[i] == (host_result ? 1 : 0)) {
            correct++;
        } else {
            cout << "Mismatch at test " << i << ": device=" << host_results[i] << ", host=" << (host_result ? 1 : 0) << endl;
        }
    }

    cout << "box_tri_overlap device test: " << correct << "/" << num_tests << " passed" << endl;
    if (correct == num_tests) {
        cout << "✅ All box_tri_overlap device tests passed!" << endl;
    } else {
        cout << "❌ Some box_tri_overlap device tests failed!" << endl;
    }

    cout << num_tests * n_loop / TimeUsed("box_tri_overlap") << " box-tri overlap tests per second" << endl;

    sycl::free(device_results, q);
    sycl::free(device_tri_points, q);
}

void test_line_seg_box_overlap_on_device() {
    cout << "==========================================" << endl;
    cout << "check: line_seg_box_overlap_test on device" << endl;

    sycl::queue q;

    const int num_tests = 1000000;
    vector<int> host_results(num_tests, 0);
    vector<double> host_seg_points(num_tests * 6);

    rander seed;
    auto R = [&seed]() { return seed() * 5 - 2; };

    for (int i = 0; i < num_tests; i++) {
        for (int j = 0; j < 6; j++) {
            host_seg_points[i * 6 + j] = R();
        }
    }

    int* device_results = sycl::malloc_device<int>(num_tests, q);
    double* device_seg_points = sycl::malloc_device<double>(num_tests * 6, q);

    q.memcpy(device_seg_points, host_seg_points.data(), host_seg_points.size() * sizeof(double)).wait();

    int n_loop = 100;

    Tic("line_seg_box_overlap")
    for (int i = 0; i < n_loop; ++i) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(num_tests), [=](sycl::id<1> idx) {
                int i = idx[0];
                bool result = line_seg_box_overlap_test(
                    0, 0, 0, 1, 1, 1,
                    device_seg_points[i * 6 + 0], device_seg_points[i * 6 + 1], device_seg_points[i * 6 + 2],
                    device_seg_points[i * 6 + 3], device_seg_points[i * 6 + 4], device_seg_points[i * 6 + 5]
                );
                device_results[i] = result ?1 : 0;
            });
        }).wait();
    }
    Toc

    q.memcpy(host_results.data(), device_results, num_tests * sizeof(int)).wait();

    int correct = 0;
    for (int i = 0; i < num_tests; i++) {
        bool host_result = line_seg_box_overlap_test(
            0, 0, 0, 1, 1, 1,
            host_seg_points[i * 6 + 0], host_seg_points[i * 6 + 1], host_seg_points[i * 6 + 2],
            host_seg_points[i * 6 + 3], host_seg_points[i * 6 + 4], host_seg_points[i * 6 + 5]
        );

        if (host_results[i] == (host_result ? 1 : 0)) {
            correct++;
        } else {
            cout << "Mismatch at test " << i << ": device=" << host_results[i] << ", host=" << (host_result ? 1 : 0) << endl;
        }
    }

    cout << "line_seg_box_overlap device test: " << correct << "/" << num_tests << " passed" << endl;
    if (correct == num_tests) {
        cout << "✅ All line_seg_box_overlap device tests passed!" << endl;
    } else {
        cout << "❌ Some line_seg_box_overlap device tests failed!" << endl;
    }

    cout << num_tests * n_loop / TimeUsed("line_seg_box_overlap") << " ray-box overlap tests per second" << endl;

    sycl::free(device_results, q);
    sycl::free(device_seg_points, q);
}

int main() {
    test_trace_to_idxs_on_device();
    test_line_seg_tri_intersect_on_device();
    test_box_tri_overlap_on_device();
    test_line_seg_box_overlap_on_device();
    return 0;
}
