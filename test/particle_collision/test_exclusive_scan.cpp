#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <algorithm>
#include <sycl/sycl.hpp>

#include "../../src/particle_collision/device_exclusive_scan.hpp" 

using namespace psum::particle_collision;

template <typename T>
bool verify(const std::vector<T>& input, const std::vector<T>& gpu_result, T gpu_total_sum) {
    std::vector<T> cpu_result(input.size());
    T current_sum = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        cpu_result[i] = current_sum;
        current_sum += input[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        if (cpu_result[i] != gpu_result[i]) {
            std::cout << "Verification Failed at index " << i 
                      << ": CPU=" << cpu_result[i] << ", GPU=" << gpu_result[i] << std::endl;
            return false;
        }
    }

    if (current_sum != gpu_total_sum) {
        std::cout << "Total Sum Verification Failed: CPU=" << current_sum 
                  << ", GPU=" << gpu_total_sum << std::endl;
        return false;
    }

    return true;
}

template <typename T>
void run_test(sycl::queue& q, device_exclusive_scan_context<T>& ctx, size_t N, const std::string& label) {
    std::cout << "Testing [" << label << "] with N = " << N << " ... ";

    std::vector<T> h_input(N);
    std::generate(h_input.begin(), h_input.end(), [=]() { return static_cast<T>(rand() % 10); });

    T* d_input = sycl::malloc_device<T>(N, q);
    T* d_output = sycl::malloc_device<T>(N, q);
    q.memcpy(d_input, h_input.data(), N * sizeof(T)).wait();

    T total_sum = exclusive_scan_device<T>(ctx, d_input, d_output, N);

    std::vector<T> h_output(N);
    q.memcpy(h_output.data(), d_output, N * sizeof(T)).wait();

    if (verify(h_input, h_output, total_sum)) {
        std::cout << "PASSED" << std::endl;
    } else {
        std::cout << "FAILED" << std::endl;
    }

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

template <typename T>
void run_benchmark(sycl::queue& q, device_exclusive_scan_context<T>& ctx, size_t N) {
    std::vector<T> h_input(N, static_cast<T>(1));
    T* d_input = sycl::malloc_device<T>(N, q);
    T* d_output = sycl::malloc_device<T>(N, q);
    q.memcpy(d_input, h_input.data(), N * sizeof(T)).wait();

    exclusive_scan_device<T>(ctx, d_input, d_output, N);

    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        exclusive_scan_device<T>(ctx, d_input, d_output, N);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;

    double avg_time = ms.count() / iterations;
    double throughput = (N * sizeof(T) * 2) / (avg_time / 1000.0) / 1e9; // GB/s (Read + Write)

    std::cout << "Benchmark N=" << N << ": Avg Time = " << avg_time << " ms, "
              << "Throughput = " << throughput << " GB/s" << std::endl;

    sycl::free(d_input, q);
    sycl::free(d_output, q);
}

template <typename T>
void run_stress_test(sycl::queue& q, device_exclusive_scan_context<T>& ctx) {
    std::cout << "Starting Stress Test..." << std::endl;
    for (int i = 0; i < 100; ++i) {
        size_t N = 1000 + (rand() % 100000);
        std::vector<T> h_in(N, 1);
        T* d_in = sycl::malloc_device<T>(N, q);
        T* d_out = sycl::malloc_device<T>(N, q);
        q.memcpy(d_in, h_in.data(), N * sizeof(T)).wait();

        T total = exclusive_scan_device<T>(ctx, d_in, d_out, N);
        
        if (total != (T)N) {
            std::cout << "Stress Test Failed at Iteration " << i << " N=" << N << std::endl;
            return;
        }
        sycl::free(d_in, q);
        sycl::free(d_out, q);
    }
    std::cout << "Stress Test PASSED." << std::endl;
}

int main() {
    try {
        sycl::queue q{sycl::default_selector_v};
        std::cout << "Running on: " << q.get_device().get_info<sycl::info::device::name>() << "\n" << std::endl;

        device_exclusive_scan_context<size_t> ctx;
        ctx.init(q);

        run_test(q, ctx, 1, "Tiny: 1 element");
        run_test(q, ctx, 255, "Edge: Just under WG_SIZE");
        run_test(q, ctx, 256, "Edge: Exactly WG_SIZE");
        run_test(q, ctx, 257, "Edge: Just over WG_SIZE");
        
        run_test(q, ctx, 10000, "Medium: Multi-block");
        run_test(q, ctx, 256 * 256 + 1, "Large: Multi-level recursion");

        run_stress_test(q, ctx);

        std::cout << "\n--- Performance Benchmarks ---" << std::endl;
        run_benchmark(q, ctx, 1 << 20); // 1M
        run_benchmark(q, ctx, 1 << 24); // 16M

        ctx.release();
    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}