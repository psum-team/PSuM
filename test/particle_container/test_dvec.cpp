#include <psum/particle_container.hpp>
#include <iostream>
#include <vector>

using psum::particle_container::device_vector;

void test_device_access(sycl::queue& q, size_t n_elements) {
    device_vector<int> d_vec(q, n_elements);

    std::cout << "\nDevice access test" << std::endl;
    std::cout << "Capacity: " << d_vec.capacity() << std::endl;
    
    q.submit(
        [&](sycl::handler &cgh){
            auto acc = d_vec.get_access(cgh);
            cgh.parallel_for(sycl::range<1>(n_elements), [=](sycl::id<1> idx) {
                if (idx.get(0) % 2 == 1)
                    acc[idx.get(0)] = idx.get(0) * 2;
                else
                    acc[idx.get(0)] = idx.get(0);
            });
        }
    ).wait();

    std::vector<int> h_vec(n_elements);
    q.memcpy(h_vec.data(), d_vec.data(), n_elements * sizeof(int)).wait();
    
    bool correct = true;
    for (size_t i = 0; i < n_elements; ++i) {
        int expected = (i % 2 == 1) ? i * 2 : i;
        if (h_vec[i] != expected) {
            std::cerr << "Error: Index " << i << ", Expected: " << expected << ", Got: " << h_vec[i] << std::endl;
            correct = false;
            break;
        }
    }
    if (correct)
        std::cout << "✅ Device access test passed." << std::endl;
    else
        std::cout << "❌ Device access test failed." << std::endl;
}

void test_concurrent_push_back(sycl::queue& q, size_t n_threads, size_t capacity, size_t expected_size) {
    // expected: n_threads >= expected_size, capacity >= expected_size
    assert(n_threads >= expected_size);
    assert(capacity >= expected_size);
    psum::particle_container::device_vector<int> d_vec(q, capacity);
    
    std::cout << "\nBasic push_back test" << std::endl;
    std::cout << "Size before push_back: " <<  d_vec.size() << "/" << d_vec.capacity() << std::endl;
    q.submit([&](sycl::handler& cgh) {
        auto acc = d_vec.get_access(cgh);
        cgh.parallel_for(sycl::range<1>(n_threads), [=](sycl::id<1> idx) {
            if (idx.get(0) < expected_size)
                acc.push_back((int)idx.get(0));
        });
    }).wait();
    std::cout << "Size after push_back: " << d_vec.size() << "/" << d_vec.capacity() << std::endl;
    if (d_vec.size() == std::min(capacity, expected_size)) {
        std::cout << "✅ Basic push_back test passed." << std::endl;
    } else {
        std::cout << "❌ Basic push_back test failed." << std::endl;
    }
    std::vector<int> host_data(expected_size);
    q.memcpy(host_data.data(), d_vec.data(), std::min(capacity, expected_size) * sizeof(int)).wait();
    std::sort(host_data.begin(), host_data.end());
    for (size_t i = 0; i < std::min(capacity, expected_size); i++) {
        if (host_data[i] != i) {
            std::cout << "❌ Data corruption detected in basic push_back test." << std::endl;
            return;
        }
    }
    std::cout << "✅ Data corruption check passed in basic push_back test." << std::endl;
}

void test_concurrent_push_back_overflow(sycl::queue& q, size_t capacity, size_t expected_size) {
    psum::particle_container::device_vector<int> d_vec(q, capacity);
    
    std::cout << "\nPush_back overflow test" << std::endl;
    std::cout << "Size before push_back: " <<  d_vec.size() << "/" << d_vec.capacity() << std::endl;

    q.submit([&](sycl::handler& h) {
        auto acc = d_vec.get_access(h);
        h.parallel_for(sycl::range<1>(capacity + 1), [=](sycl::id<1> i) {
            acc.push_back(100 + i[0]);
        });
    }).wait();

    try {
        int size = d_vec.size();
        std::cout << "❌ test failed: overflowed() should throw an exception, but it returned." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✅ Overflow detection test passed." << std::endl;
    }
}

void test_reserve_resize(sycl::queue& q) {
    std::cout << "\nReserve/resize test" << std::endl;
    device_vector<int> dvec2(q, 4);
    std::cout << "d_vec before reserve(10): " <<  dvec2.size() << "/" << dvec2.capacity();
    if (dvec2.size() == 0 && dvec2.capacity() == 4) std::cout << "✅" << std::endl;
    else std::cout << "❌" << std::endl;

    dvec2.reserve(10);
    std::cout << "d_vec after reserve(10): " << dvec2.size() << "/" << dvec2.capacity();
    if (dvec2.size() == 0 && dvec2.capacity() == 10) std::cout << "✅" << std::endl;
    else std::cout << "❌" << std::endl;

    q.memcpy(dvec2.data(), std::vector<int>{1,2,3,4}.data(), 4*sizeof(int)).wait();
    dvec2.resize(4);
    std::cout << "d_vec after resize(4): " << dvec2.size() << "/" << dvec2.capacity();
    if (dvec2.size() == 4 && dvec2.capacity() == 10) std::cout << "✅" << std::endl;
    else std::cout << "❌" << std::endl;

    q.submit([&](sycl::handler& h) {
        auto acc = dvec2.get_access(h);
        h.parallel_for(sycl::range<1>(3), [=](sycl::id<1> i) {
            acc.push_back(50 + i[0]);
        });
    }).wait();

    size_t size = dvec2.size();
    std::vector<int> host_data(size);
    q.memcpy(host_data.data(), dvec2.data(), size * sizeof(int)).wait();
    std::sort(host_data.begin(), host_data.end());

    std::cout << "d_vec after push_back()x3: " << dvec2.size() << "/" << dvec2.capacity();
    if (dvec2.size() == 7 && dvec2.capacity() == 10) std::cout << "✅" << std::endl;
    else std::cout << "❌" << std::endl;

    if (host_data == std::vector<int>{1,2,3,4,50,51,52})
        std::cout << "✅ Data corruption check passed in reserve/resize test." << std::endl;
    else
        std::cout << "❌ Data corruption detected in reserve/resize test." << std::endl;
}

void test_move_semantics(sycl::queue& q) {
    std::cout << "\nMove semantics test" << std::endl;
    device_vector<int> a(q, 4);
    q.submit([&](sycl::handler& h) {
    auto acc = a.get_access(h);
        h.single_task([=]() {
            acc.push_back(99);
        });
    }).wait();
    int* old_ptr = a.data();

    device_vector<int> b = std::move(a);
    bool correct = (b.size() == 1 && b.capacity() == 4);
    std::vector<int> host_data(b.size());
    q.memcpy(host_data.data(), b.data(), b.size() * sizeof(int)).wait();
    for (size_t i = 0; i < b.size(); i++) {
            correct &= (host_data[i] == 99);
    }
    correct &= (b.data() == old_ptr);
    if (correct)
        std::cout << "✅ Move semantics test passed." << std::endl;
    else
        std::cout << "❌ Move semantics test failed." << std::endl;
}

void copy_between_host_and_device(sycl::queue& q, size_t n_elements) {
    std::cout << "\nCopy between host and device test" << std::endl;
    std::vector<int> host_data(n_elements);
    for (size_t i = 0; i < n_elements; i++) {
        host_data[i] = i;
    }
    device_vector<int> d_vec(q, host_data);
    auto d_vec_to_host = d_vec.to_host();
    if (d_vec_to_host.size() != host_data.size())
        std::cout << "❌ Copy between host and device test failed: size mismatch." << std::endl;
    if (d_vec.to_host() == host_data) {
        std::cout << "✅ Copy between host and device test passed." << std::endl;
    } else {
        std::cout << "❌ Copy between host and device test failed." << std::endl;
    }
}

void run_concurrent_test() {
    std::cout << "\nRunning concurrent test..." << std::endl;

    struct LogEntry {
        size_t vector_id;  
        size_t element_id;
    };

    sycl::queue q{sycl::default_selector_v};
    
    const size_t num_vectors = 8;
    const size_t ops_per_vector = 1e6;
    const size_t total_ops = num_vectors * ops_per_vector;

    std::vector<psum::particle_container::device_vector<int>> vec_list;
    for (size_t i = 0; i < num_vectors; ++i) {
        vec_list.emplace_back(q, ops_per_vector);
    }
    
    psum::particle_container::device_vector<LogEntry> total_vec(q, total_ops);
    std::vector<sycl::event> events;
    
    for (size_t v_id = 0; v_id < num_vectors; ++v_id) {
        auto e = q.submit([&, v_id](sycl::handler& h) {
            auto data_acc = vec_list.at(v_id).get_access_without_overflow_check(h);
            auto total_acc = total_vec.get_access_without_overflow_check(h);

            h.parallel_for(sycl::range<1>(ops_per_vector), [=](sycl::id<1> item) {
                int value = static_cast<int>(v_id * 100000 + item[0]);
                size_t returned_idx = data_acc.push_back(value);
                total_acc.push_back({v_id, returned_idx});
            });
        });

        events.push_back(e);
    }
    
    std::cout << "Launching " << num_vectors << " concurrent kernels..." << std::endl;
    sycl::event::wait(events);
    std::cout << "All kernels finished. Analyzing results..." << std::endl;
    auto host_logs = total_vec.to_host();

    size_t problems = 0;
    if (host_logs.size() != total_ops) {
        std::cerr << "❌ total_vec missing entries! Expected " << total_ops 
                  << " but got " << host_logs.size() << std::endl;
        ++problems;
    }

    std::vector<std::set<size_t>> per_vec_indices(num_vectors);
    for (const auto& log : host_logs) {
        if (log.vector_id >= num_vectors) {
            std::cerr << "❌ Invalid vector_id in logs!" << std::endl;
            ++problems;
            continue;
        }
        
        auto& s = per_vec_indices[log.vector_id];
        if (s.count(log.element_id)) {
            std::cerr << "❌ Duplicate index " << log.element_id 
                      << " returned for vector " << log.vector_id << "!" << std::endl;
            ++problems;
        }
        s.insert(log.element_id);
    }

    for (size_t i = 0; i < num_vectors; ++i) {
        if (vec_list[i].size() != ops_per_vector) {
            std::cerr << "❌ Vector " << i << " size is " << vec_list[i].size() 
                      << ", expected " << ops_per_vector << std::endl;
            ++problems;
        }
    }

    if (problems > 0) {
        std::cerr << "❌ concurrent push_back verification FAILED (" << problems << " problems)" << std::endl;
        std::abort();
    }
    std::cout << "✅ concurrent push_back verification PASSED" << std::endl;
}

void test_empty_for_each(sycl::queue& q) {
    device_vector<int> dvec(q, 4);
    if (dvec.size() != 0) { std::cout << "FAIL: dvec size != 0\n"; std::abort(); }
    dvec.for_each([&](sycl::handler&) {
        return [=](int& v) { (void)v; };
    });
    if (dvec.size() != 0) { std::cout << "FAIL: dvec size changed\n"; std::abort(); }
    std::cout << "empty for_each regression PASSED\n";
}

int main() {
    sycl::queue q;
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    test_empty_for_each(q);
    test_device_access(q, 1e6);
    test_concurrent_push_back(q, 5e5, 1e6, 4e5);
    test_concurrent_push_back_overflow(q, 1e6, 2e6);
    test_reserve_resize(q);
    test_move_semantics(q);
    copy_between_host_and_device(q, 1e6);
    run_concurrent_test();
    return 0;
}