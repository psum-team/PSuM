#include <psum/serialization.hpp>
#include <psum/particle_container.hpp>
#include <psum/tag.hpp>
#include <iostream>

using namespace std;
using namespace psum::serialization;
using namespace psum::particle_container;

bool test_vector_fundamental(sycl::queue& q) {
    try {
        std::cout << "test fundamental vector..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // device_vector<int>
        {
            device_vector<int> int_vec(q, vector<int>{1, 2, 3, 4, 5});
            psum::serialization::save(fp, "int_vector", int_vec);

            device_vector<int> loaded_int_vec(q);
            psum::serialization::load(fp, "int_vector", loaded_int_vec);

            assert(int_vec.to_host() == loaded_int_vec.to_host());
            std::cout << "ok.." << std::endl;
        }
        // device_vector<size_t>
        {
            device_vector<size_t> int_vec(q, vector<size_t>{1, 2, 3, 4, 5});
            psum::serialization::save(fp, "uint64_vector", int_vec);

            device_vector<size_t> loaded_int_vec(q);
            psum::serialization::load(fp, "uint64_vector", loaded_int_vec);

            assert(int_vec.to_host() == loaded_int_vec.to_host());
            std::cout << "ok.." << std::endl;
        }
        // device_vector<double>
        {
            device_vector<double> int_vec(q, vector<double>{1.1, 2.2, 3.3, 4.4});
            psum::serialization::save(fp, "double_vector", int_vec);

            device_vector<double> loaded_int_vec(q);
            psum::serialization::load(fp, "double_vector", loaded_int_vec);

            assert(int_vec.to_host() == loaded_int_vec.to_host());
            std::cout << "ok.." << std::endl;
        }
        // device_vector<bool>
        {
            device_vector<bool> bool_vec(q, vector<bool>{true, false, true, false, true});
            psum::serialization::save(fp, "bool_vector", bool_vec);

            device_vector<bool> loaded_bool_vec(q);
            psum::serialization::load(fp, "bool_vector", loaded_bool_vec);

            assert(bool_vec.to_host() == loaded_bool_vec.to_host());
            std::cout << "ok.." << std::endl;
        }
        // empty device_vector
        {
            device_vector<float> empty_vec(q);
            psum::serialization::save(fp, "empty_vector", empty_vec);

            device_vector<float> loaded_empty_vec(q);
            psum::serialization::load(fp, "empty_vector", loaded_empty_vec);

            assert(loaded_empty_vec.size() == 0);
            std::cout << "ok.." << std::endl;
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_vector_Eigen(sycl::queue& q) {
    try {
        std::cout << "test Eigen vector..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // vector<RowVector>
        {
            vector<Eigen::RowVector3d> vec(10);
            for (auto& v : vec) {
                v = Eigen::RowVector3d::Random();
            }
            device_vector<Eigen::RowVector3d> vec_dev(q, vec);
            psum::serialization::save(fp, "RowVector_vec", vec_dev);

            device_vector<Eigen::RowVector3d> loaded_vec_dev(q);
            psum::serialization::load(fp, "RowVector_vec", loaded_vec_dev);

            assert(vec == loaded_vec_dev.to_host());
        }
        // vector<Vector>
        {
            vector<Eigen::Vector3d> vec(10);
            for (auto& v : vec) {
                v = Eigen::Vector3d::Random();
            }
            device_vector<Eigen::Vector3d> vec_dev(q, vec);
            psum::serialization::save(fp, "Vector_vec", vec_dev);

            device_vector<Eigen::Vector3d> loaded_vec_dev(q);
            psum::serialization::load(fp, "Vector_vec", loaded_vec_dev);

            assert(vec == loaded_vec_dev.to_host());
        }
        // vector<Matrix33>
        {
            vector<Eigen::Matrix<double, 3, 3>> vec(10);
            for (auto& m : vec) {
                m = Eigen::Matrix<double, 3, 3>::Random();
            }
            device_vector<Eigen::Matrix<double, 3, 3>> vec_dev(q, vec);
            psum::serialization::save(fp, "Matrix33_vec", vec_dev);

            device_vector<Eigen::Matrix<double, 3, 3>> loaded_vec_dev(q);
            psum::serialization::load(fp, "Matrix33_vec", loaded_vec_dev);

            assert(vec == loaded_vec_dev.to_host());
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void run_all_tests(sycl::queue& q) {

    cout << "=== unit test: masIO(container S/L) ===" << endl;
    cout << "====================" << endl;
    int passed = 0;
    int total = 0;
    auto run_test = [&](bool (*test_func)(sycl::queue&)) {
        total++;
        if (test_func(q)) {
            cout << " pass" << endl;
            passed++;
        } else {
            cout << " fail" << endl;
        }
    };
    
    run_test(test_vector_fundamental);
    run_test(test_vector_Eigen);
    
    cout << "====================" << std::endl;
    cout << "test result: " << passed << "/" << total << ", pass rate = " << 100.0 * passed / total << "%" << std::endl;
    
    if (passed == total) {
        std::cout << "All test passed." << std::endl;
    } else {
        std::cout << "Some test failed." << std::endl;
    }
}

// 主测试运行函数
int main() {

    sycl::queue q{ sycl::default_selector_v };

    std::cout << "Running particle_group test on device: "
              << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    run_all_tests(q);
    return 0;
}
