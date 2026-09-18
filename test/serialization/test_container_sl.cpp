#include <psum/serialization.hpp>
#include <psum/tag.hpp>
#include <iostream>

using namespace std;
using namespace psum::serialization;

bool test_vector_fundamental() {
    try {
        std::cout << "test fundamental vector..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // vector<int>
        {
            vector<int> int_vec = {1, 2, 3, 4, 5};
            psum::serialization::save(fp, "int_vector", int_vec);

            vector<int> loaded_int_vec;
            psum::serialization::load(fp, "int_vector", loaded_int_vec);

            assert(int_vec == loaded_int_vec);
        }
        // vector<size_t>
        {
            vector<size_t> int_vec = {1, 2, 3, 4, 5};
            psum::serialization::save(fp, "uint64_vector", int_vec);

            vector<size_t> loaded_int_vec;
            psum::serialization::load(fp, "uint64_vector", loaded_int_vec);

            assert(int_vec == loaded_int_vec);
        }
        // vector<double>
        {
            vector<double> int_vec = {1.1, 2.2, 3.3, 4.4};
            psum::serialization::save(fp, "double_vector", int_vec);

            vector<double> loaded_int_vec;
            psum::serialization::load(fp, "double_vector", loaded_int_vec);

            assert(int_vec == loaded_int_vec);
        }
        // vector<bool>
        {
            vector<bool> bool_vec = {true, false, true, false, true};
            psum::serialization::save(fp, "bool_vector", bool_vec);

            vector<bool> loaded_bool_vec;
            psum::serialization::load(fp, "bool_vector", loaded_bool_vec);

            assert(bool_vec == loaded_bool_vec);
        }
        // empty vector
        {
            vector<float> empty_vec;
            psum::serialization::save(fp, "empty_vector", empty_vec);

            vector<float> loaded_empty_vec;
            psum::serialization::load(fp, "empty_vector", loaded_empty_vec);

            assert(loaded_empty_vec.empty());
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_vector_Eigen() {
    try {
        std::cout << "test Eigen vector..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // vector<RowVector>
        {
            vector<Eigen::RowVector3d> vec(10);
            for (auto& v : vec) {
                v = Eigen::RowVector3d::Random();
            }
            psum::serialization::save(fp, "RowVector_vec", vec);

            vector<Eigen::RowVector3d> loaded_vec;
            psum::serialization::load(fp, "RowVector_vec", loaded_vec);

            assert(vec == loaded_vec);
        }
        // vector<Vector>
        {
            vector<Eigen::Vector3d> vec(10);
            for (auto& v : vec) {
                v = Eigen::Vector3d::Random();
            }
            psum::serialization::save(fp, "Vector_vec", vec);

            vector<Eigen::Vector3d> loaded_vec;
            psum::serialization::load(fp, "Vector_vec", loaded_vec);

            assert(vec == loaded_vec);
        }
        // vector<VectorX>
        {
            vector<Eigen::VectorXd> vec(10);
            for (auto& v : vec) {
                v.resize(rand() % 10 + 1);
                v = Eigen::VectorXd::Random(v.size());
            }
            psum::serialization::save(fp, "VectorXd_vec", vec);

            vector<Eigen::VectorXd> loaded_vec;
            psum::serialization::load(fp, "VectorXd_vec", loaded_vec);

            assert(vec == loaded_vec);
        }
        // vector<Matrix33>
        {
            vector<Eigen::Matrix<double, 3, 3>> vec(10);
            for (auto& m : vec) {
                m = Eigen::Matrix<double, 3, 3>::Random();
            }
            psum::serialization::save(fp, "Matrix33_vec", vec);

            vector<Eigen::Matrix<double, 3, 3>> loaded_vec;
            psum::serialization::load(fp, "Matrix33_vec", loaded_vec);

            assert(vec == loaded_vec);
        }
        // vector<MatrixX>
        {
            vector<Eigen::MatrixXd> vec(10);
            for (auto& m : vec) {
                m.resize(rand() % 10 + 1, rand() % 10 + 1);
                m = Eigen::MatrixXd::Random(m.rows(), m.cols());
            }
            psum::serialization::save(fp, "Matrix33_vec", vec);

            vector<Eigen::MatrixXd> loaded_vec;
            psum::serialization::load(fp, "Matrix33_vec", loaded_vec);

            assert(vec == loaded_vec);
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_vector_string() {
    try {
        std::cout << "test string vector..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // vector<string>
        {
            vector<std::string> str_vec = {"hello", "world", "test", "string", ""};
            psum::serialization::save(fp, "string_vector", str_vec);

            vector<std::string> loaded_str_vec;
            psum::serialization::load(fp, "string_vector", loaded_str_vec);

            assert(str_vec == loaded_str_vec);
        }
        // empty vector<string>
        {
            vector<std::string> empty_str_vec;
            psum::serialization::save(fp, "empty_string_vector", empty_str_vec);

            vector<std::string> loaded_empty_str_vec;
            psum::serialization::load(fp, "empty_string_vector", loaded_empty_str_vec);

            assert(loaded_empty_str_vec.empty());
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_map_types() {
    try {
        std::cout << "test map..." << std::endl;

        mas_file fp("tmp.mas", mas_file::replaceMode);

        // map
        {
            std::map<int, std::string> test_map = {
                {1, "one"},
                {2, "two"},
                {3, "three"}
            };

            psum::serialization::save(fp, "test_map", test_map);

            std::map<int, std::string> loaded_map;
            psum::serialization::load(fp, "test_map", loaded_map);

            assert(test_map == loaded_map);
        }
        // unordered_map
        {
            std::unordered_map<std::string, double> test_unordered_map = {
                {"apple", 1.5},
                {"banana", 2.3},
                {"cherry", 0.8}
            };

            psum::serialization::save(fp, "test_unordered_map", test_unordered_map);

            std::unordered_map<std::string, double> loaded_unordered_map;
            psum::serialization::load(fp, "test_unordered_map", loaded_unordered_map);

            assert(test_unordered_map == loaded_unordered_map);
        }
        // empty map
        {
            std::map<int, int> empty_map;
            psum::serialization::save(fp, "empty_map", empty_map);

            std::map<int, int> loaded_empty_map;
            psum::serialization::load(fp, "empty_map", loaded_empty_map);

            assert(loaded_empty_map.empty());
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool test_error_handling() {
    std::cout << "test error handling..." << std::endl;

    mas_file fp("tmp.mas", mas_file::replaceMode);
    try {
        bool load_fail = false;
        try {
            vector<int> vec;
            psum::serialization::load(fp, "non_existent", vec);
            load_fail = true;
        } catch (const std::runtime_error&) {
            load_fail = true;
        }
        return load_fail;
    } catch (const std::exception& e) {
        return false;
    }
}

void run_all_tests() {

    cout << "=== unit test: masIO(container S/L) ===" << endl;
    cout << "====================" << endl;
    int passed = 0;
    int total = 0;
    auto run_test = [&](bool (*test_func)()) {
        total++;
        if (test_func()) {
            cout << " pass" << endl;
            passed++;
        } else {
            cout << " fail" << endl;
        }
    };
    
    run_test(test_vector_fundamental);
    run_test(test_vector_string);
    run_test(test_vector_Eigen);
    run_test(test_map_types);
    run_test(test_error_handling);
    
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
    run_all_tests();
    return 0;
}
