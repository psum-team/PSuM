#include <psum/serialization.hpp>

using namespace psum::serialization;
using json = nlohmann::json;

#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>

// 固定随机种子
std::mt19937 rng(12345);

// 随机整数
int rand_int(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

// 随机浮点
double rand_double(double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

// 随机布尔
bool rand_bool() {
    return rand_int(0, 1) == 1;
}

// 随机字符串
std::string rand_string(size_t min_len = 3, size_t max_len = 12) {
    static const char charset[] =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string s;
    size_t len = rand_int(min_len, max_len);
    s.reserve(len);
    for (size_t i = 0; i < len; ++i)
        s += charset[rand_int(0, sizeof(charset) - 2)];
    return s;
}

// 递归生成随机 JSON
json random_json(int depth = 0, int max_depth = 4, int max_obj_size = 5, int max_array_size = 5) {
    if (depth >= max_depth) {
        // 基础类型
        int choice = rand_int(0, 3);
        switch(choice) {
            case 0: return rand_int(-1000, 1000);
            case 1: return rand_double(-1000.0, 1000.0);
            case 2: return rand_bool();
            case 3: return rand_string();
        }
    }

    int choice = rand_int(0, 5);
    if (choice < 2) { // 对象
        json j = json::object();
        int n = rand_int(1, max_obj_size);
        for (int i = 0; i < n; ++i) {
            j[rand_string()] = random_json(depth + 1, max_depth, max_obj_size, max_array_size);
        }
        return j;
    } else if (choice < 4) { // 数组
        json j = json::array();
        int n = rand_int(1, max_array_size);
        for (int i = 0; i < n; ++i) {
            j.push_back(random_json(depth + 1, max_depth, max_obj_size, max_array_size));
        }
        return j;
    } else { // 基础类型
        switch(rand_int(0, 3)) {
            case 0: return rand_int(-1000, 1000);
            case 1: return rand_double(-1000.0, 1000.0);
            case 2: return rand_bool();
            case 3: return rand_string();
        }
    }
    return nullptr; // fallback
}

// 生成嵌套对象
json make_nested_object(int depth, int breadth) {
    json j;
    breadth = std::max(1, breadth);

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < breadth; ++i) {
        std::string key = "obj_" + std::to_string(i);
        if (depth > 0) {
            if (dist(rng) < 0.7) {
                j[key] = make_nested_object(depth - 1, breadth / 2 + 1);
            } else {
                json arr = json::array();
                for (int k = 0; k < 3; ++k) arr.push_back(rand_string());
                j[key] = arr;
            }
        } else {
            double r = dist(rng);
            if (r < 0.25)
                j[key] = rand_string();
            else if (r < 0.5)
                j[key] = static_cast<int>(dist(rng) * 1000);
            else if (r < 0.75)
                j[key] = dist(rng);
            else
                j[key] = (dist(rng) > 0.5);
        }
    }
    return j;
}

// 生成大型数组
json make_large_array(int n) {
    json arr = json::array();
    for (int i = 0; i < n; ++i) {
        arr.push_back({
            {"id", i},
            {"value", std::sin(i * 0.01)},
            {"flag", (i % 3 == 0)}
        });
    }
    return arr;
}

// 递归验证 loader 与 json 是否一致

bool compare_json_loader_array(const json& val, json_loader::key_condition loader);

bool compare_json_loader(const json& j, auto&& loader) {
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::string key = it.key();
        const auto& val = it.value();

        if (val.is_object()) {
            if (!compare_json_loader(val, loader.in(key))) return false;
        } else if (val.is_array()) {
            if (!compare_json_loader_array(val, loader.in(key))) return false;
        } else {
            if (val.is_number_integer() && loader.template obj<int>(key) != val.get<int>()) return false;
            if (val.is_number_float() && std::abs(loader.template obj<double>(key) - val.get<double>()) > 1e-12) return false;
            if (val.is_boolean() && loader.template obj<bool>(key) != val.get<bool>()) return false;
            if (val.is_string() && loader.template obj<std::string>(key) != val.get<std::string>()) return false;
        }
    }
    return true;
}

bool compare_json_loader_array(const json& val, json_loader::key_condition loader) {
    if (val.size() != 0) {
        // check for pure array
        uint8_t type_code = (val.at(0).is_boolean() << 0) | (val.at(0).is_number_float() << 1) | (val.at(0).is_number_integer() << 2);
        bool same_type = true;
        for (auto &element : val)
            same_type &= (type_code == ((element.is_boolean() << 0) | (element.is_number_float() << 1) | (element.is_number_integer() << 2)));
        bool pure_array = (type_code != 0) && same_type;

        if (pure_array) {
            if (val[0].is_number_float()) {
                std::vector<double> data = loader.template obj<std::vector<double>>();
                if (val.size() != data.size()) return false;
                for (size_t i = 0; i < val.size(); ++i) {
                    if (std::abs(val[i].get<double>() - data[i]) > 1e-12) return false;
                }
            } else if (val[0].is_number_integer()) {
                std::vector<int> data = loader.template obj<std::vector<int>>();
                if (val.size() != data.size()) return false;
                for (size_t i = 0; i < val.size(); ++i) {
                    if (val[i].get<int>() != data[i]) return false;
                }
            } else if (val[0].is_boolean()) {
                std::vector<bool> data = loader.template obj<std::vector<bool>>();
                if (val.size() != data.size()) return false;
                for (size_t i = 0; i < val.size(); ++i) {
                    if (val[i].get<bool>() != data[i]) return false;
                }
            }
        } else {
            if (val.size() != loader.idxs().size()) throw std::runtime_error("array size not match");

            for (size_t i = 0; i < val.size(); ++i) {
                if (val[i].is_object()) {
                    if (!compare_json_loader(val[i], loader.at(i))) return false;
                } else if (val[i].is_string()) {
                    if (val[i].get<std::string>() != loader.at(i).template obj<std::string>()) return false;
                } else if (val[i].is_number_integer()) {
                    if (val[i].get<int>() != loader.at(i).template obj<int>()) return false;
                } else if (val[i].is_number_float()) {
                    if (std::abs(val[i].get<double>() - loader.at(i).template obj<double>()) > 1e-12) return false;
                } else if (val[i].is_boolean()) {
                    if (val[i].get<bool>() != loader.at(i).template obj<bool>()) return false;
                } else if (val[i].is_array()) {
                    if (!compare_json_loader_array(val[i], loader.at(i))) return false;
                } else {
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    const int test_repeat = 1000;
    for (int iter = 0; iter < test_repeat; ++iter) {
        // =====================
        // 构建复杂 JSON
        // =====================
        json j;
        j["version"] = 42;
        j["meta"] = { {"author", "unit_test"}, {"time", "2025-10-07"} };
        j["deep"] = make_nested_object(4, 5);
        j["big_array"] = make_large_array(200);
        j["matrix"] = {
            {"row0", {1,2,3}},
            {"row1", {4,5,6}},
            {"row2", {7,8,9}}
        };
        j["random_object"] = random_json(0, 4, 5, 5);

        // 写入文件
        std::ofstream("test_big.json") << j.dump(2);

        // =====================
        // 加载并测试
        // =====================
        json_loader loader;
        
        auto start = std::chrono::high_resolution_clock::now();
        loader.load_json("test_big.json");
        auto end = std::chrono::high_resolution_clock::now();
        
        loader.save("json_test.mas");

        // 断言基础值
        assert(loader.obj<int>("version") == 42);
        assert(loader.in("meta").obj<std::string>("author") == "unit_test");

        // 递归对比
        assert(compare_json_loader(j, loader));

        // 异常测试
        try {
            loader.in("non_exist").obj<int>("abc");
            assert(false); // 应该抛异常
        } catch (...) {}

        try {
            loader.in("big_array").at(20000);
            assert(false); // 应该抛异常
        } catch (...) {}

        // 可选性能测试（仅第一轮）
        if (iter == test_repeat / 4) {
            std::ifstream f("test_big.json", std::ios::binary | std::ios::ate);
            size_t size_bytes = f.tellg();
            std::function<size_t(const json&)> count_all_keys = [&](const json& j) -> size_t {
                if (!j.is_object()) return 0;
                size_t cnt = j.size();
                for (auto& el : j.items()) cnt += count_all_keys(el.value());
                return cnt;
            };
            size_t n_keys = count_all_keys(j);

            std::cout << "[Performance] Phase json(" << size_bytes/1024.0 << " KB, " << n_keys << " keys) loading took "
                      << std::chrono::duration<double, std::milli>(end - start).count()
                      << " ms\n";
        }
    }

    std::cout << "All " << test_repeat << " tests passed!\n";
}
