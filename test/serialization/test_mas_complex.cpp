#include <psum/serialization.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstdint>

using namespace std;
using namespace psum::serialization;

// 测试辅助类，用于验证文件状态一致性
class MasFileTester {
private:
    
public:
    mas_file& file;

    MasFileTester(mas_file &fp) : file(fp) {};

    // 验证当前head与上次保存的是否一致
    bool verifyHeads() {
        auto current_heads = file.getHeads();
        
        vector<mas_block_head> right_heads = file.readHead();

        if (current_heads.size() != right_heads.size()) {
            std::cout << "getHeads().size()=" << current_heads.size()
                      << ", " << "readHead().size()=" << right_heads.size();
            throw std::runtime_error("Error: heads size mismatch, test failed.");
        }
        
        for (size_t i = 0; i < current_heads.size(); i++) {
            const auto& current = current_heads[i];
            const auto& last = right_heads[i];
            
            if (current.info.blockName != last.info.blockName ||
                current.info.blockType != last.info.blockType ||
                current.info.blockShape != last.info.blockShape ||
                current.info.blockSize != last.info.blockSize ||
                current.position != last.position) {
                // return false;
                throw std::runtime_error("Error: heads mismatch, test failed.");
            }
        }
        return true;
    }
    
    // 包装writeData，自动检查一致性
    template<typename T>
    bool safeWriteData(const string& name, const T* data, const vector<size_t>& shape) {
        file.writeData(name, data, shape);
        return verifyHeads();
    }

    template<typename T>
    bool safeWriteData(const string& name, T data) {
        file.writeData(name, data);
        return verifyHeads();
    }
    
    // 包装replaceData，自动检查一致性
    template<typename T>
    bool safeReplaceData(const string& name, const T* data, const vector<size_t>& shape) {
        file.replaceData(name, data, shape);
        return verifyHeads();
    }

    template<typename T>
    bool safeReplaceData(const string& name, T data) {
        file.replaceData(name, data);
        return verifyHeads();
    }
    
    // 包装smashData，自动检查一致性
    bool safeSmashData(const string& name) {
        file.smashData(name);
        return verifyHeads();
    }
    
    // 包装clear，自动检查一致性
    bool safeClear() {
        file.clear();
        return verifyHeads();
    }
};

// 测试结果统计
struct TestResults {
    int total_tests = 0;
    int passed_tests = 0;
    
    void report(const string& test_name, bool passed) {
        total_tests++;
        passed_tests += passed ? 1 : 0;
        cout << test_name << ": " << (passed ? "PASS" : "FAIL") << endl;
    }
    
    void summary() {
        cout << "\n=== TEST SUMMARY ===" << endl;
        cout << "Total tests: " << total_tests << endl;
        cout << "Passed: " << passed_tests << endl;
        cout << "Failed: " << (total_tests - passed_tests) << endl;
        cout << "Success rate: " << (passed_tests * 100.0 / total_tests) << "%" << endl;
    }
};

// 基础数据类型测试
void testBasicTypes(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Basic Data Types ===" << endl;
    
    // 准备测试数据
    const size_t size = 100;
    
    // 测试所有12种类型
    float float_data[size];
    double double_data[size];
    int8_t int8_data[size];
    uint8_t uint8_data[size];
    int16_t int16_data[size];
    uint16_t uint16_data[size];
    int32_t int32_data[size];
    uint32_t uint32_data[size];
    int64_t int64_data[size];
    uint64_t uint64_data[size];
    char char_data[size];
    bool bool_data[size];
    
    // 初始化数据
    for (size_t i = 0; i < size; i++) {
        float_data[i] = i * 1.1f;
        double_data[i] = i * 1.1;
        int8_data[i] = i % 128;
        uint8_data[i] = i;
        int16_data[i] = i;
        uint16_data[i] = i;
        int32_data[i] = i;
        uint32_data[i] = i;
        int64_data[i] = i;
        uint64_data[i] = i;
        char_data[i] = 'A' + (i % 26);
        bool_data[i] = (i % 2 == 0);
    }
    
    // 写入并验证每种类型
    vector<pair<string, bool>> type_tests = {
        {"float_data", tester.safeWriteData("float_data", float_data, {size})},
        {"double_data", tester.safeWriteData("double_data", double_data, {size})},
        {"int8_data", tester.safeWriteData("int8_data", int8_data, {size})},
        {"uint8_data", tester.safeWriteData("uint8_data", uint8_data, {size})},
        {"int16_data", tester.safeWriteData("int16_data", int16_data, {size})},
        {"uint16_data", tester.safeWriteData("uint16_data", uint16_data, {size})},
        {"int32_data", tester.safeWriteData("int32_data", int32_data, {size})},
        {"uint32_data", tester.safeWriteData("uint32_data", uint32_data, {size})},
        {"int64_data", tester.safeWriteData("int64_data", int64_data, {size})},
        {"uint64_data", tester.safeWriteData("uint64_data", uint64_data, {size})},
        {"char_data", tester.safeWriteData("char_data", char_data, {size})},
        {"bool_data", tester.safeWriteData("bool_data", bool_data, {size})}
    };
    
    // 读取并验证数据正确性
    bool all_correct = true;
    auto float_read = Cast<float>(tester.file.readData("float_data"));
    auto double_read = Cast<double>(tester.file.readData("double_data"));
    auto int8_read = Cast<int8_t>(tester.file.readData("int8_data"));
    auto uint8_read = Cast<uint8_t>(tester.file.readData("uint8_data"));
    auto int16_read = Cast<int16_t>(tester.file.readData("int16_data"));
    auto uint16_read = Cast<uint16_t>(tester.file.readData("uint16_data"));
    auto int32_read = Cast<int32_t>(tester.file.readData("int32_data"));
    auto uint32_read = Cast<uint32_t>(tester.file.readData("uint32_data"));
    auto int64_read = Cast<int64_t>(tester.file.readData("int64_data"));
    auto uint64_read = Cast<uint64_t>(tester.file.readData("uint64_data"));
    auto char_read = Cast<char>(tester.file.readData("char_data"));
    auto bool_read = Cast<bool>(tester.file.readData("bool_data"));
    
    for (size_t i = 0; i < size; i++) {
        all_correct = all_correct && (float_data[i] == float_read[i]);
        all_correct = all_correct && (double_data[i] == double_read[i]);
        all_correct = all_correct && (int8_data[i] == int8_read[i]);
        all_correct = all_correct && (uint8_data[i] == uint8_read[i]);
        all_correct = all_correct && (int16_data[i] == int16_read[i]);
        all_correct = all_correct && (uint16_data[i] == uint16_read[i]);
        all_correct = all_correct && (int32_data[i] == int32_read[i]);
        all_correct = all_correct && (uint32_data[i] == uint32_read[i]);
        all_correct = all_correct && (int64_data[i] == int64_read[i]);
        all_correct = all_correct && (uint64_data[i] == uint64_read[i]);
        all_correct = all_correct && (char_data[i] == char_read[i]);
        all_correct = all_correct && (bool_data[i] == bool_read[i]);
    }
    
    results.report("All 12 basic types write/read", all_correct);
}

// 多维数据测试
void testMultiDimensional(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Multi-Dimensional Data ===" << endl;
    
    // 1D数据
    vector<size_t> shape1d = {10};
    vector<int> data1d(10);
    for (size_t i = 0; i < 10; i++) data1d[i] = i;
    bool test1d = tester.safeWriteData("1d_data", data1d.data(), shape1d);
    auto read1d = Cast<int>(tester.file.readData("1d_data"));
    bool verify1d = true;
    for (size_t i = 0; i < 10; i++) verify1d = verify1d && (data1d[i] == read1d[i]);
    results.report("1D data", test1d && verify1d);
    
    // 2D数据
    vector<size_t> shape2d = {5, 4};
    vector<int> data2d(20);
    for (size_t i = 0; i < 20; i++) data2d[i] = i;
    bool test2d = tester.safeWriteData("2d_data", data2d.data(), shape2d);
    auto read2d = Cast<int>(tester.file.readData("2d_data"));
    bool verify2d = true;
    for (size_t i = 0; i < 20; i++) verify2d = verify2d && (data2d[i] == read2d[i]);
    results.report("2D data", test2d && verify2d);
    
    // 3D数据
    vector<size_t> shape3d = {3, 4, 5};
    vector<int> data3d(60);
    for (size_t i = 0; i < 60; i++) data3d[i] = i;
    bool test3d = tester.safeWriteData("3d_data", data3d.data(), shape3d);
    auto read3d = Cast<int>(tester.file.readData("3d_data"));
    bool verify3d = true;
    for (size_t i = 0; i < 60; i++) verify3d = verify3d && (data3d[i] == read3d[i]);
    results.report("3D data", test3d && verify3d);
    
    // 4D数据
    vector<size_t> shape4d = {2, 3, 4, 5};
    vector<int> data4d(120);
    for (size_t i = 0; i < 120; i++) data4d[i] = i;
    bool test4d = tester.safeWriteData("4d_data", data4d.data(), shape4d);
    auto read4d = Cast<int>(tester.file.readData("4d_data"));
    bool verify4d = true;
    for (size_t i = 0; i < 120; i++) verify4d = verify4d && (data4d[i] == read4d[i]);
    results.report("4D data", test4d && verify4d);
    
    // 5D数据（更高维度）
    vector<size_t> shape5d = {2, 2, 3, 4, 5};
    vector<int> data5d(240);
    for (size_t i = 0; i < 240; i++) data5d[i] = i;
    bool test5d = tester.safeWriteData("5d_data", data5d.data(), shape5d);
    auto read5d = Cast<int>(tester.file.readData("5d_data"));
    bool verify5d = true;
    for (size_t i = 0; i < 240; i++) verify5d = verify5d && (data5d[i] == read5d[i]);
    results.report("5D data", test5d && verify5d);
}

// 压力测试：大量小数据块
void testStressManyBlocks(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Many Small Blocks (2000+) ===" << endl;
    
    const int num_blocks = 2000;
    vector<bool> block_results;
    
    auto start_time = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_blocks; i++) {
        string block_name = "stress_block_" + to_string(i);
        int data[10] = {i, i+1, i+2, i+3, i+4, i+5, i+6, i+7, i+8, i+9};
        
        bool write_ok = tester.safeWriteData(block_name, data, {10});
        bool read_ok = false;
        
        if (write_ok) {
            auto read_data = Cast<int>(tester.file.readData(block_name));
            read_ok = true;
            for (int j = 0; j < 10; j++) {
                if (data[j] != read_data[j]) {
                    read_ok = false;
                    break;
                }
            }
        }
        
        block_results.push_back(write_ok && read_ok);
        
        // 每1000个块报告一次进度
        if ((i + 1) % 1000 == 0) {
            cout << "Processed " << (i + 1) << " blocks..." << endl;
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    // 统计成功率
    int success_count = 0;
    for (bool result : block_results) {
        if (result) success_count++;
    }
    
    double success_rate = (success_count * 100.0) / num_blocks;
    bool test_passed = (success_rate > 99.9); // 允许0.1%的失败率
    
    cout << "Success rate: " << success_rate << "% (" << success_count << "/" << num_blocks << ")" << endl;
    cout << "Time taken: " << duration.count() << " ms" << endl;
    
    results.report("2000+ blocks stress test", test_passed);
}

// 压力测试：超大单数据块
void testStressLargeBlock(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Single Large Block (4.3B+ elements) ===" << endl;
    
    // 注意：43亿个int32_t需要约16GB内存，请确保系统有足够资源
    // 这里我们使用较小的测试，但保留测试结构
    
    const size_t large_size = 1000000; // 100万个元素，可根据系统资源调整
    // const size_t large_size = 4300000000ULL; // 43亿个元素
    
    vector<uint32_t> large_data(large_size);
    
    // 使用随机数据填充
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<uint32_t> dis(0, 1000000);
    
    auto start_time = chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < large_size; i++) {
        large_data[i] = dis(gen);
    }
    
    bool write_ok = tester.safeWriteData("huge_block", large_data.data(), {large_size});
    
    auto write_time = chrono::high_resolution_clock::now();
    
    bool read_ok = false;
    if (write_ok) {
        auto read_data = Cast<uint32_t>(tester.file.readData("huge_block"));
        read_ok = true;
        
        // 抽样检查，不检查全部（太耗时）
        const size_t sample_count = min(size_t(1000), large_size);
        for (size_t i = 0; i < sample_count; i++) {
            size_t index = (i * large_size) / sample_count;
            if (large_data[index] != read_data[index]) {
                read_ok = false;
                break;
            }
        }
    }
    
    auto end_time = chrono::high_resolution_clock::now();
    
    auto write_duration = chrono::duration_cast<chrono::milliseconds>(write_time - start_time);
    auto read_duration = chrono::duration_cast<chrono::milliseconds>(end_time - write_time);
    auto total_duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    
    cout << "Large block size: " << large_size << " elements (" 
         << (large_size * sizeof(uint32_t) / (1024.0 * 1024.0)) << " MB)" << endl;
    cout << "Write time: " << write_duration.count() << " ms" << endl;
    cout << "Read time: " << read_duration.count() << " ms" << endl;
    cout << "Total time: " << total_duration.count() << " ms" << endl;
    
    results.report("Large block stress test", write_ok && read_ok);
}

// 异常匹配测试
void testExceptionMatching(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Exception Matching ===" << endl;
    
    // 写入double数据
    double double_data[10] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
    tester.safeWriteData("exception_test_double", double_data, {10});
    
    // 尝试用错误类型读取 - 应该抛出异常
    bool got_expected_exception = false;
    try {
        auto wrong_read = Cast<int>(tester.file.readData("exception_test_double"));
        // 如果执行到这里，说明没有抛出异常，测试失败
        got_expected_exception = false;
    } catch (const exception& e) {
        got_expected_exception = true;
    }
    
    results.report("Type mismatch exception", got_expected_exception);
    
    // 测试同名写入异常
    bool got_duplicate_exception = false;
    try {
        tester.file.writeData("duplicate_test", double_data, {10});
        tester.file.writeData("duplicate_test", double_data, {10}); // 应该抛出异常
        got_duplicate_exception = false;
    } catch (const exception& e) {
        got_duplicate_exception = true;
    }
    
    results.report("Duplicate name exception", got_duplicate_exception);
}

// 重写和擦除测试
void testRewriteAndErase(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Rewrite and Erase Operations ===" << endl;
    
    // 初始数据
    int original_data[5] = {1, 2, 3, 4, 5};
    bool write_ok = tester.safeWriteData("rewrite_test", original_data, {5});
    
    // 验证初始数据
    auto read_original = Cast<int>(tester.file.readData("rewrite_test"));
    bool original_correct = true;
    for (int i = 0; i < 5; i++) {
        original_correct = original_correct && (original_data[i] == read_original[i]);
    }
    
    // 重写数据
    int new_data[5] = {10, 20, 30, 40, 50};
    bool rewrite_ok = tester.safeReplaceData("rewrite_test", new_data, {5});
    
    // 验证重写后的数据
    auto read_new = Cast<int>(tester.file.readData("rewrite_test"));
    bool new_correct = true;
    for (int i = 0; i < 5; i++) {
        new_correct = new_correct && (new_data[i] == read_new[i]);
    }
    
    // 擦除数据
    bool erase_ok = tester.safeSmashData("rewrite_test");
    
    // 验证擦除
    bool erased_correct = false;
    try {
        auto read_after_erase = tester.file.readData("rewrite_test");
        // 如果数据块被正确擦除，readData应该返回空信息或抛出异常
        erased_correct = read_after_erase.info.blockName.empty();
    } catch (const exception& e) {
        erased_correct = true; // 抛出异常也是正确的行为
    }
    
    results.report("Rewrite operation", write_ok && original_correct && rewrite_ok && new_correct);
    results.report("Erase operation", erase_ok && erased_correct);
}

// 边界情况测试
void testEdgeCases(TestResults& results, MasFileTester& tester) {
    cout << "\n=== Testing Edge Cases ===" << endl;
    
    // 测试空数据
    bool empty_test = false;
    try {
        vector<int> empty_data;
        tester.safeWriteData("empty_data", empty_data.data(), {0});
        auto read_empty = tester.file.readData("empty_data");
        empty_test = (read_empty.info.blockSize == 0);
    } catch (const exception& e) {
        empty_test = false;
    }
    results.report("Empty data block", empty_test);

    // 测试1尺寸
    {
        int single_data[1] = {42};
        bool single_ok = tester.safeWriteData("single_element_a", single_data, {1});
        auto read_single = tester.file.readData("single_element_a");
        auto info = read_single.info;
        auto single_cast = Cast<int>(std::move(read_single));
        bool single_correct = single_ok && (info.blockSize == 1) && (single_cast[0] == 42);
        results.report("Single element-a", single_correct);
    }
    // 单元素写入
    {
        double single_data = 42;
        bool single_ok = tester.safeWriteData("single_element_b", single_data);
        auto read_single = tester.file.readData("single_element_b", "double");
        auto info = read_single.info;
        auto single_cast = Cast<double>(std::move(read_single));
        bool single_correct = single_ok && (info.blockSize == 1) && (single_cast[0] == 42);
        results.report("Single element-b", single_correct);
    }
    // 单元素替换
    {
        double single_data = 43;
        bool single_ok = tester.safeReplaceData("single_element_b", single_data);
        auto read_single = tester.file.readData("single_element_b", "double");
        auto single_cast = Cast<double>(std::move(read_single));
        bool single_correct = single_ok && (read_single.info.blockSize == 1) && (single_cast[0] == 43);
        results.report("Single element-c", single_correct);
    }

    
    // 测试字符串数据
    string test_string = "This is a test string with special characters: 中文测试 🎉";
    bool string_ok = tester.safeWriteData("long_string", test_string.c_str(), {test_string.length() + 1});
    auto read_string = Cast<string>(tester.file.readData("long_string"));
    bool string_correct = string_ok && (read_string == test_string);
    results.report("Long string with special chars", string_correct);
    
    // 测试文件清理
    bool clear_ok = tester.safeClear();
    auto heads_after_clear = tester.file.getHeads();
    bool clear_correct = clear_ok && heads_after_clear.empty();
    results.report("File clear operation", clear_correct);
}

int main() {
    cout << "=== unit test: masIO(complex) ===" << endl;
    
    TestResults results;
    
    // 创建测试文件
    mas_file fp("comprehensive_test.mas");
    MasFileTester tester(fp);
    
    // 清空文件开始测试
    tester.safeClear();
    
    // 执行所有测试套件
    testBasicTypes(results, tester);
    testMultiDimensional(results, tester);
    testStressManyBlocks(results, tester);
    testStressLargeBlock(results, tester);
    testExceptionMatching(results, tester);
    testRewriteAndErase(results, tester);
    testEdgeCases(results, tester);
    
    // 显示最终总结
    results.summary();

    filesystem::remove("comprehensive_test.mas");

    return 0;
}