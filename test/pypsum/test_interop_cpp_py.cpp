#include <psum/serialization.hpp>
#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace psum::serialization;

int main() {
    cout << "==========================================================\n";
    cout << "C++ <-> Python Interaction Demo\n";
    cout << "==========================================================\n";
    
    const string test_file = "test/pypsum/interop_test.mas";
    
    // Clear any existing file first
    {
        mas_file fp(test_file, mas_file::replaceMode);
        cout << "Created new test file\n";
    }
    
    // Write initial data
    cout << "\n[C++] Step 1: Writing initial data\n";
    {
        mas_file fp(test_file);
        
        double arr_data[] = {1, 2, 3, 4, 5, 6};
        fp.writeData("cpp_array", arr_data, {2, 3});
        fp.writeData("cpp_string", "Hello from C++");
        fp.writeData("cpp_scalar", 3.14159);
        
        cout << "  Wrote cpp_array: [[1,2,3],[4,5,6]]\n";
        cout << "  Wrote cpp_string: 'Hello from C++'\n";
        cout << "  Wrote cpp_scalar: 3.14159\n";
    }
    
    // Invoke Python to read and write
    cout << "\n[C++] Step 2: Invoking Python...\n";
    const string python_cmd = "python3 -c \""
        R"PYTH(
import sys
import numpy as np

test_file = 'test/pypsum/interop_test.mas'
import pypsum

file = pypsum.MasFile(test_file)

print('[Python] Started...')

# Read and verify
cpp_array = file.get_data('cpp_array')
cpp_string = file.get_data('cpp_string')
cpp_scalar = file.get_data('cpp_scalar')

if cpp_array is None or cpp_string is None or cpp_scalar is None:
    print('[Python] ERROR: Could not read C++ data')
    sys.exit(1)

print(f'[Python] Read array: {cpp_array}')
print(f'[Python] Read string: {cpp_string}')
print(f'[Python] Read scalar: {type(cpp_scalar)}')

# Extract scalar value
if hasattr(cpp_scalar, 'shape') and cpp_scalar.shape == (1, 1):
    scalar_val = float(cpp_scalar[0, 0])
else:
    scalar_val = float(cpp_scalar)

print(f'[Python] Read scalar value: {scalar_val}')

# Modify and write back
print('[Python] Modifying and writing new data...')

new_array = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float64)
new_string = 'Data written by Python'
new_scalar = 99.9

file.write('py_array', new_array)
file.write('py_string', new_string)
file.write('py_scalar', new_scalar)

print(f'[Python] Wrote array: shape={new_array.shape}')
print(f'[Python] Wrote string: {new_string}')
print(f'[Python] Wrote scalar: {new_scalar}')

print('[Python] Finished')
)PYTH"
        "\"";
    
    int ret = system(python_cmd.c_str());
    if (ret != 0) {
        cout << "Error: Python script failed with code " << ret << "\n";
        return 1;
    }
    
    // C++ reads Python's data
    cout << "\n[C++] Step 3: Reading data written by Python\n";
    {
        mas_file fp(test_file);

        cout << "  py_array: ";
        auto arr_info = fp.readData("py_array").info;
        auto arr_content = Cast<double>(fp.readData("py_array"));
        for (int i = 0; i < arr_info.blockSize; i++) {
            cout << arr_content[i] << " ";
        }
        cout << "\n";

        auto str = Cast<std::string>(fp.readData("py_string"));
        auto scalar = Cast<double>(fp.readData("py_scalar"));
        cout << "  py_string: '" << str << "'\n";
        cout << "  py_scalar: " << scalar[0] << "\n";
        
        double scalar_val = scalar[0];
        
        bool pass = (arr_info.blockShape.size() == 2 && arr_info.blockShape[0] == 2 && arr_info.blockShape[1]);
        pass = pass && (str == "Data written by Python");
        pass = pass && (abs(scalar_val - 99.9) < 0.001);
        
        if (pass) {
            cout << "  ✅ Python -> C++ roundtrip: PASSED\n";
        } else {
            cout << "  ❌ Python -> C++ roundtrip: FAILED\n";
            return 1;
        }
    }
    
    cout << "\n==========================================================\n";
    cout << "C++ <-> Python Interaction: ALL TESTS PASSED ✅\n";
    cout << "==========================================================\n";
    
    return 0;
}
