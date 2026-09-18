#!/usr/bin/env python3
"""
Comprehensive and concise tests for MasFile Python interface.
"""

import sys
import os
import numpy as np
import pypsum

class MasFileTester:
    """Concise MasFile test suite."""
    
    def __init__(self, test_file):
        self.test_file = test_file
        self.passed = 0
        self.failed = 0
    
    def assert_eq(self, a, b, msg):
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if not np.array_equal(a, b):
                self.failed += 1
                print(f"❌ FAIL: {msg}")
                return
        elif isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            if not np.isclose(a, b):
                self.failed += 1
                print(f"❌ FAIL: {msg}")
                return
        else:
            if a != b:
                self.failed += 1
                print(f"❌ FAIL: {msg}")
                return
        self.passed += 1
        print(f"✅ PASS: {msg}")
    
    def test_type_roundtrip(self, name, data):
        """Generic roundtrip test for any data type."""
        mas = pypsum.MasFile(self.test_file)
        mas.write(name, data)
        read_data = mas.get_data(name)
        if isinstance(data, str):
            self.assert_eq(read_data, data, f"String roundtrip: {name}")
        elif isinstance(data, bool):
            self.assert_eq(read_data.dtype, np.bool_, f"Bool dtype: {name}")
            self.assert_eq(np.array_equal(read_data, data), True, f"Bool values: {name}")
        else:
            self.assert_eq(read_data.dtype, data.dtype, f"{name} dtype")
            self.assert_eq(read_data.shape, data.shape, f"{name} shape")
            self.assert_eq(read_data, data, f"{name} values")
    
    def test_array_types(self):
        """Test all supported array types."""
        print("\n=== Array Types ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        # All supported types
        test_cases = [
            ('bool', np.array([True, False], dtype=np.bool_)),
            ('int8', np.array([1, -128, 127], dtype=np.int8)),
            ('uint8', np.array([0, 128, 255], dtype=np.uint8)),
            ('int16', np.array([1, -32768, 32767], dtype=np.int16)),
            ('uint16', np.array([0, 32768, 65535], dtype=np.uint16)),
            ('int32', np.array([1, -2147483648, 2147483647], dtype=np.int32)),
            ('uint32', np.array([0, 2147483648, 4294967295], dtype=np.uint32)),
            ('int64', np.array([1, -2**63, 2**63-1], dtype=np.int64)),
            ('uint64', np.array([0, 2**63, 2**64-1], dtype=np.uint64)),
            ('float32', np.array([1.0, -3.4e38, 3.4e38], dtype=np.float32)),
            ('float64', np.array([1.0, -1.7e308, 1.7e308], dtype=np.float64)),
            ('str', 'test string'),
            ('empty', ''),
        ]
        
        for name, data in test_cases:
            self.test_type_roundtrip(name, data)
        
        # Check metadata
        names = mas.get_names()
        self.assert_eq(len(names), len(test_cases), "Item count")
        self.assert_eq(sorted(names), sorted([n for n, _ in test_cases]), "Names match")
        
        # Check info for each
        for info in mas.get_info():
            self.assert_eq('name' in info and 'type' in info and 'shape' in info, True, f"Info has fields")
    
    def test_replace_operations(self):
        """Test replace with all data types."""
        print("\n=== Replace Operations ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        # Write initial data
        test_cases = [
            ('bool', np.array([True], dtype=np.bool_)),
            ('int8', np.array([1], dtype=np.int8)),
            ('uint8', np.array([1], dtype=np.uint8)),
            ('int16', np.array([1], dtype=np.int16)),
            ('uint16', np.array([1], dtype=np.uint16)),
            ('int32', np.array([1], dtype=np.int32)),
            ('uint32', np.array([1], dtype=np.uint32)),
            ('int64', np.array([1], dtype=np.int64)),
            ('uint64', np.array([1], dtype=np.uint64)),
            ('float32', np.array([1.0], dtype=np.float32)),
            ('float64', np.array([1.0], dtype=np.float64)),
        ]
        
        for name, initial in test_cases:
            mas.write(name, initial)
            # Replace with new data
            if isinstance(initial, np.ndarray):
                new_data = initial * 2 if initial.dtype.kind != 'b' else ~initial
            else:
                new_data = ~initial
            mas.replace(name, new_data)
            # Verify
            read = mas.get_data(name)
            if initial.dtype.kind != 'b':
                self.assert_eq(np.array_equal(read, new_data), True, f"Replace {name}")
    
    def test_metadata_operations(self):
        """Test get_names and get_info."""
        print("\n=== Metadata Operations ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        # Write varied data
        mas.write('arr', np.array([1,2,3], dtype=np.float64))
        mas.write('scalar', 42)
        mas.write('string', 'test')
        
        # Test get_names
        names = mas.get_names()
        self.assert_eq(len(names), 3, "Names count")
        self.assert_eq(set(names), {'arr', 'scalar', 'string'}, "Names content")
        
        # Test get_info
        infos = mas.get_info()
        self.assert_eq(len(infos), 3, "Info count")
        arr_info = next((i for i in infos if i['name'] == 'arr'), None)
        self.assert_eq(arr_info['type'], 'double', "Array type")
        self.assert_eq(arr_info['shape'], [3], "Array shape")
    
    def test_access_methods(self):
        """Test get_data by name and index."""
        print("\n=== Access Methods ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        data1 = np.array([1,2,3], dtype=np.float64)
        data2 = np.array([4,5,6], dtype=np.float64)
        mas.write('item1', data1)
        mas.write('item2', data2)
        
        # Access by name
        read1 = mas.get_data('item1')
        self.assert_eq(np.array_equal(read1, data1), True, "Access by name")
        
        # Access by index
        read2 = mas.get_data(1)
        self.assert_eq(np.array_equal(read2, data2), True, "Access by index")
        
        # Invalid index
        invalid = mas.get_data(999)
        self.assert_eq(invalid, None, "Invalid index returns None")
        
        # Nonexistent name
        nonexistent = mas.get_data('nonexistent')
        self.assert_eq(nonexistent, None, "Nonexistent name returns None")
    
    def test_delete_operations(self):
        """Test delete by name and index."""
        print("\n=== Delete Operations ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        for i in range(5):
            mas.write(f'item{i}', i)
        
        # Delete by name
        result = mas.delete('item2')
        self.assert_eq(result, True, "Delete by name succeeds")
        self.assert_eq(len(mas.get_names()), 4, "Count after delete by name")
        
        # Delete by index
        result = mas.delete(0)
        self.assert_eq(result, True, "Delete by index succeeds")
        self.assert_eq(len(mas.get_names()), 3, "Count after delete by index")
        
        # Delete invalid
        result = mas.delete('nonexistent')
        self.assert_eq(result, False, "Delete nonexistent fails")
    
    def test_clear_and_multiple_shapes(self):
        """Test clear and various array shapes."""
        print("\n=== Clear and Shapes ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        # Write various shapes
        shapes = [(5,), (2,3), (2,2,2), (1,2,3,4)]
        for i, shape in enumerate(shapes):
            data = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
            mas.write(f'arr{i}', data)
        
        self.assert_eq(len(mas.get_names()), len(shapes), "Shape arrays count")
        
        # Clear and verify
        mas.clear()
        self.assert_eq(len(mas.get_names()), 0, "Count after clear")
        
        # Verify can write after clear
        mas.write('new', np.array([1], dtype=np.float64))
        self.assert_eq(len(mas.get_names()), 1, "Count after new write")
    
    def test_large_and_special_data(self):
        """Test large arrays and special characters."""
        print("\n=== Large and Special Data ===")
        mas = pypsum.MasFile(self.test_file)
        mas.clear()
        
        # Large arrays
        large_1d = np.arange(10000, dtype=np.float64)
        mas.write('large_1d', large_1d)
        self.assert_eq(np.array_equal(mas.get_data('large_1d'), large_1d), True, "Large 1D array")
        
        large_2d = np.arange(100*100, dtype=np.int32).reshape(100, 100)
        mas.write('large_2d', large_2d)
        self.assert_eq(np.array_equal(mas.get_data('large_2d'), large_2d), True, "Large 2D array")
        
        # Special strings
        special_strings = [
            "hello world",
            "中文测试",
            "日本語テスト",
            "emoji 😊",
            "a" * 1000,
            "",
        ]
        
        for i, s in enumerate(special_strings):
            name = f'str{i}'
            mas.write(name, s)
            self.assert_eq(mas.get_data(name), s, f"Special string {i}")
        
        # Edge values for integer types
        edge_cases = [
            ('int8_min', np.array([-128], dtype=np.int8)),
            ('int8_max', np.array([127], dtype=np.int8)),
            ('uint8_max', np.array([255], dtype=np.uint8)),
            ('int16_min', np.array([-32768], dtype=np.int16)),
            ('int16_max', np.array([32767], dtype=np.int16)),
            ('uint16_max', np.array([65535], dtype=np.uint16)),
        ]
        
        for name, data in edge_cases:
            mas.write(name, data)
            self.assert_eq(np.array_equal(mas.get_data(name), data), True, f"Edge {name}")
    
    def run_all(self):
        """Run all tests."""
        print("=" * 50)
        print("MasFile Concise Test Suite")
        print("=" * 50)
        
        self.test_array_types()
        self.test_replace_operations()
        self.test_metadata_operations()
        self.test_access_methods()
        self.test_delete_operations()
        self.test_clear_and_multiple_shapes()
        self.test_large_and_special_data()
        
        print("\n" + "=" * 50)
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print("=" * 50)
        
        return self.failed == 0

if __name__ == "__main__":
    test_file = os.path.join(os.path.dirname(__file__), "test_concise.mas")
    tester = MasFileTester(test_file)
    success = tester.run_all()
    sys.exit(0 if success else 1)
