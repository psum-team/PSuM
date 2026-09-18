#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <iostream>
#include "src/serialization/mas.hpp"
#include <filesystem>
#include <variant>
#include <unordered_set>
#include <type_traits>

namespace py = pybind11;
using namespace psum::serialization;

class MasFileInterface {
public:
    MasFileInterface(const std::string& path) : _fp(path), _path(path) {}

    std::vector<std::string> get_item_names() {
        std::vector<std::string> names;
        auto& heads = _fp.readHead();
        for (const auto& h : heads) {
            names.push_back(h.info.blockName);
        }
        return names;
    }

    py::list get_items_info() {
        py::list info_list;
        auto& heads = _fp.readHead();
        for (const auto& h : heads) {
            py::dict d;
            d["name"] = h.info.blockName;
            d["type"] = h.info.blockType;
            d["shape"] = h.info.blockShape;
            info_list.append(d);
        }
        return info_list;
    }

    void clear_file() {
        _fp.clear();
    }

    bool delete_by_name(const std::string& name) {
        return _fp.smashData(name);
    }

    bool delete_by_idx(int idx) {
        auto& heads = _fp.readHead();
        if (idx >= 0 && idx < (int)heads.size()) {
            return _fp.smashData(heads[idx].info.blockName);
        }
        return false;
    }

    py::object get_data_by_name(const std::string& name) {
        auto data = _fp.readData(name);
        
        // Check if data exists (blockName is empty for nonexistent data)
        if (data.info.blockName.empty()) {
            return py::none();
        }
        
        return std::visit([&data](auto& v) -> py::object {
            using T = typename std::decay_t<decltype(v)>::element_type;
            if constexpr (std::is_same_v<T, char>) {
                return py::str(v.get(), data.info.blockSize);
            } else {
                std::vector<size_t> shape = data.info.blockShape;
                if (shape.empty()) shape.push_back(data.info.blockSize / sizeof(T));
                
                return py::array_t<T>(shape, v.get());
            }
        }, data.content);
    }

    py::object get_data_by_idx(int idx) {
        auto& heads = _fp.readHead();
        if (idx >= 0 && idx < (int)heads.size()) {
            return get_data_by_name(heads[idx].info.blockName);
        }
        return py::none();
    }

    void write_array(const std::string& name, py::array data) {
        auto buf = data.request();
        std::vector<size_t> shape;
        for (int i = 0; i < buf.ndim; ++i) {
            shape.push_back(buf.shape[i]);
        }

        if (buf.ndim == 0) {
            throw std::runtime_error("Cannot write 0-dimensional array. Use write(name, scalar) for scalars.");
        }

        auto dtype = data.dtype();
        auto kind = dtype.kind();
        auto dtype_itemsize = py::int_(dtype.attr("itemsize"));
        int itemsize = dtype_itemsize.cast<int>();
 
        // numpy dtype: https://numpy.org/doc/stable/reference/arrays.dtypes.html
        // Supported types: bool, int8/16/32/64, uint8/16/32/64, float32/64
        // Not supported: complex (kind='c'), float16 (kind='f', itemsize=2)
        // Note: char handled by write_string() or as int8/int16/int32/int64 (kind='i')
        // So kind='c' in write_array() always means complex
        
        if (kind == 'c') {
            throw std::runtime_error("Unsupported complex type: " + std::string(py::str(dtype)));
        }

        if (kind == 'f') {
            if (itemsize == 8) {  // float64
                _fp.writeData(name, static_cast<const double*>(buf.ptr), shape);
            } else if (itemsize == 4) {  // float32
                _fp.writeData(name, static_cast<const float*>(buf.ptr), shape);
            } else {
                throw std::runtime_error("Unsupported float type: " + std::string(py::str(dtype)));
            }
        } else if (kind == 'i') {
            if (itemsize == 8) {  // int64
                _fp.writeData(name, static_cast<const int64_t*>(buf.ptr), shape);
            } else if (itemsize == 4) {  // int32
                _fp.writeData(name, static_cast<const int32_t*>(buf.ptr), shape);
            } else if (itemsize == 2) {  // int16
                _fp.writeData(name, static_cast<const int16_t*>(buf.ptr), shape);
            } else if (itemsize == 1) {  // int8
                _fp.writeData(name, static_cast<const int8_t*>(buf.ptr), shape);
            } else {
                throw std::runtime_error("Unsupported integer type: " + std::string(py::str(dtype)));
            }
        } else if (kind == 'u') {
            if (itemsize == 8) {  // uint64
                _fp.writeData(name, static_cast<const uint64_t*>(buf.ptr), shape);
            } else if (itemsize == 4) {  // uint32
                _fp.writeData(name, static_cast<const uint32_t*>(buf.ptr), shape);
            } else if (itemsize == 2) {  // uint16
                _fp.writeData(name, static_cast<const uint16_t*>(buf.ptr), shape);
            } else if (itemsize == 1) {  // uint8
                _fp.writeData(name, static_cast<const uint8_t*>(buf.ptr), shape);
            } else {
                throw std::runtime_error("Unsupported unsigned integer type: " + std::string(py::str(dtype)));
            }
        } else if (kind == 'b') {
            if (itemsize == 1) {  // bool
                _fp.writeData(name, static_cast<const bool*>(buf.ptr), shape);
            } else {
                throw std::runtime_error("Unsupported bool type: " + std::string(py::str(dtype)));
            }
        } else if (kind == 'c') {
            _fp.writeData(name, static_cast<const char*>(buf.ptr), shape);
        } else {
            throw std::runtime_error("Unsupported data type: " + std::string(py::str(dtype)));
        }
    }

    void write_string(const std::string& name, const std::string& data) {
        _fp.writeData(name, data);
    }

    void write_double(const std::string& name, double value) {
        _fp.writeData(name, value);
    }

    void write_float(const std::string& name, float value) {
        _fp.writeData(name, value);
    }

    void write_int32(const std::string& name, int32_t value) {
        _fp.writeData(name, value);
    }

    void write_int64(const std::string& name, int64_t value) {
        _fp.writeData(name, value);
    }

    void write_uint32(const std::string& name, uint32_t value) {
        _fp.writeData(name, value);
    }

    void write_uint64(const std::string& name, uint64_t value) {
        _fp.writeData(name, value);
    }

    void write_int16(const std::string& name, int16_t value) {
        _fp.writeData(name, value);
    }

    void write_int8(const std::string& name, int8_t value) {
        _fp.writeData(name, value);
    }

    void write_uint16(const std::string& name, uint16_t value) {
        _fp.writeData(name, value);
    }

    void write_uint8(const std::string& name, uint8_t value) {
        _fp.writeData(name, value);
    }

    void write_bool(const std::string& name, bool value) {
        _fp.writeData(name, value);
    }

    void replace_array(const std::string& name, py::array data) {
        _fp.smashData(name);
        write_array(name, data);
    }

    void replace_string(const std::string& name, const std::string& data) {
        _fp.smashData(name);
        write_string(name, data);
    }
    
    void replace_double(const std::string& name, double value) {
        _fp.smashData(name);
        write_double(name, value);
    }
    
    void replace_float(const std::string& name, float value) {
        _fp.smashData(name);
        write_float(name, value);
    }
    
    void replace_int32(const std::string& name, int32_t value) {
        _fp.smashData(name);
        write_int32(name, value);
    }
    
    void replace_int64(const std::string& name, int64_t value) {
        _fp.smashData(name);
        write_int64(name, value);
    }
    
    void replace_uint32(const std::string& name, uint32_t value) {
        _fp.smashData(name);
        write_uint32(name, value);
    }
    
    void replace_uint64(const std::string& name, uint64_t value) {
        _fp.smashData(name);
        write_uint64(name, value);
    }

    void replace_int16(const std::string& name, int16_t value) {
        _fp.smashData(name);
        write_int16(name, value);
    }

    void replace_int8(const std::string& name, int8_t value) {
        _fp.smashData(name);
        write_int8(name, value);
    }

    void replace_uint16(const std::string& name, uint16_t value) {
        _fp.smashData(name);
        write_uint16(name, value);
    }

    void replace_uint8(const std::string& name, uint8_t value) {
        _fp.smashData(name);
        write_uint8(name, value);
    }

    void replace_bool(const std::string& name, bool value) {
        _fp.smashData(name);
        write_bool(name, value);
    }

private:
    mas_file _fp;
    std::string _path;
};

PYBIND11_MODULE(mas_file_py, m) {
    m.doc() = "Python interface for MasFile using pybind11";

    py::class_<MasFileInterface>(m, "MasFile")
        .def(py::init<const std::string&>())
        .def("get_names", &MasFileInterface::get_item_names, "Get list of all block names")
        .def("get_info", &MasFileInterface::get_items_info, "Get detailed metadata for all blocks")
        .def("clear", &MasFileInterface::clear_file, "Clear all data in file")
        .def("delete", &MasFileInterface::delete_by_name, "Delete item by name")
        .def("delete", &MasFileInterface::delete_by_idx, "Delete item by index")
        .def("get_data", &MasFileInterface::get_data_by_name, "Get item data as a NumPy array")
        .def("get_data", &MasFileInterface::get_data_by_idx, "Get item data as a NumPy array")
        .def("write", &MasFileInterface::write_array, "Write NumPy array data")
        .def("write", &MasFileInterface::write_string, "Write string data")
        .def("write", &MasFileInterface::write_double, "Write double scalar")
        .def("write", &MasFileInterface::write_float, "Write float scalar")
        .def("write", &MasFileInterface::write_int32, "Write int32 scalar")
        .def("write", &MasFileInterface::write_int64, "Write int64 scalar")
        .def("write", &MasFileInterface::write_uint32, "Write uint32 scalar")
        .def("write", &MasFileInterface::write_uint64, "Write uint64 scalar")
        .def("write", &MasFileInterface::write_int16, "Write int16 scalar")
        .def("write", &MasFileInterface::write_int8, "Write int8 scalar")
        .def("write", &MasFileInterface::write_uint16, "Write uint16 scalar")
        .def("write", &MasFileInterface::write_uint8, "Write uint8 scalar")
        .def("write", &MasFileInterface::write_bool, "Write bool scalar")
        .def("replace", &MasFileInterface::replace_array, "Replace NumPy array data")
        .def("replace", &MasFileInterface::replace_string, "Replace string data")
        .def("replace", &MasFileInterface::replace_double, "Replace double scalar")
        .def("replace", &MasFileInterface::replace_float, "Replace float scalar")
        .def("replace", &MasFileInterface::replace_int32, "Replace int32 scalar")
        .def("replace", &MasFileInterface::replace_int64, "Replace int64 scalar")
        .def("replace", &MasFileInterface::replace_uint32, "Replace uint32 scalar")
        .def("replace", &MasFileInterface::replace_uint64, "Replace uint64 scalar")
        .def("replace", &MasFileInterface::replace_int16, "Replace int16 scalar")
        .def("replace", &MasFileInterface::replace_int8, "Replace int8 scalar")
        .def("replace", &MasFileInterface::replace_uint16, "Replace uint16 scalar")
        .def("replace", &MasFileInterface::replace_uint8, "Replace uint8 scalar")
        .def("replace", &MasFileInterface::replace_bool, "Replace bool scalar");
}
