#ifndef PSUM_SERIALIZATION_FOUNDATION_HPP
#define PSUM_SERIALIZATION_FOUNDATION_HPP

#include <string>
#include <stdexcept>
#include <variant>
#include <functional>
#include <memory>
#include <map>
#include <cstdint>
#include <iostream>

namespace psum {

namespace serialization {

namespace foundation {

    // all available types
    template <typename Type>
    concept mas_fundamental =
        std::is_same_v<std::remove_cvref_t<Type>, double> ||
        std::is_same_v<std::remove_cvref_t<Type>, float> ||
        std::is_same_v<std::remove_cvref_t<Type>, int64_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, uint64_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, int32_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, uint32_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, int16_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, uint16_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, int8_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, uint8_t> ||
        std::is_same_v<std::remove_cvref_t<Type>, bool> ||
        std::is_same_v<std::remove_cvref_t<Type>, char>;

    #define PSUM_UNIQUE_PTR_TYPE(x) std::unique_ptr<x[]>
	typedef std::variant<
        PSUM_UNIQUE_PTR_TYPE(double),
        PSUM_UNIQUE_PTR_TYPE(float),
        PSUM_UNIQUE_PTR_TYPE(int64_t),
        PSUM_UNIQUE_PTR_TYPE(uint64_t),
        PSUM_UNIQUE_PTR_TYPE(int32_t),
        PSUM_UNIQUE_PTR_TYPE(uint32_t),
        PSUM_UNIQUE_PTR_TYPE(int16_t),
        PSUM_UNIQUE_PTR_TYPE(uint16_t),
        PSUM_UNIQUE_PTR_TYPE(int8_t),
        PSUM_UNIQUE_PTR_TYPE(uint8_t),
        PSUM_UNIQUE_PTR_TYPE(bool),
        PSUM_UNIQUE_PTR_TYPE(char)
    > vPtr;

    // simple reflection mechanism
    #define PSUM_MATCH_TYPE_MEMSET(x) {std::string(#x), [](size_t size){return vPtr( std::make_unique<x[]>(size) );}}

    const std::map<std::string, std::function<vPtr(size_t)>>& vPtr_typestr2_setmem() {
        static const std::map<std::string, std::function<vPtr(size_t)>> _vPtr_typestr2_setmem{
            PSUM_MATCH_TYPE_MEMSET(double),
            PSUM_MATCH_TYPE_MEMSET(float),
            PSUM_MATCH_TYPE_MEMSET(int64_t),
            PSUM_MATCH_TYPE_MEMSET(uint64_t),
            PSUM_MATCH_TYPE_MEMSET(int32_t),
            PSUM_MATCH_TYPE_MEMSET(uint32_t),
            PSUM_MATCH_TYPE_MEMSET(int16_t),
            PSUM_MATCH_TYPE_MEMSET(uint16_t),
            PSUM_MATCH_TYPE_MEMSET(int8_t),
            PSUM_MATCH_TYPE_MEMSET(uint8_t),
            PSUM_MATCH_TYPE_MEMSET(bool),
            PSUM_MATCH_TYPE_MEMSET(char)
        };
        return _vPtr_typestr2_setmem;
    };

    #define PSUM_MATCH_TYPE_TYPESTR(x) if constexpr (std::is_same_v<x, std::remove_cvref_t<T>>) { return #x; }

    template<mas_fundamental T>
    std::string basic_type_name() {
        PSUM_MATCH_TYPE_TYPESTR(double)
        PSUM_MATCH_TYPE_TYPESTR(float)
        PSUM_MATCH_TYPE_TYPESTR(int64_t)
        PSUM_MATCH_TYPE_TYPESTR(uint64_t)
        PSUM_MATCH_TYPE_TYPESTR(int32_t)
        PSUM_MATCH_TYPE_TYPESTR(uint32_t)
        PSUM_MATCH_TYPE_TYPESTR(int16_t)
        PSUM_MATCH_TYPE_TYPESTR(uint16_t)
        PSUM_MATCH_TYPE_TYPESTR(int8_t)
        PSUM_MATCH_TYPE_TYPESTR(uint8_t)
        PSUM_MATCH_TYPE_TYPESTR(bool)
        PSUM_MATCH_TYPE_TYPESTR(char)
        throw std::runtime_error("Error in basic_type_name: Unsupported type, typeid(T).name() = " + std::string(typeid(T).name()));
    }

    const std::map<std::string, std::string>& synonym_matlab2cpp_map() {
        static const std::map<std::string, std::string> _synonym_matlab2cpp_map {
            {"single","float"},{"logical","bool"} ,
            {"int64","int64_t"},{"uint64","uint64_t"} ,
            {"int32","int32_t"},{"uint32","uint32_t"} ,
            {"int16","int16_t"},{"uint16","uint16_t"} ,
            {"int8","int8_t"}, {"uint8","uint8_t"} ,
            {"char","char"}, {"double", "double"}
        };
        return _synonym_matlab2cpp_map;
    };

    const std::map<std::string, std::string>& synonym_cpp2matlab_map() {
        static const std::map<std::string, std::string> _synonym_cpp2matlab_map {
            {"float","single"},{"bool","logical"} ,
            {"int64_t","int64"},{"uint64_t","uint64"} ,
            {"int32_t","int32"},{"uint32_t","uint32"} ,
            {"int16_t","int16"},{"uint16_t","uint16"} ,
            {"int8_t","int8"}, {"uint8_t","uint8"} ,
            {"char","char"}, {"double", "double"}
        };
        return _synonym_cpp2matlab_map;
    };
}

}

}

#endif
