#ifndef PSUM_SERIALIZATION_CONTAINER_SL_HPP
#define PSUM_SERIALIZATION_CONTAINER_SL_HPP

#include "mas.hpp"
#include "../tag/tagged_struct.hpp"

namespace psum {

namespace serialization {

    // S/L for vector<fundamental> (except for vector<bool>)
    template <foundation::mas_fundamental T>
    void save(mas_file& fp, const std::string& name, const std::vector<T>& obj) {
        fp.writeData(name, obj.data(), {obj.size(), 1});
    }

    template <foundation::mas_fundamental T>
    void load(mas_file& fp, const std::string& name, std::vector<T>& obj) {
        auto res = fp.readData(name);
        bool shape_check = false;
        if (res.info.blockShape.size() == 2)
            if (res.info.blockShape[1] == 1)
                shape_check = true;
        if (shape_check) {
            size_t size = res.info.blockShape[0];
            auto data = Cast<T>(std::move(res));
            obj.assign(data.get(), data.get() + size);
        }
        else {
            throw std::runtime_error("Error: block '" + res.info.blockName + "' cannot be loaded as vector.");
        }
    }

    // S/L for vector<bool>
    void save(mas_file& fp, const std::string& name, const std::vector<bool>& obj) {
        std::unique_ptr<bool[]> raw(new bool[obj.size()]);
        for (size_t i = 0; i < obj.size(); ++i)
            raw[i] = obj[i];
        fp.writeData(name, raw.get(), {obj.size(), 1});
    }

    void load(mas_file& fp, const std::string& name, std::vector<bool>& obj) {
        auto res = fp.readData(name);
        bool shape_check = false;
        if (res.info.blockShape.size() == 2)
            if (res.info.blockShape[1] == 1)
                shape_check = true;
        if (shape_check) {
            size_t size = res.info.blockShape[0];
            auto data = Cast<bool>(std::move(res));
            obj.assign(data.get(), data.get() + size);
        }
        else {
            throw std::runtime_error("Error: block '" + res.info.blockName + "' cannot be loaded as vector<bool>.");
        }
    }    

    template <typename T>
    void save_vector_fool(mas_file& fp, const std::string& name, const std::vector<T>& data) {
        save(fp, name + ".size", data.size());
        for (size_t i = 0; i < data.size(); i++) {
            save(fp, name + '[' + std::to_string(i) + ']', data[i]);
        }
    }
    
    template <typename T>
    void load_vector_fool(mas_file& fp, const std::string& name, std::vector<T>& data) {
        size_t size;
        load(fp, name + ".size", size);
        data.resize(size);
        for (size_t i = 0; i < data.size(); i++) {
            load(fp, name + '[' + std::to_string(i) + ']', data[i]);
        }
    }

    // T is not mas_fundamental or matrix or tagged_struct
    template <typename T>
    concept fallback_element = 
        !foundation::mas_fundamental<T> &&
        !std::is_base_of_v<Eigen::MatrixBase<std::remove_cvref_t<T>>, std::remove_cvref_t<T>> &&
        !tag::is_tagged_struct<T>::value;

    // fallback for vector<...>
    template <fallback_element T>
    void save(mas_file& fp, const std::string& name, const std::vector<T>& data) {
        save_vector_fool(fp, name, data);
    }
    
    template <fallback_element T>
    void load(mas_file& fp, const std::string& name, std::vector<T>& data) {
        load_vector_fool(fp, name, data);
    }

    // S/L for vector<Eigen::Matrix>
    template <typename ScalarType, int Rows, int Cols, int Opt>
	inline void save(mas_file& fp, const std::string& name, const std::vector<Eigen::Matrix<ScalarType, Rows, Cols, Opt>>& data) {
        if constexpr(Rows == -1 || Cols == -1) {
            save_vector_fool(fp, name, data);
        }
        else {
		    std::vector<size_t> shape = {data.size(), Rows, Cols};
            std::vector<ScalarType> buffer(data.size() * Rows * Cols);
            for (size_t i = 0; i < data.size(); i++)
                for (size_t j = 0; j < Rows; j++)
                    for (size_t k = 0; k < Cols; k++)
                        buffer[i * Rows * Cols + j * Cols + k] = data[i](j, k);
            fp.writeData(name, buffer.data(), shape);
        }
    }
    
    template <typename ScalarType, int Rows, int Cols, int Opt>
	inline void load(mas_file& fp, const std::string& name, std::vector<Eigen::Matrix<ScalarType, Rows, Cols, Opt>>& data) {
        if constexpr(Rows == -1 || Cols == -1) {
            load_vector_fool(fp, name, data);
        } else {
            auto res = fp.readData(name);
            bool shape_check = false;
            if (res.info.blockShape.size() == 3)
                if (res.info.blockShape[1] == Rows && res.info.blockShape[2] == Cols)
                    shape_check = true;
            if (shape_check) {
                size_t size = res.info.blockShape[0];
                data.resize(size);
                auto buffer = Cast<ScalarType>(std::move(res));
                for (size_t i = 0; i < data.size(); i++)
                    for (size_t j = 0; j < Rows; j++)
                        for (size_t k = 0; k < Cols; k++)
                            data[i](j, k) = buffer[i * Rows * Cols + j * Cols + k];
            }
            else {
                throw std::runtime_error("Error: block '" + res.info.blockName + "' cannot be loaded as vector<Eigen::Matrix>.");
            }
        }
    }

    // S/L for map
    template<typename T>
    concept Maplike = requires(T container) {
        typename T::key_type;
        typename T::mapped_type;
    };

    template<Maplike MapType>
    void save(mas_file& fp, const std::string& name, const MapType& obj) {
        // MapType = map / unordered_map
        std::vector<typename MapType::key_type> K_vec;
        std::vector<typename MapType::mapped_type> V_vec;
        for (const auto& pair : obj) {
            K_vec.push_back(pair.first);
            V_vec.push_back(pair.second);
        }
        save(fp, name + ".keys", K_vec);
        save(fp, name + ".values", V_vec);
    }

    template<Maplike MapType>
    void load(mas_file& fp, const std::string& name, MapType& obj) {
        std::vector<typename MapType::key_type> K_vec;
        std::vector<typename MapType::mapped_type> V_vec;
        load(fp, name + ".keys", K_vec);
        load(fp, name + ".values", V_vec);
        if (K_vec.size() != V_vec.size())
            throw std::runtime_error("Error: K_vec.size() != V_vec.size() so '" + name + "' cannot be loaded.");
        obj.clear();
        for (size_t i = 0; i < K_vec.size(); i++) {
            obj[K_vec[i]] = V_vec[i];
        }
    }
}

}

#endif