#ifndef PSUM_SERIALIZATION_JSON_LOADER_HPP
#define PSUM_SERIALIZATION_JSON_LOADER_HPP

#include <set>
#include "object_manager.hpp"
#include "../../third_party/json.hpp"

namespace psum {

namespace serialization {

    struct json_loader: public object_manager {
    
    private:
        using json = nlohmann::json;
        std::vector<std::string> all_keys;
        bool convert_scientific_notation_string = false;

    public:

        json_loader(std::filesystem::path path = "") : object_manager(path) {};

        void set_convert_scientific_notation_string(bool flag) {
            convert_scientific_notation_string = flag;
        }

        inline bool load_json(std::string json_fname)
        {
            json data = flatten_json(json_fname);
            bool ans = true;

            auto is_scientific_notation = [](const std::string &str) {
                return std::regex_search(str, std::regex("^[-+]?[0-9]*\\.?[0-9]+[eE][-+]?[0-9]+$"));
            };

            for (auto &it : data.items())
            {
                if (it.value().is_boolean()) {
                    obj<bool>(it.key()) = it.value();
                    all_keys.push_back(it.key());
                }
                else if (it.value().is_number_float()) {
                    obj<double>(it.key()) = it.value();
                    all_keys.push_back(it.key());
                }
                else if (it.value().is_number_integer()) {
                    obj<int>(it.key()) = it.value();
                    all_keys.push_back(it.key());
                }
                else if (it.value().is_string()) {
                    if (is_scientific_notation(it.value()) && convert_scientific_notation_string) {
                        obj<double>(it.key()) = std::stod(std::string(it.value()));
                        all_keys.push_back(it.key());
                    }
                    else {
                        obj<std::string>(it.key()) = it.value();
                        all_keys.push_back(it.key());
                    }
                }
                else if (it.value().is_array())
                {
                    if (it.value().begin()->is_boolean())
                    {
                        std::vector<bool> buf;
                        for (auto &ii : it.value().items())
                            if (ii.value().is_boolean())
                                buf.push_back(ii.value());
                            else{
                                std::cerr << "Type error in load_json: view '" << it.key() << "' as vector<bool>" << std::endl;
                                ans = false;
                            }
                        obj<std::vector<bool>>(it.key()) = buf;
                        all_keys.push_back(it.key());
                    }
                    else if (it.value().begin()->is_number_float())
                    {
                        std::vector<double> buf;
                        for (auto &ii : it.value().items())
                            if (ii.value().is_number_float())
                                buf.push_back(ii.value());
                            else{
                                std::cerr << "Type error in load_json: view '" << it.key() << "' as vector<double>" << std::endl;
                                ans = false;
                            }
                        obj<std::vector<double>>(it.key()) = buf;
                        all_keys.push_back(it.key());
                    }
                    else if (it.value().begin()->is_number_integer())
                    {
                        std::vector<int> buf;
                        for (auto &ii : it.value().items())
                            if (ii.value().is_number_integer())
                                buf.push_back(ii.value());
                            else{
                                std::cerr << "Type error in load_json: view '" << it.key() << "' as vector<int>" << std::endl;
                                ans = false;
                            }
                        obj<std::vector<int>>(it.key()) = buf;
                        all_keys.push_back(it.key());
                    }
                    else
                    {
                        std::cerr << "Type error in load_json: view '" << it.key() << "' cannot be loaded, because only boolean, number and integer arrays are supported." << std::endl;
                    }
                }
                else
                {
                    std::cerr << "Type error in load_json: item '" << it.key() << "' cannot be loaded." << std::endl;
                    ans = false;
                }
            }

            return ans;
        }


        inline void try_insert(const json& data, const std::string& parent_key, json& result, std::set<std::string>& keys) {
            if (keys.find(parent_key) != keys.end())
                throw std::runtime_error("Duplicate key detected: " + parent_key);
            keys.insert(parent_key);
            result[parent_key] = data;
        }

        inline void flatten(const json& data, const std::string& parent_key, json& result, std::set<std::string>& keys) {
            if (data.is_object()) {
                // JSON object: walk key/value pairs and flatten nested keys recursively
                for (auto& [key, value] : data.items()) {
                    std::string new_key = parent_key.empty() ? key : parent_key + "." + key;
                    flatten(value, new_key, result, keys);
                }
            }
            // Other cases: basic types (number/string/bool) and arrays
            else {
                if (data.is_array())
                {
                    bool acceptable = false;
                    if (data.size() > 0) {
                        // code for is_bool| is_float | is_int
                        uint8_t type_code = (data.at(0).is_boolean() << 0) | (data.at(0).is_number_float() << 1) | (data.at(0).is_number_integer() << 2);
                        if (type_code != 0) acceptable = true;
                        for (auto &element : data)
                        if (type_code != ((element.is_boolean() << 0) | (element.is_number_float() << 1) | (element.is_number_integer() << 2))) {
                            acceptable = false;
                            break;
                        }
                    }
                    if (acceptable)
                        try_insert(data, parent_key, result, keys);
                    else
                    {
                        size_t index = 0;
                        for (auto& element : data) {
                            std::string new_key = parent_key + "(" + std::to_string(index) + ")";
                            flatten(element, new_key, result, keys);
                            ++index;
                        }
                    }
                }
                else {
                    bool acceptable = data.is_boolean() || data.is_number_float() || data.is_number_integer() || data.is_string();
                    if (acceptable)
                        try_insert(data, parent_key, result, keys);
                    else
                        throw std::runtime_error("Invalid data type detected: " + parent_key);
                }
            }
        }

        inline json flatten_json(std::string json_fname) {
            // Read a JSON file
            std::ifstream file(json_fname);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open file: " + json_fname);
            }

            json data;
            try {
                file >> data; // parse JSON content
            } catch (const json::parse_error& e) {
                throw std::runtime_error("JSON parse error: " + std::string(e.what()));
            }
        
            json result;
            std::set<std::string> keys; // existing keys, used to detect conflicts
            flatten(data, "", result, keys); // recursive flatten starting from the root key
            return result;
        }

        struct key_condition {
            json_loader &para_table;
            std::string prefix;
            key_condition(json_loader &pt) : 
                para_table(pt) {};
        
            template <typename Type>
            decltype(auto) obj(const std::string& name)
            {
                if (para_table.contains<Type>(prefix + "." + name))
                    return para_table.obj<Type>(prefix + "." + name);
                else
                    throw std::runtime_error("invalid key in key_condition::obj. name=" + name);
            }

            template <typename Type>
            decltype(auto) obj()
            {
                if (para_table.contains<Type>(prefix))
                    return para_table.obj<Type>(prefix);
                else
                    throw std::runtime_error("invalid key in key_condition::obj. name=" + prefix);
            }
        
            template <typename Type>
            void set(const std::string& name, Type& v)
            {
                if (para_table.contains<Type>(prefix + "." + name))
                    v = para_table.obj<Type>(prefix + "." + name);
                else
                    throw std::runtime_error("invalid key in key_condition::set. name=" + name);
            }

            template <typename Type>
            void set(Type& v)
            {
                if (para_table.contains<Type>(prefix))
                    v = para_table.obj<Type>(prefix);
                else
                    throw std::runtime_error("invalid key in key_condition::set. name=" + prefix);
            }
        
            inline bool feasible()
            {
                for (auto &key : para_table.all_keys)
                {
                    if (key.starts_with(prefix))
                        return true;
                }
                return false;
            }
        
            inline std::vector<std::string> keys()
            {
                std::set<std::string> ans;
                for (auto &key : para_table.all_keys)
                {
                    if (key.starts_with(prefix + "."))
                    {
                        std::string post_key = key.substr(prefix.size() + 1, key.size() - prefix.size() - 1);
                        ans.insert(post_key.substr(0, post_key.find(".")));
                    }
                }
                return std::vector<std::string>(ans.begin(), ans.end());
            }

            inline key_condition in(const std::string& name)
            {
                key_condition ans(para_table);
                ans.prefix = prefix + (prefix.empty() ? "" : ".") + name;
                if (!ans.feasible())
                    throw std::runtime_error("invalid key in key_condition::in. prefix = " + ans.prefix);
                return ans;
            }
        
            inline std::vector<int> idxs()
            {
                std::set<int> ans;
                for (auto &key : para_table.all_keys)
                {
                    if (key.starts_with(prefix + "("))
                    {
                        std::string post_key = key.substr(prefix.size() + 1, key.size() - prefix.size() - 1);
                        std::string idx_str = post_key.substr(0, post_key.find(")"));
                        try {
                            int idx = std::stoi(idx_str);
                            ans.insert(idx);
                        } catch (...) {
                            // ignore invalid index
                        }
                    }
                }
                return std::vector<int>(ans.begin(), ans.end());
            }
        
            inline key_condition at(int idx)
            {
                key_condition ans(para_table);
                ans.prefix = prefix + "(" + std::to_string(idx) + ")";
                if (!ans.feasible())
                    throw std::runtime_error("invalid key in key_condition::at. prefix = " + ans.prefix);
                return ans;
            }
        
            inline key_condition operator[](const std::string& name) {
                return in(name);
            }
            
            inline key_condition operator[](int idx){
                return at(idx);
            }
        };

        inline decltype(auto) in(const std::string& name)
        {
            key_condition ans(*this);
            return ans.in(name);
        }

        inline decltype(auto) operator[](const std::string& name) {
            return in(name);
        }

        template <typename Type>
        void set(const std::string& name, Type& v)
        {
            if (contains<Type>(name))
                v = obj<Type>(name);
            else
                throw std::runtime_error("invalid key in json_loader::set. name=" + name);
        }
    };

}

}

#endif