#ifndef PSUM_SERIALIZATION_SIMPLIFY_TYPE_NAME_HPP
#define PSUM_SERIALIZATION_SIMPLIFY_TYPE_NAME_HPP

#include <regex>

namespace psum {

namespace serialization {

    std::string simplify_single_eigen_matrix(const std::string& scalar_type, 
                                           const std::string& rows_str, 
                                           const std::string& cols_str) {
        char suffix = '?';
        if (scalar_type.find("double") != std::string::npos) suffix = 'd';
        else if (scalar_type.find("float") != std::string::npos) suffix = 'f';
        else if (scalar_type.find("int") != std::string::npos) suffix = 'i';
        else if (scalar_type.find("long") != std::string::npos) suffix = 'l';
        else {
            return "Eigen::Matrix<" + scalar_type + ", " + rows_str + ", " + cols_str + ">";
        }

        int rows = -1, cols = -1;
        try {
            rows = std::stoi(rows_str);
            cols = std::stoi(cols_str);
        } catch (...) {
            return "Eigen::Matrix<" + scalar_type + ", " + rows_str + ", " + cols_str + ">";
        }

        if (rows == 1) {
            if (cols == -1) {
                return "Eigen::RowVectorX" + std::string(1, suffix);
            } else if (cols < 10) {
                return "Eigen::RowVector" + std::to_string(cols) + suffix;
            }
        } else if (cols == 1) {
            if (rows == -1) {
                return "Eigen::VectorX" + std::string(1, suffix);
            } else if (rows < 10) {
                return "Eigen::Vector" + std::to_string(rows) + suffix;
            }
        } else {
            if (rows == -1 && cols == -1) {
                return "Eigen::MatrixX" + std::string(1, suffix);
            } else if (rows == -1 && cols < 10) {
                return "Eigen::MatrixX" + std::to_string(cols) + std::string(1, suffix);
            } else if (rows < 10 && cols == -1) {
                return "Eigen::Matrix" + std::to_string(rows) + "X" + std::string(1, suffix);
            } else if (rows == cols && cols < 10) {
                return "Eigen::Matrix" + std::to_string(rows) + suffix;
            }
        }

        return "Eigen::Matrix<" + scalar_type + ", " + std::to_string(rows) + ", " + std::to_string(cols) + ">";
    }

    std::string simplify_eigen_types(const std::string& input) {
        std::string result = input;
        size_t pos = result.find("Eigen::Matrix<");
        while (pos != std::string::npos) {
            size_t end = result.find('>', pos);
            if (end == std::string::npos) break;
            std::string inside = result.substr(pos + 13, end - pos - 13);

            std::istringstream ss(inside);
            std::string scalar, row, col;
            std::getline(ss, scalar, ',');
            std::getline(ss, row, ',');
            std::getline(ss, col, ',');
            
            scalar.erase(0, scalar.find_first_not_of(" \t"));
            scalar.erase(scalar.find_last_not_of(" \t") + 1);
            row.erase(0, row.find_first_not_of(" \t"));
            row.erase(row.find_last_not_of(" \t") + 1);
            col.erase(0, col.find_first_not_of(" \t"));
            col.erase(col.find_last_not_of(" \t") + 1);

            std::string simplified = simplify_single_eigen_matrix(scalar, row, col);
            result.replace(pos, end - pos + 1, simplified);

            pos = result.find("Eigen::Matrix<", pos + simplified.size());
        }
        return result;
    }

    std::string simplify_reduce_params(const std::string& input, const std::string& template_str, int n_params) {
        std::string result = input;
        size_t pos = 0;
        while ((pos = result.find(template_str + "<", pos)) != std::string::npos) {
            size_t start = pos + template_str.size();
            int depth = 0;
            size_t end = start;
            for (; end < result.size(); ++end) {
                if (result[end] == '<') ++depth;
                else if (result[end] == '>') --depth;
                if (depth == 0) break;
            }
            if (depth != 0) break;

            std::vector<std::pair<size_t, size_t>> param_positions;
            param_positions.resize(n_params);

            int param_depth = 1;
            size_t p_start = start + 1;
            size_t p_end = p_start;
            for (int p_idx = 0; p_idx < n_params; ++p_idx) {
                for (; p_end <= end; ++p_end) {
                    if (result[p_end] == '<') ++param_depth;
                    else if (result[p_end] == '>') --param_depth;
                    if ((result[p_end] == ',' && param_depth == 1) || p_end == end) {
                        param_positions[p_idx] = {p_start, p_end};
                        p_start = p_end + 1;
                        p_end = p_start;
                        break;
                    }
                }
                if (p_end == end && p_idx != n_params - 1)
                    throw std::runtime_error("Invalid template parameter list in " + input + ", expected " + std::to_string(n_params) + " parameters");
            }

            std::string new_params;
            for (int p_idx = 0; p_idx < n_params; ++p_idx) {
                std::string param_type = result.substr(param_positions[p_idx].first, param_positions[p_idx].second - param_positions[p_idx].first);
                new_params += param_type;
                if (p_idx != n_params - 1) new_params += ",";
            }

            result.replace(start, end - start + 1, "<" + new_params + ">");
            pos = start;
        }
        return result;
    }


    std::string simplify_type_name(const std::string& input) {
        std::string result = input;

        const std::vector<std::pair<std::regex, std::string>> rules = {
            // remove trivial namespace
            {std::regex("\\bpsum::tag::"), ""},
            {std::regex("\\bpsum::serialization::"), ""},
            {std::regex("\\bstd::__cxx11::"), "std::"},

            // simplify std::string
            {std::regex("std::basic_string<char, std::char_traits<char>, std::allocator<char> >"), "std::string"},
        
            // 3. std::vector<T, allocator<T>> → std::vector<T>
            {std::regex("std::vector<([^,<>]+), std::allocator<\\1> >"), "std::vector<$1>"},
        
            // 4. std::map<K,V,...> → std::map<K,V>
            {std::regex("std::map<([^,<>]+), ([^,<>]+), std::less<\\1>, std::allocator<std::pair<const \\1, \\2> > >"),
             "std::map<$1, $2>"},
            
            // 5. std::unordered_map<K,V,...> → std::unordered_map<K,V>
            {std::regex("std::unordered_map<([^,<>]+), ([^,<>]+), std::hash<\\1>, std::equal_to<\\1>, std::allocator<std::pair<const \\1, \\2> > >"),
             "std::unordered_map<$1, $2>"},
            
            {std::regex(" >"), ">"},
            {std::regex("< "), "<"}
        };

        bool changed;
        do {
            changed = false;
            for (auto& [pattern, replacement] : rules) {
                std::string new_result = std::regex_replace(result, pattern, replacement);
                if (new_result != result) {
                    result = std::move(new_result);
                    changed = true;
                }
            }
        } while (changed);

        // simplify Eigen types
        result = serialization::simplify_eigen_types(result);
        // simplify std::vector
        result = serialization::simplify_reduce_params(result, "std::vector", 1);
        // simplify std::map and std::unordered_map
        result = serialization::simplify_reduce_params(result, "std::map", 2);
        result = serialization::simplify_reduce_params(result, "std::unordered_map", 2);

        return result;
    }

}

}

#endif