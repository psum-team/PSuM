#ifndef PSUM_FIELD_SOLVER_MIXED_BOUNDARY_HPP
#define PSUM_FIELD_SOLVER_MIXED_BOUNDARY_HPP

#include <functional>
#include <optional>
#include <stdexcept>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "../field/foundation.hpp"

namespace psum {

namespace field_solver {

template <int Dim>
struct mixed_boundary {
private:
    // two way to set value
    std::optional<double> value_var_;
    std::function<double(const std::array<double, Dim>&)> value_func_;
    // first: value coeff, second: grad coeff
    std::function<std::pair<double, double>(const std::array<double, Dim>&)> coeff_func_;
    // one boundary have only one direction
    using direction_type = psum::field::boundary_direction_traits<Dim>::type;
    std::optional<direction_type> direction_;
    // map: element_idx -> position
    std::unordered_map<size_t, std::array<double, Dim>> related_elements_;
    // for fast evaluation
    bool set_and_vec_built_ = false;
    std::unordered_set<size_t> element_indexes_set_;
    std::vector<std::pair<size_t, std::array<double, Dim>>> element_idx_pos_pairs_;

    void set_map(const std::unordered_map<size_t, std::array<double, Dim>>& related_elements) {
        related_elements_ = related_elements;
        set_and_vec_built_ = false;
    }

    void make_vector() {
        if (set_and_vec_built_) return;
        element_indexes_set_.clear();
        element_idx_pos_pairs_.clear();
        element_idx_pos_pairs_.reserve(related_elements_.size());
        for (auto& e : related_elements_) {
            element_indexes_set_.insert(e.first);
            element_idx_pos_pairs_.emplace_back(e.first, e.second);
        }
        set_and_vec_built_ = true;
    }

public:
    mixed_boundary() {}

    inline void set_related_elements(const std::unordered_map<size_t, std::array<double, Dim>>& related_elements) {
        set_map(related_elements);
    }

    inline mixed_boundary& operator=(double val) {
        value_var_ = val;
        value_func_ = nullptr;
        return *this;
    }

    inline mixed_boundary& operator=(const std::function<double(const std::array<double, Dim>&)>& func) {
        value_func_ = func;
        value_var_ = std::nullopt;
        return *this;
    }

    inline void erase_at(const std::vector<size_t> &erase_list) {
        for (auto& idx : erase_list) {
            related_elements_.erase(idx);
        }
        set_and_vec_built_ = false;
    }

    inline void evaluate(std::vector<double>& results) {
        make_vector();
        if (value_func_ && value_var_)
            throw std::runtime_error("mixed_boundary::evaluate: both value_var_ and value_func_ are set");
        
        if(value_func_) {
            results.resize(element_idx_pos_pairs_.size());
            for (size_t i = 0; i < element_idx_pos_pairs_.size(); i++) {
                results[i] = value_func_(element_idx_pos_pairs_[i].second);
            }
        } else if(value_var_) {
            results.resize(element_idx_pos_pairs_.size());
            for (size_t i = 0; i < element_idx_pos_pairs_.size(); i++) {
                results[i] = *value_var_;
            }
        } else {
            throw std::runtime_error("mixed_boundary::evaluate: no value is set");
        }
    }

    inline const auto& get_element_indexes() {
        make_vector();
        return element_indexes_set_;
    }

    inline void set_coeff_func(const std::function<std::pair<double, double>(const std::array<double, Dim>&)>& func) {
        coeff_func_ = func;
    }

    inline void set_direction(direction_type d) { direction_ = d; }

    inline auto evaluate_coeffs_at(size_t idx, direction_type d) const {
        if (!coeff_func_)
            throw std::runtime_error("mixed_boundary::evaluate_coeffs_at: coefficient functions are not set");
        if (related_elements_.count(idx) == 0)
            throw std::runtime_error("mixed_boundary::evaluate_coeffs_at: element is not related to this boundary");
        if (get_direction() != d)
            throw std::runtime_error("mixed_boundary::evaluate_coeffs_at: direction is different from argument");
        return coeff_func_(related_elements_.at(idx));
    }

    inline auto get_direction() const {
        if (!direction_)
            throw std::runtime_error("mixed_boundary::get_direction: direction is not set");
        return *direction_;
    }

    // the minimum unit of boundary conditions
    struct mixed_boundary_node {
        size_t element;
        direction_type direction;
        std::pair<double, double> coeffs;
    };
    
    inline auto get_boundary_nodes() {
        make_vector();
        std::vector<mixed_boundary_node> result;
        result.resize(element_idx_pos_pairs_.size());
        for (size_t i = 0; i < element_idx_pos_pairs_.size(); i++) {
            result[i].element = element_idx_pos_pairs_[i].first;
            result[i].direction = get_direction();
            if (coeff_func_)
                result[i].coeffs = coeff_func_(element_idx_pos_pairs_[i].second);
        }
        return result;
    }
};

template <int Dim>
struct mixed_boundary_list {
private:
    using direction_type = psum::field::boundary_direction_traits<Dim>::type;
    using mixed_boundary_node = typename mixed_boundary<Dim>::mixed_boundary_node;
    std::vector<mixed_boundary<Dim>> boundaries_;
    std::unordered_map<size_t, int> index_map_;    // {element_idx, direction} -> boundary_idx
    std::unordered_set<size_t> all_indexes_;       // all {element_idx, direction} in all boundaries

    static inline size_t get_key(size_t idx, direction_type d) {
        return idx * int(direction_type::Count) + static_cast<size_t>(d);
    }

public:
    mixed_boundary_list(const std::initializer_list<mixed_boundary<Dim>>& in) { set_related_boundaries(in); }
    mixed_boundary_list(){}
    inline void set_related_boundaries(const std::initializer_list<mixed_boundary<Dim>>& in) {
        all_indexes_.clear();
        index_map_.clear();
        std::vector<mixed_boundary<Dim>> tmp(in);
        std::vector<std::vector<size_t>> repeated_elements(tmp.size()); // boundary_idx -> repeated_elements

        // backward iteration: for the feature that the later ones cover the earlier ones
        for (int i = tmp.size() - 1; i >= 0; i--) {
            auto& indexes = tmp[i].get_element_indexes();
            for (auto& idx : indexes) {
                size_t k = get_key(idx, tmp[i].get_direction());
                if (all_indexes_.count(k) == 0) {
                    all_indexes_.insert(k);
                    index_map_[k] = i;
                } else {
                    repeated_elements[i].push_back(idx);
                }
            }
        }
        for (size_t i = 0; i < tmp.size(); i++) {
            if (repeated_elements[i].size() > 0) {
                tmp[i].erase_at(repeated_elements[i]);
            }
        }
        boundaries_ = tmp;
    }

    inline void erase_at(const std::vector<size_t> &erase_list) {
        std::vector<std::vector<size_t>> erase_elements(boundaries_.size());
        for (auto& idx : erase_list) {
            for (int i = 0; i < static_cast<int>(direction_type::Count); ++i) {
                direction_type d = static_cast<direction_type>(i);
                size_t k = get_key(idx, d);
                if (index_map_.count(k) == 0) {
                    continue;
                } else {
                    int boundary_idx = index_map_[k];
                    erase_elements[boundary_idx].push_back(idx);
                    index_map_.erase(k);
                    all_indexes_.erase(k);
                }
            }
        }
        for (size_t i = 0; i < boundaries_.size(); i++) {
            boundaries_[i].erase_at(erase_elements[i]);
        }
    }

    inline void evaluate(std::vector<double>& result_v) {
        result_v.clear();
        result_v.reserve(all_indexes_.size());
        std::vector<double> single_result;
        // split result
        for (size_t i = 0; i < boundaries_.size(); i++) {
            boundaries_[i].evaluate(single_result);
            result_v.insert(result_v.end(), single_result.begin(), single_result.end());
        }
    }

    inline size_t elements_size() const {
        return all_indexes_.size();
    }

    inline bool element_exists(size_t idx, direction_type d) const {
        return all_indexes_.count(get_key(idx, d)) > 0;
    }

    inline bool element_exists(size_t idx) const {
        for (int i = 0; i < static_cast<int>(direction_type::Count); ++i) {
            direction_type d = static_cast<direction_type>(i);
            if (all_indexes_.count(get_key(idx, d)) > 0) {
                return true;
            }
        }
        return false;
    }

    inline auto evaluate_coeffs_at(size_t idx, direction_type d) const {
        if (index_map_.count(get_key(idx, d)) == 0)
            throw std::runtime_error("mixed_boundary_list::evaluate_coeffs_at: element is not related to any boundary");
        int boundary_idx = index_map_.at(get_key(idx, d));
        return boundaries_[boundary_idx].evaluate_coeffs_at(idx, d);
    }

    inline auto get_boundary_nodes() {
        std::vector<mixed_boundary_node> result;
        for (auto& b : boundaries_) {
            auto nodes = b.get_boundary_nodes();
            result.insert(result.end(), nodes.begin(), nodes.end());
        }
        return result;
    }

};

using mixed_boundary_1d = mixed_boundary<1>;
using mixed_boundary_1d_list = mixed_boundary_list<1>;
using mixed_boundary_2d = mixed_boundary<2>;
using mixed_boundary_2d_list = mixed_boundary_list<2>;
using mixed_boundary_3d = mixed_boundary<3>;
using mixed_boundary_3d_list = mixed_boundary_list<3>;

}

}

#endif
