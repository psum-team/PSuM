#ifndef PSUM_FIELD_SOLVER_LOCAL_EQUATION_HPP
#define PSUM_FIELD_SOLVER_LOCAL_EQUATION_HPP

#include <cmath>
#include <map>

namespace psum {

namespace field_solver {

struct local_equation {
private:
    std::map<size_t, double> terms_;
    static constexpr double eps_ = 1e-14;

public:
    local_equation() = default;
    local_equation(std::initializer_list<std::pair<const size_t, double>> init) : terms_(init) {}
    local_equation(const std::map<size_t, double>& terms) : terms_(terms) {}

    inline local_equation& operator*=(double scalar) {
        for (auto& [index, coeff] : terms_) {
            coeff *= scalar;
        }
        if (std::abs(scalar) == 0) {
            terms_.clear();
        }
        return *this;
    }

    inline local_equation& operator+=(const local_equation& other) {
        for (const auto& [index, coeff] : other.terms_) {
            if (terms_.count(index) == 0) {
                terms_[index] = coeff;
            } else {
                double old_scale = std::abs(coeff) + std::abs(terms_[index]);
                terms_[index] += coeff;
                if (std::abs(terms_[index]) < old_scale * eps_) {
                    terms_.erase(index);
                }
            }
        }
        return *this;
    }

    inline local_equation& operator-=(const local_equation& other) {
        for (const auto& [index, coeff] : other.terms_) {
            if (terms_.count(index) == 0) {
                terms_[index] = -coeff;
            } else {
                double old_scale = std::abs(coeff) + std::abs(terms_[index]);
                terms_[index] -= coeff;
                if (std::abs(terms_[index]) < old_scale * eps_) {
                    terms_.erase(index);
                }
            }
        }
        return *this;
    }

    inline auto& get_terms() const { return terms_; }

    double norm_inf() const {
         double max_coeff = 0;
         for (const auto& [index, coeff] : terms_) {
             max_coeff = std::max(max_coeff, std::abs(coeff));
         }
         return max_coeff;
    }
};

inline local_equation operator*(local_equation lhs, double rhs) { return lhs *= rhs; }
inline local_equation operator*(double lhs, local_equation rhs) { return rhs *= lhs; }
inline local_equation operator+(local_equation lhs, const local_equation& rhs) { return lhs += rhs; }
inline local_equation operator-(local_equation lhs, const local_equation& rhs) { return lhs -= rhs; }

}

}

#endif