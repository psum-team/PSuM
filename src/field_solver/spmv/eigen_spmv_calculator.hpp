#ifndef PSUM_FIELD_SOLVER_SPMV_EIGEN_SPMV_CALCULATOR_HPP
#define PSUM_FIELD_SOLVER_SPMV_EIGEN_SPMV_CALCULATOR_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <cstring>
#include <stdexcept>
#include "spmv_calculator.hpp"

namespace psum {

namespace field_solver {

namespace spmv {

class eigen_spmv_calculator : public spmv_calculator {
public:
    inline eigen_spmv_calculator() = default;

    explicit inline eigen_spmv_calculator(const Eigen::SparseMatrix<double>& matrix) {
        set_matrix(matrix);
    }

    inline void set_matrix(const Eigen::SparseMatrix<double>& matrix) override {
        matrix_ = matrix;
        matrix_.makeCompressed();
        scratch_.resize(matrix_.rows());
    }

    inline void apply(const double* x, double* y) override {
        Eigen::Map<const Eigen::VectorXd> x_in(x, matrix_.cols());
        Eigen::Map<Eigen::VectorXd> y_out(y, matrix_.rows());
        y_out = matrix_ * x_in;
    }

    inline void apply_inplace(double* buf) override {
        if (matrix_.rows() != matrix_.cols()) {
            throw std::runtime_error("eigen_spmv_calculator::apply_inplace requires a square matrix");
        }
        Eigen::Map<const Eigen::VectorXd> x_in(buf, matrix_.cols());
        scratch_ = matrix_ * x_in;
        std::memcpy(buf, scratch_.data(), sizeof(double) * scratch_.size());
    }

    inline unsigned long long rows() const override {
        return static_cast<unsigned long long>(matrix_.rows());
    }

    inline unsigned long long cols() const override {
        return static_cast<unsigned long long>(matrix_.cols());
    }

private:
    Eigen::SparseMatrix<double> matrix_;
    Eigen::VectorXd scratch_;
};

}

}

}

#endif
