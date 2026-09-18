#ifndef PSUM_FIELD_SOLVER_SPMV_SPMV_CALCULATOR_HPP
#define PSUM_FIELD_SOLVER_SPMV_SPMV_CALCULATOR_HPP

#include <Eigen/SparseCore>

namespace psum {

namespace field_solver {

namespace spmv {

class spmv_calculator {
public:
    virtual ~spmv_calculator() = default;

    virtual void set_matrix(const Eigen::SparseMatrix<double>& matrix) = 0;
    virtual void apply(const double* x, double* y) = 0;

    // Uses implementation-owned scratch storage and is not thread-safe.
    virtual void apply_inplace(double* buf) = 0;

    virtual unsigned long long rows() const = 0;
    virtual unsigned long long cols() const = 0;
};

}

}

}

#endif
