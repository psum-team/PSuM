#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SPARSE_LINEAR_TRAIT_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_SPARSE_LINEAR_TRAIT_HPP

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <type_traits>

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template <typename MatrixType>
struct matrix_traits;

template <typename VectorType>
struct vector_traits;

template <>
struct matrix_traits<Eigen::SparseMatrix<double>> {
    using Scalar = double;
    using Vector = Eigen::VectorXd;

    static void multiply(const Eigen::SparseMatrix<double>& mat, const Vector& in, Vector& out) {
        out = mat * in;
    }

    static void multiply_add(const Eigen::SparseMatrix<double>& mat, const Vector& x,
                             const Vector& b, Vector& out, double b_coeff = 1.0, double ans_coeff = 1.0) {
        Eigen::VectorXd tmp = mat * x;
        out.noalias() = (tmp * ans_coeff + b * (b_coeff * ans_coeff));
    }

    static void set(Eigen::SparseMatrix<double>& dest, const Eigen::SparseMatrix<double>& src) {
        dest = src;
    }
};

template <>
struct matrix_traits<Eigen::SparseMatrix<double, Eigen::RowMajor>> {
    using Scalar = double;
    using Vector = Eigen::VectorXd;

    static void multiply(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const Vector& in, Vector& out) {
        out = mat * in;
    }

    static void multiply_add(const Eigen::SparseMatrix<double, Eigen::RowMajor>& mat, const Vector& x,
                             const Vector& b, Vector& out, double b_coeff = 1.0, double ans_coeff = 1.0) {
        Eigen::VectorXd tmp = mat * x;
        out.noalias() = (tmp * ans_coeff + b * (b_coeff * ans_coeff));
    }

    static void set(Eigen::SparseMatrix<double, Eigen::RowMajor>& dest, const Eigen::SparseMatrix<double>& src) {
        dest = src;
    }
};

template <>
struct vector_traits<Eigen::VectorXd> {
    static Eigen::VectorXd add(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a + b;
    }

    static Eigen::VectorXd subtract(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a - b;
    }

    static Eigen::VectorXd hadamard_product(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
        return a.array() * b.array();
    }

    static Eigen::VectorXd multiply_with_array(const Eigen::VectorXd& vec, const Eigen::ArrayXd& arr) {
        return vec.array() * arr;
    }

    static void set(Eigen::VectorXd& dest, const Eigen::VectorXd& src) {
        dest = src;
    }

    static void set(std::vector<Eigen::VectorXd>& dest, const std::vector<Eigen::ArrayXd>& src) {
        dest.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            dest[i] = src[i];
        }
    }

    static void copy_data(Eigen::VectorXd& dst, const Eigen::VectorXd& src) {
        dst = src;
    }

    static void hadamard_inplace(Eigen::VectorXd& vec, const Eigen::VectorXd& arr) {
        vec.array() *= arr.array();
    }

    static void hadamard_to(const Eigen::VectorXd& in, const Eigen::VectorXd& arr, Eigen::VectorXd& out) {
        out = in.array() * arr.array();
    }
};

}

}

}

}

#endif
