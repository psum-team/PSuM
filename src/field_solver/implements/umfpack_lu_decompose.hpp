#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_UMFPACK_LU_DECOMPOSE_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_UMFPACK_LU_DECOMPOSE_HPP

#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/SparseLU> // for PermutationMatrix
#include <suitesparse/umfpack.h>

namespace psum {

namespace field_solver {

namespace implements {

struct umfpack_lu_result {
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P;
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> Q;
    Eigen::SparseMatrix<double> L;
    Eigen::SparseMatrix<double> U;
};

inline umfpack_lu_result umfpack_lu_decompose(const Eigen::SparseMatrix<double>& A) {

    assert(A.isCompressed());
    int n = A.cols();
    assert(A.rows() == n);

    umfpack_lu_result result;
    result.P.resize(n);
    result.Q.resize(n);

    const int* Ap = A.outerIndexPtr();
    const int* Ai = A.innerIndexPtr();
    const double* Ax = A.valuePtr();

    void* Symbolic = nullptr;
    void* Numeric = nullptr;
    double Control[20];
    int status;

    umfpack_di_defaults(Control);
    Control[UMFPACK_SCALE] = UMFPACK_SCALE_NONE; // for simpler

    status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, Control, nullptr);
    if (status != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK symbolic factorization failed");
    }

    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, Control, nullptr);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK) {
        umfpack_di_free_numeric(&Numeric);
        throw std::runtime_error("UMFPACK numeric factorization failed");
    }

    // get number
    int lnz, unz, nrows, ncols, nz_udiag;
    status = umfpack_di_get_lunz(&lnz, &unz, &nrows, &ncols, &nz_udiag, Numeric);
    if (status != UMFPACK_OK) {
        umfpack_di_free_numeric(&Numeric);
        throw std::runtime_error("UMFPACK get_lunz failed");
    }

    std::vector<int> Lp(n + 1), Lj(lnz);
    std::vector<double> Lx(lnz);
    std::vector<int> Up(n + 1), Ui(unz);
    std::vector<double> Ux(unz);
    std::vector<int> P(n), Q(n);
    std::vector<double> Rs(n);
    int do_recip = 0;

    status = umfpack_di_get_numeric(
        Lp.data(), Lj.data(), Lx.data(),
        Up.data(), Ui.data(), Ux.data(),
        P.data(), Q.data(),
        NULL, 
        &do_recip, NULL,
        Numeric
    );
    umfpack_di_free_numeric(&Numeric);
    if (status != UMFPACK_OK) {
        throw std::runtime_error("UMFPACK get_numeric failed");
    }

    // make L matrix
    std::vector<Eigen::Triplet<double>> L_triplets;
    for (int i = 0; i < n; ++i) {
        for (int k = Lp[i]; k < Lp[i + 1]; ++k) {
            int j = Lj[k];
            double val = Lx[k];
            L_triplets.emplace_back(i, j, val);
        }
    }
    Eigen::SparseMatrix<double> L(n, n);
    L.setFromTriplets(L_triplets.begin(), L_triplets.end());

    // make U matrix
    std::vector<Eigen::Triplet<double>> U_triplets;
    // different with L matrix cosntruction
    for (int j = 0; j < n; ++j) {
        for (int k = Up[j]; k < Up[j + 1]; ++k) {
            int i = Ui[k];
            double val = Ux[k];
            U_triplets.emplace_back(i, j, val);
        }
    }
    Eigen::SparseMatrix<double> U(n, n);
    U.setFromTriplets(U_triplets.begin(), U_triplets.end());

    // make P, Q matrix
    Eigen::VectorXi P_indices(n), Q_indices(n);
    for (int k = 0; k < n; ++k) {
        P_indices[P[k]] = k;
        Q_indices[k] = Q[k];
    }
    result.P.indices() = P_indices;
    result.Q.indices() = Q_indices;

    result.L = std::move(L);
    result.U = std::move(U);

    return result;
}

}

}

}

#endif