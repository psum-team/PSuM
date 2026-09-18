#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_SCHUR_COMPLEMENT_SCHUR_COMPLEMENT_CORE_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_SCHUR_COMPLEMENT_SCHUR_COMPLEMENT_CORE_HPP

#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/LU>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace psum {

namespace field_solver {

namespace implements {

namespace schur_complement {

enum class schur_contribution_backend {
    cpu,
    external
};

inline int effective_thread_count(int requested) {
#ifdef _OPENMP
    if (requested > 0) {
        return requested;
    }
    return omp_get_max_threads();
#else
    (void)requested;
    return 1;
#endif
}

struct schur_complement_core {
    int n = 0;
    int n_interface = 0;
    int n_interior = 0;

    std::vector<std::pair<int, int>> interface_map;
    std::vector<std::pair<int, int>> interior_map;

    Eigen::SparseMatrix<double> mat_A;
    Eigen::SparseMatrix<double> mat_B;
    Eigen::SparseMatrix<double> mat_C;

    Eigen::MatrixXd S;
    Eigen::MatrixXd S_inv;

    void setup(const Eigen::SparseMatrix<double>& A_global,
               const std::vector<int>& selected,
               int num_threads = 0,
               schur_contribution_backend contribution_backend = schur_contribution_backend::cpu) {
        const int thread_count = effective_thread_count(num_threads);
        n = static_cast<int>(A_global.rows());
        if (static_cast<int>(A_global.cols()) != n || n == 0) {
            throw std::runtime_error("schur_complement_core: A_global must be square and non-empty");
        }

        std::vector<int> sel = selected;
        std::sort(sel.begin(), sel.end());
        sel.erase(std::unique(sel.begin(), sel.end()), sel.end());
        for (int s : sel) {
            if (s < 0 || s >= n) {
                throw std::runtime_error("schur_complement_core: selected index out of range");
            }
        }

        n_interface = static_cast<int>(sel.size());
        n_interior = n - n_interface;
        if (n_interface == 0) {
            throw std::runtime_error("schur_complement_core: interface set must be non-empty");
        }

        std::vector<int> global_to_interface(n, -1);
        for (int i = 0; i < n_interface; ++i) {
            global_to_interface[sel[i]] = i;
        }

        interface_map.resize(n_interface);
#pragma omp parallel for if(n_interface > 1024) num_threads(thread_count)
        for (int i = 0; i < n_interface; ++i) {
            interface_map[i] = {sel[i], i};
        }

        interior_map.resize(n_interior);
        std::vector<int> global_to_interior(n, -1);
        {
#pragma omp parallel for if(n > 4096) num_threads(thread_count)
            for (int g = 0; g < n; ++g) {
                if (global_to_interface[g] < 0) {
                    global_to_interior[g] = g - static_cast<int>(std::upper_bound(sel.begin(), sel.end(), g) - sel.begin());
                }
            }

#pragma omp parallel for if(n_interior > 1024) num_threads(thread_count)
            for (int g = 0; g < n; ++g) {
                if (global_to_interface[g] < 0) {
                    const int local = global_to_interior[g];
                    interior_map[local] = {g, local};
                }
            }
        }

        std::vector<Eigen::Triplet<double>> a_triplets, b_triplets, c_triplets, d_triplets;
        const int outer_size = A_global.outerSize();

        if (thread_count <= 1 || outer_size < 1024) {
            a_triplets.reserve(A_global.nonZeros());
            b_triplets.reserve(A_global.nonZeros());
            c_triplets.reserve(A_global.nonZeros());
            d_triplets.reserve(A_global.nonZeros());

            for (int col = 0; col < outer_size; ++col) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A_global, col); it; ++it) {
                    const int row = it.row();
                    const double val = it.value();
                    const bool row_if = global_to_interface[row] >= 0;
                    const bool col_if = global_to_interface[col] >= 0;

                    if (row_if && col_if) {
                        d_triplets.emplace_back(global_to_interface[row], global_to_interface[col], val);
                    } else if (!row_if && !col_if) {
                        a_triplets.emplace_back(global_to_interior[row], global_to_interior[col], val);
                    } else if (!row_if && col_if) {
                        b_triplets.emplace_back(global_to_interior[row], global_to_interface[col], val);
                    } else {
                        c_triplets.emplace_back(global_to_interface[row], global_to_interior[col], val);
                    }
                }
            }
        } else {
            struct local_triplets {
                std::vector<Eigen::Triplet<double>> a;
                std::vector<Eigen::Triplet<double>> b;
                std::vector<Eigen::Triplet<double>> c;
                std::vector<Eigen::Triplet<double>> d;
            };

            std::vector<local_triplets> local(static_cast<std::size_t>(thread_count));

#pragma omp parallel num_threads(thread_count)
            {
                const int tid =
#ifdef _OPENMP
                    omp_get_thread_num();
#else
                    0;
#endif
                auto& out = local[static_cast<std::size_t>(tid)];

#pragma omp for schedule(static)
                for (int col = 0; col < outer_size; ++col) {
                    for (Eigen::SparseMatrix<double>::InnerIterator it(A_global, col); it; ++it) {
                        const int row = it.row();
                        const double val = it.value();
                        const bool row_if = global_to_interface[row] >= 0;
                        const bool col_if = global_to_interface[col] >= 0;

                        if (row_if && col_if) {
                            out.d.emplace_back(global_to_interface[row], global_to_interface[col], val);
                        } else if (!row_if && !col_if) {
                            out.a.emplace_back(global_to_interior[row], global_to_interior[col], val);
                        } else if (!row_if && col_if) {
                            out.b.emplace_back(global_to_interior[row], global_to_interface[col], val);
                        } else {
                            out.c.emplace_back(global_to_interface[row], global_to_interior[col], val);
                        }
                    }
                }
            }

            auto total_size = [](const auto& parts, auto member) {
                std::size_t total = 0;
                for (const auto& p : parts) {
                    total += (p.*member).size();
                }
                return total;
            };
            a_triplets.reserve(total_size(local, &local_triplets::a));
            b_triplets.reserve(total_size(local, &local_triplets::b));
            c_triplets.reserve(total_size(local, &local_triplets::c));
            d_triplets.reserve(total_size(local, &local_triplets::d));
            for (auto& p : local) {
                a_triplets.insert(a_triplets.end(), p.a.begin(), p.a.end());
                b_triplets.insert(b_triplets.end(), p.b.begin(), p.b.end());
                c_triplets.insert(c_triplets.end(), p.c.begin(), p.c.end());
                d_triplets.insert(d_triplets.end(), p.d.begin(), p.d.end());
            }
        }

        mat_A.resize(n_interior, n_interior);
        mat_B.resize(n_interior, n_interface);
        mat_C.resize(n_interface, n_interior);
        mat_A.setFromTriplets(a_triplets.begin(), a_triplets.end());
        mat_B.setFromTriplets(b_triplets.begin(), b_triplets.end());
        mat_C.setFromTriplets(c_triplets.begin(), c_triplets.end());
        mat_A.makeCompressed();
        mat_B.makeCompressed();
        mat_C.makeCompressed();

        S = Eigen::MatrixXd::Zero(n_interface, n_interface);
        {
            std::vector<Eigen::Triplet<double>> d_sorted(d_triplets);
            Eigen::SparseMatrix<double> D(n_interface, n_interface);
            D.setFromTriplets(d_sorted.begin(), d_sorted.end());
            S = Eigen::MatrixXd(D);
        }

        if (n_interior == 0) {
            S_inv = S.inverse();
            return;
        }

        if (contribution_backend == schur_contribution_backend::external) {
            return;
        }

        Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
        lu.analyzePattern(mat_A);
        lu.factorize(mat_A);
        if (lu.info() != Eigen::Success) {
            throw std::runtime_error("schur_complement_core: A factorization failed");
        }

        constexpr std::size_t max_rhs_chunk_bytes = 256ULL * 1024ULL * 1024ULL;
        int chunk_cols = n_interface;
        if (n_interior > 0) {
            const std::size_t bytes_per_col = static_cast<std::size_t>(n_interior) * sizeof(double);
            chunk_cols = static_cast<int>(std::max<std::size_t>(1, max_rhs_chunk_bytes / bytes_per_col));
            chunk_cols = std::min(chunk_cols, n_interface);
        }
        for (int begin = 0; begin < n_interface; begin += chunk_cols) {
            const int cols = std::min(chunk_cols, n_interface - begin);
            Eigen::MatrixXd B_chunk = Eigen::MatrixXd(mat_B.middleCols(begin, cols));
            Eigen::MatrixXd W_chunk = lu.solve(B_chunk);
            if (lu.info() != Eigen::Success) {
                throw std::runtime_error("schur_complement_core: A solve for Schur contribution failed");
            }
            S.middleCols(begin, cols) -= Eigen::MatrixXd(mat_C * W_chunk);
        }

        S_inv = S.inverse();
    }

    void finalize_inverse() {
        S_inv = S.inverse();
    }
};

}

}

}

}

#endif
