#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_ALGORITHM_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_ALGORITHM_HPP

#include <Eigen/SparseLU>
#include <vector>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include "sparse_linear_trait.hpp"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template <typename MatrixType, typename InnerSolverType>
struct multigrid_core {
    using Vector = typename matrix_traits<MatrixType>::Vector;
    using EigenSpMatrix = Eigen::SparseMatrix<double>;

    InnerSolverType inner_solver_;
    std::vector<Vector> dmat_inv_;
    std::vector<MatrixType> relax_mats_;
    std::vector<MatrixType> restrict_mats_;
    std::vector<MatrixType> interpolate_mats_;
    std::vector<MatrixType> a_mats_;
    std::vector<Vector> b_items_;
    std::vector<Vector> x_items_;
    std::vector<Vector> buf_vec_;

    enum class ShiftGrid {
        go_coarser,
        go_finer,
        stay
    };

    // Legacy interface for externally supplied transfer operators. Structured
    // callers use the explicit R/P overload below, including boundary weights.
    inline void setup(const EigenSpMatrix& A, const std::vector<EigenSpMatrix>& RMats, double c_coeff)
    {
        std::vector<EigenSpMatrix> PMats;
        for (const auto& R : RMats) PMats.emplace_back(R.transpose() * c_coeff);
        setup(A, RMats, PMats);
    }

    inline void setup(const EigenSpMatrix& A, const std::vector<EigenSpMatrix>& RMats,
                      const std::vector<EigenSpMatrix>& PMats)
    {
        std::vector<Eigen::ArrayXd> dmat_inv_tmp;
        std::vector<EigenSpMatrix> relax_mats_tmp;
        std::vector<EigenSpMatrix> restrict_mats_tmp;
        std::vector<EigenSpMatrix> interpolate_mats_tmp;
        std::vector<EigenSpMatrix> a_mats_tmp;
        
        if (A.cols() != A.rows() || RMats.size() <= 0)
            throw std::runtime_error("invalid setup parameter");
        
        dmat_inv_tmp.resize(RMats.size() + 1);
        dmat_inv_tmp[0].resize(A.cols());
        for (int i = 0; i < A.cols(); i++)
        {
            dmat_inv_tmp[0][i] = 1 / A.coeff(i, i);
        }
        
        restrict_mats_tmp.clear();
        for(auto& i: RMats)
        {
            EigenSpMatrix temp = i;
            restrict_mats_tmp.push_back(temp);
        }
        
        relax_mats_tmp.resize(restrict_mats_tmp.size());
        interpolate_mats_tmp.resize(restrict_mats_tmp.size());
        a_mats_tmp.resize(restrict_mats_tmp.size() + 1);
        b_items_.resize(a_mats_tmp.size());
        x_items_.resize(a_mats_tmp.size());
        buf_vec_.resize(a_mats_tmp.size());

        a_mats_tmp[0] = A;

        for (int i = 0; i < a_mats_tmp[0].outerSize(); i++)
        {
            for (EigenSpMatrix::InnerIterator it(a_mats_tmp[0], i); it; ++it)
            {
                it.valueRef() *= dmat_inv_tmp[0][it.row()];
            }
        }
        
        for (int i = 0; i < restrict_mats_tmp.size(); i++)
        {
            interpolate_mats_tmp[i] = PMats[i];
        }
        
        for (int i = 1; i < a_mats_tmp.size(); i++)
        {
            EigenSpMatrix temp = restrict_mats_tmp[i - 1] * a_mats_tmp[i - 1] * interpolate_mats_tmp[i - 1];
            a_mats_tmp[i] = temp;
            a_mats_tmp[i] = a_mats_tmp[i].pruned();

            dmat_inv_tmp[i].resize(a_mats_tmp[i].cols());
            for (int j = 0; j < a_mats_tmp[i].cols(); j++)
            {
                dmat_inv_tmp[i][j] = 1 / a_mats_tmp[i].coeff(j, j);
            }
            for (int j = 0; j < a_mats_tmp[i].outerSize(); j++)
            {
                for (EigenSpMatrix::InnerIterator it(a_mats_tmp[i], j); it; ++it)
                {
                    it.valueRef() *= dmat_inv_tmp[i][it.row()];
                }
            }
        }
        
        for (int i = 0; i < a_mats_tmp.size(); i++)
        {
            x_items_[i].resize(a_mats_tmp[i].cols());
            b_items_[i].resize(a_mats_tmp[i].cols());
            buf_vec_[i].resize(a_mats_tmp[i].cols());
        }

        EigenSpMatrix A_coarse = a_mats_tmp.back();
        inner_solver_.compute(A_coarse);
        
        for (int i = 0; i < relax_mats_tmp.size(); i++)
        {
            EigenSpMatrix Eye(a_mats_tmp[i].rows(), a_mats_tmp[i].cols());
            Eye.setIdentity();
            relax_mats_tmp[i] = Eye - a_mats_tmp[i] * 2.0 / 3;
        }
        vector_traits<Vector>::set(dmat_inv_, dmat_inv_tmp);

        relax_mats_.resize(relax_mats_tmp.size());
        for (int i = 0; i < relax_mats_.size(); i++)
            matrix_traits<MatrixType>::set(relax_mats_[i], relax_mats_tmp[i]);
        
        restrict_mats_.resize(restrict_mats_tmp.size());
        for (int i = 0; i < restrict_mats_.size(); i++)
            matrix_traits<MatrixType>::set(restrict_mats_[i], restrict_mats_tmp[i]);
        
        interpolate_mats_.resize(interpolate_mats_tmp.size());
        for (int i = 0; i < interpolate_mats_.size(); i++)
            matrix_traits<MatrixType>::set(interpolate_mats_[i], interpolate_mats_tmp[i]);

        a_mats_.resize(a_mats_tmp.size());
        for (int i = 0; i < a_mats_.size(); i++)
            matrix_traits<MatrixType>::set(a_mats_[i], a_mats_tmp[i]);
    }

    inline void set_relax_counts(int pre, int post)
    {
        pre_relax_ = pre;
        post_relax_ = post;
    }

    inline void v_cycle(const Vector& in, Vector& out)
    {
        v_cycle_multi(in, out, 1);
    }

    inline void v_cycle_multi(const Vector& in, Vector& out, int cycles)
    {
        if (relax_mats_.size() != restrict_mats_.size() || restrict_mats_.size() != interpolate_mats_.size()
            || relax_mats_.size() == 0 || b_items_.size() != relax_mats_.size() + 1 
            || b_items_.size() != x_items_.size() || a_mats_.size() != x_items_.size())
        {
            throw std::runtime_error("invalid multigrid_core status");
        }
        
        if (cached_cycles_ == cycles && cached_levels_ == relax_mats_.size()
            && cached_pre_ == pre_relax_ && cached_post_ == post_relax_) {
            solve_sequence(cached_sequence_, in, out);
            return;
        }
        std::vector<std::pair<int, ShiftGrid>> opt_seq;

        for (int i = 0; i < cycles; i++)
        {
            for (int stage = 0; stage < relax_mats_.size(); stage++)
            {
                opt_seq.push_back(std::make_pair(pre_relax_, ShiftGrid::go_coarser));
            }
            opt_seq.push_back(std::make_pair(0, ShiftGrid::go_finer));
            for (int stage = relax_mats_.size() - 1; stage >= 0; stage--)
            {
                opt_seq.push_back(std::make_pair(post_relax_, stage != 0 ? ShiftGrid::go_finer : ShiftGrid::stay));
            }
        }
        opt_seq.push_back(std::make_pair(0, ShiftGrid::go_finer));
        
        cached_sequence_ = std::move(opt_seq);
        cached_cycles_ = cycles; cached_levels_ = relax_mats_.size();
        cached_pre_ = pre_relax_; cached_post_ = post_relax_;
        solve_sequence(cached_sequence_, in, out);
    }

private:
    std::vector<std::pair<int, ShiftGrid>> cached_sequence_;
    int cached_cycles_=-1, cached_pre_=-1, cached_post_=-1;
    size_t cached_levels_=0;
    int pre_relax_ = 1;
    int post_relax_ = 1;

    inline void solve_sequence(const std::vector<std::pair<int, ShiftGrid>>& sequence, const Vector& in, Vector& out)
    {
        vector_traits<Vector>::hadamard_to(in, dmat_inv_[0], b_items_[0]);
        if (x_items_.front().rows() != out.rows())
        {
            x_items_[0].setZero();
            out.resize(x_items_.front().rows());
        }
        else
        {
            vector_traits<Vector>::copy_data(x_items_.front(), out);
        }

        int stage = 0;
        for (auto opt : sequence)
        {
            if(stage != relax_mats_.size())
            {
                for (int k = 0; k < opt.first; k++)
                {
                    smooth(stage);
                }
                if(opt.second == ShiftGrid::go_coarser)
                {
                    matrix_traits<MatrixType>::multiply_add(a_mats_[stage], x_items_[stage], b_items_[stage], buf_vec_[stage], -1.0, -1.0);
                    matrix_traits<MatrixType>::multiply(restrict_mats_[stage], buf_vec_[stage], b_items_[stage + 1]);
                    vector_traits<Vector>::hadamard_inplace(b_items_[stage + 1], dmat_inv_[stage + 1]);
                    x_items_[stage + 1].setZero();
                    stage++;
                }
                else if(opt.second == ShiftGrid::go_finer)
                {
                    if(stage != 0)
                    {
                        matrix_traits<MatrixType>::multiply_add(interpolate_mats_[stage - 1], x_items_[stage], x_items_[stage - 1], x_items_[stage - 1], 1.0, 1.0);
                        stage--;
                    }
                    else
                    {
                        if (x_items_.front().rows() != out.rows()) out.resize(x_items_.front().rows());
                        vector_traits<Vector>::copy_data(out, x_items_.front());
                        return;
                    }
                }
            }
            else
            {
                if(opt.first != 0 || opt.second != ShiftGrid::go_finer)
                    throw std::runtime_error("invalid sequence in MultiGrid.");
                if constexpr (requires { inner_solver_.solve_to(b_items_.back(), x_items_.back()); }) {
                    inner_solver_.solve_to(b_items_.back(), x_items_.back());
                } else {
                    x_items_.back() = inner_solver_.solve(b_items_.back());
                }
                matrix_traits<MatrixType>::multiply_add(interpolate_mats_[stage - 1], x_items_[stage], x_items_[stage - 1], x_items_[stage - 1], 1.0, 1.0);
                stage--;
            }
        }
    }

    inline void smooth(int level)
    {
        // Jacobi must read the complete old iterate. Ping-pong buffers permit
        // a fused matrix/add kernel without overwriting a neighbour's input.
        matrix_traits<MatrixType>::multiply_add(relax_mats_[level], x_items_[level], b_items_[level], buf_vec_[level], 2.0/3, 1.0);
        std::swap(x_items_[level], buf_vec_[level]);
    }
};

}

}

}

}

#endif
