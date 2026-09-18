#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_SPARSE_MATRIX_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_CONV_SPARSE_MATRIX_HPP

// This file provides ConvSparseMatrix for accelerated matrix-vector operations
// File name: convSparseMatrix.hpp (lowercase 'c', 'S', 'M')

#include <vector>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <map>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <sstream>
#include "sparse_linear_trait.hpp"

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

template<typename T>
struct ConvSparseMatrix {
    std::vector<std::pair<std::vector<std::pair<T, int>>, std::vector<int>>> convCmpt;
    std::vector<std::vector<int>> convs_Anchor;
    bool Anchor_at_max = false;
    Eigen::SparseMatrix<T, Eigen::RowMajor> restMat;
    Eigen::SparseMatrix<T, Eigen::RowMajor> original_mat;
    mutable Eigen::VectorX<T> pruned_output_vec;
    std::vector<int> non_all_zero_rows;
    int MatRows;
    int thread_num = 0;

    template<int Eigen_T2, typename Eigen_T3>
    void set(const Eigen::SparseMatrix<T, Eigen_T2, Eigen_T3>& _mat)
    {
        original_mat = _mat;
        MatRows = _mat.rows();
        Anchor_at_max = false;
        convCmpt.clear();
        convs_Anchor.clear();
        non_all_zero_rows.clear();
        restMat = Eigen::SparseMatrix<T, Eigen::RowMajor>(MatRows, MatRows);
    }

    template<int Eigen_T2, typename Eigen_T3>
    void set(const Eigen::SparseMatrix<T, Eigen_T2, Eigen_T3>& _mat, int num_conv_mask)
    {
        using namespace std;
        typedef Eigen::SparseMatrix<T, Eigen::RowMajor> MatType;
        typedef std::vector<std::pair<T, int>> coeff_and_index;
        MatType mat = _mat;
        original_mat = mat;
        int try_rows = num_conv_mask * 100;
        if (mat.cols() != mat.rows()) Anchor_at_max = true;
        std::map<coeff_and_index, int> convTs;

        if (mat.rows() != 0)
            for (int i = 0; i < try_rows; i++)
            {
                int rand_row = rand() % (mat.rows());
                coeff_and_index this_row;
                for (typename MatType::InnerIterator it(mat, rand_row); it; ++it)
                {
                    this_row.push_back(std::make_pair(it.value(), it.col() - it.row()));
                }
                if (Anchor_at_max)
                {
                    int anchor = 0;
                    sort(this_row.begin(), this_row.end(),
                         [](pair<T, int>& a, pair<T, int>& b) { return abs(a.first) < abs(b.first) || ((abs(a.first) == abs(b.first)) && (a.second > b.second)); });
                    if (this_row.size() > 0) anchor = this_row.back().second + i;
                    for (auto& p : this_row)
                    {
                        p.second += i - anchor;
                    }
                }
                sort(this_row.begin(), this_row.end(),
                     [](pair<T, int>& a, pair<T, int>& b) { return a.second < b.second; });

                if (convTs.find(this_row) == convTs.end())
                    convTs[this_row] = 0;
                else
                    convTs[this_row]++;
            }
        vector<pair<coeff_and_index, int>> convTs_vec;
        convTs_vec.insert(convTs_vec.end(), convTs.begin(), convTs.end());
        sort(convTs_vec.begin(), convTs_vec.end(),
             [](pair<coeff_and_index, int>& a, pair<coeff_and_index, int>& b)
             { return a.second > b.second; });

        std::map<coeff_and_index, vector<int>> chosed;
        std::map<coeff_and_index, vector<int>> chosed_anchor;
        for (int i = 0; i < num_conv_mask && i < convTs_vec.size(); i++)
        {
            chosed[convTs_vec[i].first] = vector<int>();
            chosed_anchor[convTs_vec[i].first] = vector<int>();
        }

        for (int i = 0; i < mat.outerSize(); i++)
        {
            coeff_and_index this_row;
            for (typename MatType::InnerIterator it(mat, i); it; ++it)
            {
                this_row.push_back(std::make_pair(it.value(), it.col() - it.row()));
                if (std::isnan(it.value()))
                {
                    std::stringstream ss;
                    ss << "\trow=" << it.row() << ", col=" << it.col() << std::endl;
                    throw std::runtime_error("NaN Error occurred in ConvSparseMatrix constructor." + ss.str());
                }
            }
            int anchor = 0;
            if (Anchor_at_max)
            {
                sort(this_row.begin(), this_row.end(),
                     [](pair<T, int>& a, pair<T, int>& b) { return abs(a.first) < abs(b.first) || ((abs(a.first) == abs(b.first)) && (a.second > b.second)); });
                if (this_row.size() > 0) anchor = this_row.back().second + i;
                for (auto& p : this_row)
                {
                    p.second += i - anchor;
                }
            }
            sort(this_row.begin(), this_row.end(),
                 [](pair<T, int>& a, pair<T, int>& b) { return a.second < b.second; });
            if (chosed.find(this_row) != chosed.end())
            {
                chosed[this_row].push_back(i);
                if (Anchor_at_max) chosed_anchor[this_row].push_back(anchor);
            }
        }
        convCmpt.clear();
        for (auto& i : chosed)
            convCmpt.push_back(make_pair(i.first, i.second));
        if (Anchor_at_max)
            for (auto& i : chosed)
                convs_Anchor.push_back(chosed_anchor[i.first]);

        restMat = mat;
        for (auto& rows : convCmpt)
        {
            for (auto row : rows.second)
            {
                for (typename MatType::InnerIterator it(restMat, row); it; ++it)
                {
                    it.valueRef() *= 0;
                }
            }
        }
        restMat = (restMat).pruned();
        MatRows = restMat.rows();

        non_all_zero_rows.clear();
        for (int i = 0; i < restMat.outerSize(); i++)
        {
            int counter = 0;
            for (typename MatType::InnerIterator it(restMat, i); it; ++it)
            {
                counter++;
            }
            if (counter != 0)
                non_all_zero_rows.push_back(i);
        }
        if (non_all_zero_rows.size() > 0)
        {
            pruned_output_vec.resize(non_all_zero_rows.size());
            Eigen::SparseMatrix<T> reshapeMat;
            reshapeMat.resize(non_all_zero_rows.size(), restMat.rows());
            vector<Eigen::Triplet<T>> trips;
            for (int i = 0; i < non_all_zero_rows.size(); i++)
            {
                trips.push_back(Eigen::Triplet<T>(i, non_all_zero_rows[i], 1));
            }
            reshapeMat.setFromTriplets(trips.begin(), trips.end());
            restMat = reshapeMat * restMat;
        }
    }

    void set_thread_num(int num) {
        thread_num = num;
    }

    Eigen::VectorX<T> operator*(const Eigen::VectorX<T>& in) const
    {
        if (convCmpt.empty())
        {
            return original_mat * in;
        }

        Eigen::VectorX<T> ans;
        ans.resize(MatRows);
        if (non_all_zero_rows.size() > 0)
        {
            pruned_output_vec = restMat * in;
            for (int i = 0; i < non_all_zero_rows.size(); i++)
            {
                ans[non_all_zero_rows[i]] = pruned_output_vec[i];
            }
        }
        for (int conv_id = 0; conv_id < convCmpt.size(); conv_id++)
        {
            const auto& conv = convCmpt[conv_id];
            const std::vector<std::pair<T, int>>& coeff_and_index = conv.first;
            const std::vector<int>& rows = conv.second;
            if (!Anchor_at_max)
            {
#pragma omp parallel for num_threads(thread_num)
                for (int row : rows)
                {
                    double v = 0;
                    for (auto p : coeff_and_index)
                    {
                        v += p.first * (in[p.second + row]);
                    }
                    ans[row] = v;
                }
            }
            else
            {
#pragma omp parallel for num_threads(thread_num)
                for (int i = 0; i < rows.size(); i++)
                {
                    double v = 0;
                    for (auto p : coeff_and_index)
                    {
                        v += p.first * (in[p.second + convs_Anchor[conv_id][i]]);
                    }
                    ans[rows[i]] = v;
                }
            }
        }
        return ans;
    }

    Eigen::VectorX<T> multiply_add(const Eigen::VectorX<T>& in, const Eigen::VectorX<T>& b, T b_coeff = 1, T ans_coeff = 1) const
    {
        if (convCmpt.empty())
        {
            return (original_mat * in + b_coeff * b) * ans_coeff;
        }

        Eigen::VectorX<T> ans;
        ans.resize(MatRows);
        if (non_all_zero_rows.size() > 0)
        {
            pruned_output_vec = restMat * in;
            for (int i = 0; i < non_all_zero_rows.size(); i++)
            {
                ans[non_all_zero_rows[i]] = (pruned_output_vec[i] + b[non_all_zero_rows[i]] * b_coeff) * ans_coeff;
            }
        }
        for (int conv_id = 0; conv_id < convCmpt.size(); conv_id++)
        {
            const auto& conv = convCmpt[conv_id];
            const std::vector<std::pair<T, int>>& coeff_and_index = conv.first;
            const std::vector<int>& rows = conv.second;
            if (!Anchor_at_max)
            {
#pragma omp parallel for num_threads(thread_num)
                for (int row : rows)
                {
                    double v = b[row] * b_coeff;
                    for (auto p : coeff_and_index)
                    {
                        v += p.first * (in[p.second + row]);
                    }
                    ans[row] = v * ans_coeff;
                }
            }
            else
            {
#pragma omp parallel for num_threads(thread_num)
                for (int i = 0; i < rows.size(); i++)
                {
                    double v = b[rows[i]] * b_coeff;
                    for (auto p : coeff_and_index)
                    {
                        v += p.first * (in[p.second + convs_Anchor[conv_id][i]]);
                    }
                    ans[rows[i]] = v * ans_coeff;
                }
            }
        }
        return ans;
    }

    int rows() const { return MatRows; }
    int cols() const { return MatRows; }
    int outerSize() const { return original_mat.outerSize(); }

    T coeff(int row, int col) const {
        return original_mat.coeff(row, col);
    }

    using InnerIterator = typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator;

    void multiply(const Eigen::VectorX<T>& in, Eigen::VectorX<T>& out) const {
        out = (*this) * in;
    }

    void multiply_add(const Eigen::VectorX<T>& in, const Eigen::VectorX<T>& b,
                      Eigen::VectorX<T>& out, T b_coeff = 1, T ans_coeff = 1) const {
        Eigen::VectorX<T> tmp = (*this) * in;
        out = (tmp * ans_coeff + b * (b_coeff * ans_coeff));
    }
};

template <>
struct matrix_traits<ConvSparseMatrix<double>> {
    using Scalar = double;
    using Vector = Eigen::VectorXd;

    static void multiply(const ConvSparseMatrix<double>& mat, const Vector& in, Vector& out) {
        mat.multiply(in, out);
    }

    static void multiply_add(const ConvSparseMatrix<double>& mat, const Vector& x,
                             const Vector& b, Vector& out, double b_coeff = 1.0, double ans_coeff = 1.0) {
        mat.multiply_add(x, b, out, b_coeff, ans_coeff);
    }

    static void set(ConvSparseMatrix<double>& dest, const Eigen::SparseMatrix<double>& src) {
        dest.set(src);
    }
};

}

}

}

}

#endif
