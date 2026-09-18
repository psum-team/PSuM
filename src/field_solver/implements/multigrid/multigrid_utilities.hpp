#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_UTILITIES_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_UTILITIES_HPP

#include <Eigen/SparseCore>
#include <unordered_set>
#include <unordered_map>

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {

inline Eigen::SparseMatrix<double> get_restriction_matrix_2d(int I, int J)
{
    int Ih = (I / 2 + I % 2);
    int Jh = (J / 2 + J % 2);
    Eigen::SparseMatrix<double> rmat(Ih * Jh, I * J);
    std::vector<Eigen::Triplet<double>> trips;
    for (int i = 0; i < Ih; i++)
    {
        for (int j = 0; j < Jh; j++)
        {
            int row = i * Jh + j;
            int col_main = (2 * i) * J + (2 * j);

            std::unordered_set<int> neighbors_1, neighbors_2;
            neighbors_1.insert({col_main + 1, col_main - 1, col_main + J, col_main - J});
            neighbors_2.insert({col_main + 1 + J, col_main + 1 - J,
                                col_main - 1 + J, col_main - 1 - J});

            std::unordered_map<int, double> weights;
            for (int il = 2 * i - 1; il <= 2 * i + 1; il++)
            {
                for (int jl = 2 * j - 1; jl <= 2 * j + 1; jl++)
                {
                    int col = il * J + jl;
                    if (il >= 0 && il < I && jl >= 0 && jl < J) {}
                    else
                    {
                        if (il < 0)
                            col += J * 2;
                        if (il >= I)
                            col -= J * 2;
                        if (jl < 0)
                            col += 2;
                        if (jl >= J)
                            col -= 2;
                    }
                    if (weights.find(col) == weights.end())
                        weights[col] = 0;
                    if (col == col_main)
                        weights[col] = 1.0 / 4;
                    else
                    {
                        if (neighbors_1.find(col) != neighbors_1.end())
                            weights[col] += 1.0 / 8;
                        if (neighbors_2.find(col) != neighbors_2.end())
                            weights[col] += 1.0 / 16;
                    }
                }
            }
            for (auto it : weights)
            {
                trips.push_back(Eigen::Triplet<double>(row, it.first, it.second));
            }
        }
    }
    rmat.setFromTriplets(trips.begin(), trips.end());
    return rmat;
}

inline Eigen::SparseMatrix<double> get_restriction_matrix_3d(int I, int J, int K)
{
    if(K <= 1)
        return get_restriction_matrix_2d(I, J);
    int Ih = (I / 2 + I % 2);
    int Jh = (J / 2 + J % 2);
    int Kh = (K / 2 + K % 2);
    Eigen::SparseMatrix<double> rmat(Ih * Jh * Kh, I * J * K);
    std::vector<Eigen::Triplet<double>> trips;
    for (int i = 0; i < Ih; i++)
    {
        for (int j = 0; j < Jh; j++)
        {
            for (int k = 0; k < Kh; k++)
            {
                int row = i * Jh * Kh + j * Kh + k;
                int col_main = (2 * i) * J * K + (2 * j) * K + (2 * k);

                std::unordered_set<int> neighbors_1, neighbors_2, neighbors_3;
                neighbors_1.insert({col_main + 1, col_main - 1, col_main + K, col_main - K, col_main + J * K, col_main - J * K});
                neighbors_2.insert({col_main + 1 + K, col_main + 1 - K, col_main + 1 + J * K, col_main + 1 - J * K,
                                    col_main - 1 + K, col_main - 1 - K, col_main - 1 + J * K, col_main - 1 - J * K,
                                    col_main + K + J * K, col_main + K - J * K, col_main - K + J * K, col_main - K - J * K});
                neighbors_3.insert({col_main + 1 + K + J * K, col_main + 1 + K - J * K, col_main + 1 - K + J * K, col_main + 1 - K - J * K,
                                    col_main - 1 + K + J * K, col_main - 1 + K - J * K, col_main - 1 - K + J * K, col_main - 1 - K - J * K});

                std::unordered_map<int, double> weights;
                for (int il = 2 * i - 1; il <= 2 * i + 1; il++)
                {
                    for (int jl = 2 * j - 1; jl <= 2 * j + 1; jl++)
                    {
                        for (int kl = 2 * k - 1; kl <= 2 * k + 1; kl++)
                        {
                            int col = il * J * K + jl * K + kl;
                            if (il >= 0 && il < I && jl >= 0 && jl < J && kl >= 0 && kl < K) {}
                            else
                            {
                                if(il < 0) col += J * K * 2;
                                if(il >= I) col -= J * K * 2;
                                if(jl < 0) col += K * 2;
                                if(jl >= J) col -= K * 2;
                                if(kl < 0) col += 2;
                                if(kl >= K) col -= 2;
                            }
                            if (weights.find(col) == weights.end())
                                weights[col] = 0;
                            if(col == col_main)
                                weights[col] = 1.0 / 8;
                            else
                            {
                                if (neighbors_1.find(col) != neighbors_1.end())
                                    weights[col] += 1.0 / 16;
                                if (neighbors_2.find(col) != neighbors_2.end())
                                    weights[col] += 1.0 / 32;
                                if (neighbors_3.find(col) != neighbors_3.end())
                                    weights[col] += 1.0 / 64;
                            }
                        }
                    }
                }
                for(auto it: weights)
                {
                    trips.push_back(Eigen::Triplet<double>(row, it.first, it.second));
                }
            }
        }
    }
    rmat.setFromTriplets(trips.begin(), trips.end());
    return rmat;
}

inline std::vector<Eigen::SparseMatrix<double>> get_restriction_matrices_3d(int I, int J, int K, int depth)
{
    std::vector<Eigen::SparseMatrix<double>> ans;
    int Icur = I;
    int Jcur = J;
    int Kcur = K;
    for (int i = 0; i < depth; i++)
    {
        ans.push_back(get_restriction_matrix_3d(Icur, Jcur, Kcur));
        Icur = (Icur / 2 + Icur % 2);
        Jcur = (Jcur / 2 + Jcur % 2);
        Kcur = (Kcur / 2 + Kcur % 2);
    }
    return ans;
}

inline std::vector<Eigen::SparseMatrix<double>> get_restriction_matrices_2d(int I, int J, int depth)
{
    std::vector<Eigen::SparseMatrix<double>> ans;
    int Icur = I;
    int Jcur = J;
    for (int i = 0; i < depth; i++)
    {
        ans.push_back(get_restriction_matrix_2d(Icur, Jcur));
        Icur = (Icur / 2 + Icur % 2);
        Jcur = (Jcur / 2 + Jcur % 2);
    }
    return ans;
}

}

}

}

}

#endif