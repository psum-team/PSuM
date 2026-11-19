#ifndef PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_UTILITIES_HPP
#define PSUM_FIELD_SOLVER_IMPLEMENTS_MULTIGRID_UTILITIES_HPP

#include <Eigen/SparseCore>
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace psum {

namespace field_solver {

namespace implements {

namespace multigrid {


// P interpolates values. R is its lumped-volume adjoint, normalized to
// preserve constants. They are built together; P is NOT a scaled R transpose.
struct structured_transfers {
    std::vector<Eigen::SparseMatrix<double>> restriction;
    std::vector<Eigen::SparseMatrix<double>> interpolation;
};

namespace transfer_detail {
using Axis = std::vector<double>;
inline Axis coarsen(const Axis& fine) {
    Axis coarse;
    for (size_t i = 0; i < fine.size(); i += 2) coarse.push_back(fine[i]);
    if (coarse.back() != fine.back()) coarse.push_back(fine.back());
    return coarse;
}
inline std::vector<double> volumes(const Axis& x) {
    std::vector<double> w(x.size(), 1.0);
    if (x.size() == 1) return w;
    for (size_t i = 0; i < x.size(); ++i) {
        w[i] = ((i ? x[i] - x[i-1] : 0.0)
              + (i+1 < x.size() ? x[i+1] - x[i] : 0.0)) * 0.5;
    }
    return w;
}
using Weights = std::vector<std::pair<int,double>>;
inline Weights interpolate(double x, const Axis& coarse) {
    auto it = std::lower_bound(coarse.begin(), coarse.end(), x);
    if (it == coarse.end()) throw std::runtime_error("multigrid: point outside coarse grid");
    const int hi = static_cast<int>(it - coarse.begin());
    if (*it == x) return {{hi, 1.0}};
    if (hi == 0) throw std::runtime_error("multigrid: point outside coarse grid");
    const double t = (x - coarse[hi-1]) / (coarse[hi] - coarse[hi-1]);
    return {{hi-1, 1.0-t}, {hi, t}};
}
}

inline structured_transfers get_structured_transfers(int I, int J, int K, int depth) {
    if (I < 1 || J < 1 || K < 1 || depth < 1)
        throw std::invalid_argument("multigrid: positive grid dimensions and depth required");
    using namespace transfer_detail;
    std::array<Axis,3> axes;
    const std::array<int,3> sizes{I,J,K};
    for (int d=0; d<3; ++d) {
        axes[d].resize(sizes[d]);
        for (int i=0; i<sizes[d]; ++i) axes[d][i]=i;
    }
    structured_transfers result;
    for (int level=0; level<depth; ++level) {
        std::array<Axis,3> coarse;
        std::array<std::vector<double>,3> volume;
        std::array<std::vector<Weights>,3> weights;
        for (int d=0; d<3; ++d) {
            coarse[d]=coarsen(axes[d]);
            volume[d]=volumes(axes[d]);
            for (double x:axes[d]) weights[d].push_back(interpolate(x,coarse[d]));
        }
        const int ni=axes[0].size(), nj=axes[1].size(), nk=axes[2].size();
        const int ci=coarse[0].size(), cj=coarse[1].size(), ck=coarse[2].size();
        // Two endpoint nodes cannot be coarsened further without losing one.
        if (ni==ci && nj==cj && nk==ck) break;
        Eigen::SparseMatrix<double> P(ni*nj*nk,ci*cj*ck);
        std::vector<Eigen::Triplet<double>> entries;
        Eigen::VectorXd fine_volume(ni*nj*nk);
        for (int i=0;i<ni;++i) for (int j=0;j<nj;++j) for (int k=0;k<nk;++k) {
            const int row=(i*nj+j)*nk+k;
            fine_volume[row]=volume[0][i]*volume[1][j]*volume[2][k];
            for (auto [a,wa]:weights[0][i])
                for (auto [b,wb]:weights[1][j])
                    for (auto [c,wc]:weights[2][k])
                        entries.emplace_back(row,(a*cj+b)*ck+c,wa*wb*wc);
        }
        P.setFromTriplets(entries.begin(),entries.end());
        Eigen::VectorXd coarse_volume=P.transpose()*fine_volume;
        Eigen::SparseMatrix<double> R=P.transpose();
        for (int c=0;c<R.outerSize();++c)
            for (Eigen::SparseMatrix<double>::InnerIterator it(R,c);it;++it)
                it.valueRef()*=fine_volume[it.col()]/coarse_volume[it.row()];
        result.interpolation.push_back(std::move(P));
        result.restriction.push_back(std::move(R));
        // Retain physical positions: the last interval can be shorter.
        axes=std::move(coarse);
    }
    if (result.restriction.empty())
        throw std::invalid_argument("multigrid: grid cannot be coarsened while retaining endpoints");
    return result;
}

// Compatibility accessors. New structured callers should request the pair.
inline Eigen::SparseMatrix<double> get_restriction_matrix_2d(int I,int J) {
    return get_structured_transfers(I,J,1,1).restriction.front();
}
inline Eigen::SparseMatrix<double> get_restriction_matrix_3d(int I,int J,int K) {
    return get_structured_transfers(I,J,K,1).restriction.front();
}
inline std::vector<Eigen::SparseMatrix<double>> get_restriction_matrices_2d(int I,int J,int depth) {
    return get_structured_transfers(I,J,1,depth).restriction;
}
inline std::vector<Eigen::SparseMatrix<double>> get_restriction_matrices_3d(int I,int J,int K,int depth) {
    return get_structured_transfers(I,J,K,depth).restriction;
}

}

}

}

}

#endif