#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Sparse>

#include "../../src/field_solver/ghost_element_manager.hpp"

using SpMat = Eigen::SparseMatrix<double>;
using Triplet = Eigen::Triplet<double>;

enum direction {
    left, right, bottom, top
};

enum BCType {
    Neumann, Robin
};

struct BC {
    size_t element;
    size_t ghost_element;
    direction dir;
    BCType type;
    double a,b,c;
};

int main() {

    const size_t Nx = 6;
    const size_t Ny = 5;

    const double dx = 0.2;
    const double dy = 0.35;

    const size_t N = Nx*Ny;

    auto idx = [&](size_t i,size_t j){
        return j*Nx + i;
    };

    psum::field_solver::ghost_element_manager ghost_mgr(N);

    std::vector<BC> bcs;
    std::map<std::pair<size_t,direction>, size_t> bc_lookup;

    // -----------------------------
    // STEP 1: create BC list
    // -----------------------------

    for(size_t j=0;j<Ny;j++)
    {
        size_t e = idx(0,j);
        bcs.push_back({e,size_t(-1),left,Neumann,0,0,1.0}); // left Neumann
    }

    for(size_t j=0;j<Ny;j++)
    {
        size_t e = idx(Nx-1,j);
        bcs.push_back({e,size_t(-1),right,Robin,1.0,2.0,0.5}); // right Robin
    }

    for(size_t i=0;i<Nx;i++)
    {
        size_t e = idx(i,0);
        bcs.push_back({e,size_t(-1),bottom,Neumann,0,0,-0.3}); // bottom Neumann
    }

    for(size_t i=0;i<Nx;i++)
    {
        size_t e = idx(i,Ny-1);
        bcs.push_back({e,size_t(-1),top,Robin,2.0,1.0,1.0}); // top Robin
    }

    for(size_t k=0;k<bcs.size();k++)
        bc_lookup[{bcs[k].element,bcs[k].dir}] = k;

    // -----------------------------
    // STEP 2: assemble interior eq
    // -----------------------------

    std::vector<Triplet> A_trip;
    std::vector<Triplet> A_global_trip;

    for(int j=0;j<Ny;j++)
    for(int i=0;i<Nx;i++)
    {
        size_t id = idx(i,j);

        std::map<size_t,double> terms;

        double cx = 1.0/(dx*dx);
        double cy = 1.0/(dy*dy);

        terms[id] = -2*cx -2*cy;

        auto handle_neighbor = [&](int ni,int nj,direction dir)
        {
            if(ni>=0 && ni<Nx && nj>=0 && nj<Ny) {
                size_t nid = idx(ni,nj);
                if(dir<2) terms[nid]+=cx;
                else terms[nid]+=cy;
            } else {
                auto it = bc_lookup.find({id, dir});
                if (it == bc_lookup.end())
                    throw std::runtime_error("BC missing");

                BC &bc = bcs[it->second];

                if(bc.ghost_element==size_t(-1))
                    bc.ghost_element = ghost_mgr.new_ghost_element();

                if(dir<2) terms[bc.ghost_element]+=cx;
                else terms[bc.ghost_element]+=cy;
            }
        };

        handle_neighbor(i-1,j,left);
        handle_neighbor(i+1,j,right);
        handle_neighbor(i,j-1,bottom);
        handle_neighbor(i,j+1,top);

        psum::field_solver::local_equation eq(terms);

        ghost_mgr.extract_ghost_contribute(id,eq);

        for(auto &t:terms) {
            if(t.first < N)
                A_trip.emplace_back(id,t.first,t.second);
            A_global_trip.emplace_back(id,t.first,t.second);
        }
    }

    SpMat A(N,N);
    A.setFromTriplets(A_trip.begin(),A_trip.end());

    // -----------------------------
    // STEP 3: build BC equations
    // -----------------------------

    for(size_t k=0;k<bcs.size();k++)
    {
        auto& bc = bcs[k];
        size_t i = bc.element;
        size_t g = bc.ghost_element;

        std::map<size_t,double> terms;

        double h = (bc.dir<2)?dx:dy;

        if(bc.type==Neumann)
        {
            terms[i] = -1;
            terms[g] = 1;
            bc.c *= h;
        }
        else if(bc.type==Robin)
        {
            terms[i] = bc.a - bc.b/h;
            terms[g] = bc.b/h;
        }

        ghost_mgr.set_ghost_equation(
            psum::field_solver::local_equation(terms)
        );

        for(auto &t:terms)
            A_global_trip.emplace_back(N + k,t.first,t.second);
    }

    // -----------------------------
    // eliminate ghosts
    // -----------------------------

    auto [Aelim,M] = ghost_mgr.get_elimination_system(A);

    std::cout<<"A size = "<<A.rows()<<" "<<A.cols()<<"\n";
    std::cout<<"Aelim nnz = "<<Aelim.nonZeros()<<"\n";
    std::cout<<"M nnz = "<<M.nonZeros()<<"\n";

    // -------------------------------------------------
    // Build full system for verification
    // -------------------------------------------------

    size_t Ng = ghost_mgr.get_n_ghost_equations();

    SpMat A_global(N+Ng,N+Ng);

    A_global.setFromTriplets(A_global_trip.begin(),A_global_trip.end());

    // -------------------------------------------------
    // build RHS
    // -------------------------------------------------

    // elimination RHS
    Eigen::VectorXd rhs_elim(N);
    rhs_elim.setRandom();

    Eigen::VectorXd cvec(Ng);
    for (size_t k = 0; k < bcs.size(); k++) {
        cvec[k] = bcs[k].c;
    }

    Eigen::VectorXd rhs_full(N+Ng);
    rhs_full.head(N) = rhs_elim;
    rhs_full.tail(Ng) = cvec;

    rhs_elim += M * cvec;

    // -------------------------------------------------
    // solve systems
    // -------------------------------------------------

    Eigen::SparseLU<SpMat> solver;

    solver.compute(A_global);
    Eigen::VectorXd sol_full = solver.solve(rhs_full);

    solver.compute(Aelim);
    Eigen::VectorXd sol_elim = solver.solve(rhs_elim);

    // -------------------------------------------------
    // compare
    // -------------------------------------------------

    Eigen::VectorXd x_full = sol_full.head(N);

    double err = (x_full - sol_elim).norm() / x_full.norm();

    // Eigen::MatrixXd ans;
    // ans.resize(N, 3);
    // ans.col(0) = x_full;
    // ans.col(1) = sol_elim;
    // ans.col(2) = x_full - sol_elim;
    // std::cout << ans << "\n";

    std::cout<<"relative error = "<<err<<"\n";

}