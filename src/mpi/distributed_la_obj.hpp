#ifndef PSUM_MPI_DISTRIBUTED_LA_OBJ_HPP
#define PSUM_MPI_DISTRIBUTED_LA_OBJ_HPP

#include "comm_wrapper.hpp"
#include <iostream>
#include <algorithm>
#include <Eigen/SparseCore>

// la_obj: linear algebra object, including vector and sparse matrix

namespace psum {

namespace mpi {

    template<typename CommType>
    struct struct_with_commor
    {
    protected:
        CommType commor;
    public:
        int my_id() const
        {
            return commor.my_id;
        }
        int num_process() const
        {
            return commor.num_process;
        }
        std::vector<int> getCommMatrix() const
        {
            return commor.getCommMatrix();
        }
    };

    template<typename containerType, typename dataType>
    struct distributed_vector: containerType, struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>
    {
    private:
        using struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>::commor;
        std::vector<std::size_t> ownership;
        std::vector<std::vector<std::size_t>> ownership_table;
        std::vector<std::size_t> hold_list;

    public:
        using containerType::size;
        using containerType::resize;
        using containerType::operator[];
        using struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>::my_id;
        using struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>::num_process;

        distributed_vector<containerType, dataType>& operator= (const containerType& in)
        {
            resize(in.size());
            for (std::size_t i = 0; i < in.size(); i++)
            {
                operator[](i) = in[i];
            }
            return *this;
        }

        void setOwner(const std::vector<std::size_t>& in)
        {
            ownership = in;
            ownership_table.clear();
            ownership_table.resize(commor.num_process);
            hold_list.clear();
            for (size_t i = 0; i < in.size(); i++)
            {
                if(ownership[i]<0||ownership[i]>=commor.num_process)
                    throw std::runtime_error("Error: invalid MPI rank in distributed_vector::setOwner().");
                ownership_table[ownership[i]].push_back(i);
                if(ownership[i]==commor.my_id) hold_list.push_back(i);
            }
            commor.init();
            commor.clearCommMatrix();
        }

        std::vector<std::size_t> getOwner() const
        {
            return ownership;
        }

        std::vector<std::size_t>::const_iterator Owning_begin() const
        {
            return hold_list.begin();
        }

        std::vector<std::size_t>::const_iterator Owning_end() const
        {
            return hold_list.end();
        }

        void sync()
        {
            if(commor.is_inited()==false) throw std::runtime_error("Error: 'setOwner' must be called before distributed_vector::sync().");
            if (ownership.size() != size()) throw std::runtime_error("Error: different data size and ownership size in distributed_vector::sync().");

            std::vector<dataType> data_as_vec;
            for (auto i: ownership_table[commor.my_id])
            {
                data_as_vec.push_back(operator[](i));
            }

    #pragma omp parallel for
            for (int i = 0; i < commor.num_process; i++)
                if (i != commor.my_id)
                    commor.send_buffer[i] = data_as_vec;
            commor.callMPI();

    #pragma omp parallel for
            for (int i = 0; i < commor.num_process; i++)
            {
                if (i != commor.my_id)
                {
                    for (std::size_t j = 0; j < ownership_table[i].size(); j++)
                    {
                        operator[](ownership_table[i][j]) = commor.receive_buffer[i][j];
                    }
                }
            }
        }

    };

    template <typename _Scalar>
    using distributed_EigenVector = distributed_vector<Eigen::VectorX<_Scalar>, _Scalar>;

    template<typename dataType>
    struct distributed_Exchanger: struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>
    {
    private:
        using struct_with_commor<psum::mpi::MPI_fixed_array_wrapper<dataType>>::commor;
        std::vector<std::vector<std::vector<std::size_t>>> sharelist_matrix;

    public:
        void setByNeedlist(const std::vector<std::size_t> &owner, const std::vector<std::vector<std::size_t>> &needlist)
        {
            std::vector<std::vector<std::size_t>> necessary_vecvec = needlist;
            for (auto &list : necessary_vecvec)
            {
                std::sort(list.begin(), list.end());
            }

            sharelist_matrix.clear();
            sharelist_matrix.resize(commor.num_process);
            for (auto &v : sharelist_matrix)
                v.resize(commor.num_process);

            for (int i = 0; i < commor.num_process; i++)
            {
                for (auto id : necessary_vecvec[i])
                {
                    if (owner[id] != i)
                    {
                        sharelist_matrix[owner[id]][i].push_back(id);
                    }
                }
            };
            commor.init();
            commor.clearCommMatrix();
        }

        template <typename _Scalar, int _Options>
        void setBySparsemat(const std::vector<std::size_t> &owner_i, const std::vector<std::size_t> &owner_o, const Eigen::SparseMatrix<_Scalar, _Options> &mat)
        {
            if (mat.cols() != owner_i.size() || mat.rows() != owner_o.size())
                throw std::runtime_error("Error: invalid sparse matrix input in function call 'get_NecessaryList'.");
            std::vector<std::vector<std::size_t>> nnzs; // row major
            nnzs.resize(mat.rows());
            for (int k = 0; k < mat.outerSize(); ++k)
            {
                for (typename Eigen::SparseMatrix<_Scalar, _Options>::InnerIterator it(mat, k); it; ++it)
                {
                    nnzs[it.row()].push_back(it.col());
                }
            }
            std::vector<std::vector<std::size_t>> needlist;
            needlist.resize(commor.num_process);
            for (std::size_t i = 0; i < nnzs.size(); i++)
            {
                needlist[owner_o[i]].insert(needlist[owner_o[i]].end(), nnzs[i].begin(), nnzs[i].end());
            }
            setByNeedlist(owner_i, needlist);
        }

        template <typename _Scalar, int _Options>
        void setBySparsemat(const std::vector<std::size_t> &owner, const Eigen::SparseMatrix<_Scalar, _Options> &mat)
        {
            setBySparsemat(owner, owner, mat);
        }

        template<typename T>
        void operator()(T& shared_array, int thread_num = 1)
        {
            if (commor.is_inited() == false)
                throw std::runtime_error("Error: 'setOwner' must be called before distributed_vector::share().");

            for (int i = 0; i < commor.num_process; i++)
            {
                if (i != commor.my_id)
                {
                    commor.send_buffer[i].resize(sharelist_matrix[commor.my_id][i].size());
    #pragma omp parallel for num_threads(thread_num)
                    for (std::size_t j = 0; j < sharelist_matrix[commor.my_id][i].size(); j++)
                        commor.send_buffer[i][j] = shared_array[sharelist_matrix[commor.my_id][i][j]];
                }
            }
            commor.callMPI();

            for (int i = 0; i < commor.num_process; i++)
            {
                if (i != commor.my_id)
                {
                    auto &shareList = sharelist_matrix[i][commor.my_id];
    #pragma omp parallel for num_threads(thread_num)
                    for (std::size_t j = 0; j < shareList.size(); j++)
                    {
                        shared_array[shareList[j]] = commor.receive_buffer[i][j];
                    }
                }
            }
        }

    };

    template<typename _Scalar, int _Options = 0>
    struct distributed_EigenSpMat
    {
    protected:
        Eigen::SparseMatrix<_Scalar, _Options> Amat;
        Eigen::SparseMatrix<_Scalar, _Options> Amat_local;
        std::vector<std::size_t> global_index;
        distributed_Exchanger<_Scalar> exchanger;
        Eigen::VectorX<_Scalar> tempVec;
    public:
        template<typename containerType>
        void set(const Eigen::SparseMatrix<_Scalar, _Options> &Matrix,
                const distributed_vector<containerType, _Scalar> &vec_i,
                const distributed_vector<containerType, _Scalar> &vec_o)
        {
            exchanger.setBySparsemat(vec_i.getOwner(), vec_o.getOwner(), Matrix);
            global_index.clear();
            auto o_owner = vec_o.getOwner();
            for (std::size_t i = 0; i < o_owner.size(); i++)
                if (o_owner[i] == exchanger.my_id())
                    global_index.push_back(i);
            Amat = Matrix;
            
            std::vector<std::vector<std::pair<std::size_t, _Scalar>>> nnzs; // row major
            nnzs.resize(Amat.rows());
            std::vector<Eigen::Triplet<_Scalar>> trips;
            for (int k = 0; k < Amat.outerSize(); ++k)
            {
                for (typename Eigen::SparseMatrix<_Scalar, _Options>::InnerIterator it(Amat, k); it; ++it)
                {
                    nnzs[it.row()].push_back(std::make_pair(it.col(), it.value()));
                }
            }

            for (std::size_t i = 0; i < global_index.size(); i++)
            {
                for(auto& ijv: nnzs[global_index[i]])
                    trips.push_back(Eigen::Triplet<_Scalar>(i, ijv.first, ijv.second));
            }
            Amat_local.resize(global_index.size(), Amat.cols());
            Amat_local.setFromTriplets(trips.begin(), trips.end());
            tempVec.resize(global_index.size());
        }

        void product(distributed_vector<Eigen::VectorX<_Scalar>, _Scalar> &vec_i,
                    distributed_vector<Eigen::VectorX<_Scalar>, _Scalar> &vec_o)
        {
            exchanger(vec_i);
            tempVec = Amat_local * vec_i;
            for (std::size_t i = 0; i < global_index.size(); i++)
                vec_o[global_index[i]] = tempVec[i];
        }

        // out = (A*in+b*c)*d
        void product_add(distributed_vector<Eigen::VectorX<_Scalar>, _Scalar> &vec_i,
                        distributed_vector<Eigen::VectorX<_Scalar>, _Scalar> &vec_b,
                        distributed_vector<Eigen::VectorX<_Scalar>, _Scalar> &vec_o, double c = 1.0,double d = 1.0)
        {
            exchanger(vec_i);
            tempVec = Amat_local * vec_i;
            for (std::size_t i = 0; i < global_index.size(); i++)
                vec_o[global_index[i]] = (tempVec[i] + vec_b[global_index[i]] * c) * d;
        }

        Eigen::MatrixXi getCommMat()
        {
            std::vector<int> mat = exchanger.getCommMatrix();
            Eigen::MatrixXi ans;
            ans.resize(exchanger.num_process(), exchanger.num_process());
            for (int i = 0; i < ans.size(); i++)
            {
                ans(i / exchanger.num_process(), i % exchanger.num_process()) = mat[i];
            }
            return ans;
        }
    };

}

}
#endif