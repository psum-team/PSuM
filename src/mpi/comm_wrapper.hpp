#ifndef PSUM_MPI_COMM_WRAPPER_HPP
#define PSUM_MPI_COMM_WRAPPER_HPP

#include <vector>
#include <set>
#include <map>
#include <stdexcept>
#include "interface.h"

namespace psum {

namespace mpi {

    /*
    set send_buffer[i]
    use callMPI()
    and get in receive_buffer[j]
    */
    template <typename dataType>
    class MPI_array_wrapper{
    private:
        bool inited = false;
        std::vector<psum_mpi_request_t*> request_send;
        std::vector<psum_mpi_request_t*> request_receive;

        int rc_check(int rc) const {
            if (rc != PSUM_MPI_SUCCESS)
                throw std::runtime_error("psum_mpi error in MPI_array_wrapper::callMPI");
            return rc;
        }

    public:
        std::vector<std::vector<dataType>> send_buffer;
        std::vector<std::vector<dataType>> receive_buffer;

        int my_id;
        int num_process;

        /* matrix layout:
        i -> j
        idx: i * num_process + j

            | 0 -> 1 | 0 -> 2 | 0 -> 3 | 0 -> ..| ...
        1 -> 0 |        | 1 -> 2 | 1 -> 3 | 1 -> ..| ...
        2 -> 0 | 2 -> 1 |        | 2 -> 3 | 2 -> ..| ...
        3 -> 0 | 3 -> 1 | 3 -> 2 |        | 3 -> ..| ...
        4 -> 0 | 4 -> 1 | 4 -> 2 | 4 -> 3 |        | ...
        ...    | ...    | ...    | ...    | ...    | ...

        */
        std::vector<int> mpi_size_matrix_local;
        std::vector<int> mpi_size_matrix_all;

        MPI_array_wrapper()
        {
            rc_check(psum_mpi_comm_rank(&my_id));
            rc_check(psum_mpi_comm_size(&num_process));
        }

        void init(){
            mpi_size_matrix_local.resize(num_process * num_process);
            mpi_size_matrix_all.resize(num_process * num_process);
            send_buffer.resize(num_process);
            receive_buffer.resize(num_process);
            inited = true;
        }

        std::vector<int> getCommMatrix() const
        {
            return mpi_size_matrix_all;
        }

        void callMPI(){

            if(inited == false) init();

            //==================== calculate the sending size matrix====================
            for(int i = 0; i < num_process; i++){
                mpi_size_matrix_local[my_id * num_process + i] = (int)send_buffer[i].size();
            }
            rc_check(psum_mpi_allreduce(
                mpi_size_matrix_local.data(),
                mpi_size_matrix_all.data(),
                num_process * num_process,
                PSUM_MPI_INT,
                PSUM_MPI_SUM
            ));
            for(int i = 0; i < num_process; i++){
                receive_buffer[i].resize(mpi_size_matrix_all[i * num_process + my_id]);
            }

            //reserve request for non-empty send
            int counter_send = 0;
            for(auto& i: send_buffer)
                if(i.size()!=0) counter_send++;
            request_send.resize(counter_send);

            int counter_receive = 0;
            for(auto& i: receive_buffer)
                if(i.size()!=0) counter_receive++;
            request_receive.resize(counter_receive);

            int id_send = 0;
            int id_receive = 0;
            for(int i = 0; i < num_process; i++){
                //non-blocking send
                if(mpi_size_matrix_all[my_id * num_process + i] != 0){
                    rc_check(psum_mpi_isend(
                        send_buffer[i].data(),
                        (int)(mpi_size_matrix_all[my_id * num_process + i] * sizeof(dataType)),
                        PSUM_MPI_BYTE,
                        i,
                        0,
                        &request_send[id_send]
                    ));
                    id_send++;
                }
                // non-blocking receive
                if(mpi_size_matrix_all[i * num_process + my_id] != 0)
                {
                    rc_check(psum_mpi_irecv(
                        receive_buffer[i].data(),
                        (int)(mpi_size_matrix_all[i * num_process + my_id] * sizeof(dataType)),
                        PSUM_MPI_BYTE,
                        i,
                        0,
                        &request_receive[id_receive]
                    ));
                    id_receive++;
                }
            }

            //wait for all receive to complete
            for(int i = 0; i < (int)request_receive.size(); i++){
                int receive_index = 0;
                rc_check(psum_mpi_waitany(
                    (int)request_receive.size(),
                    request_receive.data(),
                    &receive_index,
                    PSUM_MPI_STATUS_IGNORE
                ));
            }

            // callMPI is synchronous operation
            rc_check(psum_mpi_waitall((int)request_send.size(), request_send.data()));

            // release all request handles
            for(auto& r : request_receive) rc_check(psum_mpi_request_free(&r));
            for(auto& r : request_send)    rc_check(psum_mpi_request_free(&r));

            //clear send_buffer
            for(int i = 0; i < num_process; i++){
                send_buffer[i].resize(0);
            }
        }

    };


    /*
    specialized version for fixed size array, which is more efficient than the general version
    */
    template <typename dataType>
    class MPI_fixed_array_wrapper{
    private:
        bool inited = false;
        std::vector<int> mpi_size_matrix_all;
        std::vector<psum_mpi_request_t*> request_send;
        std::vector<psum_mpi_request_t*> request_receive;

        int rc_check(int rc) const {
            if (rc != PSUM_MPI_SUCCESS)
                throw std::runtime_error("psum_mpi error in MPI_fixed_array_wrapper::callMPI");
            return rc;
        }
    public:

        std::vector<std::vector<dataType>> send_buffer;
        std::vector<std::vector<dataType>> receive_buffer;

        int my_id;
        int num_process;


        MPI_fixed_array_wrapper()
        {
            rc_check(psum_mpi_comm_rank(&my_id));
            rc_check(psum_mpi_comm_size(&num_process));
        }

    //input_matrix[i][j] := size of data to send from rank i to rank j
        void init(const std::vector<std::vector<int>>& input_matrix){

            //check input_matrix size
            if(input_matrix.size() != (size_t)num_process)
                throw std::runtime_error("invalid input size");
            for(auto& i: input_matrix)
                if(i.size() != (size_t)num_process)
                    throw std::runtime_error("invalid input size");

            mpi_size_matrix_all.resize(num_process * num_process);
            for(int i = 0; i < num_process; i++)
                for(int j = 0; j < num_process; j++)
                    mpi_size_matrix_all[i * num_process + j] = input_matrix[i][j];

            send_buffer.resize(num_process);
            receive_buffer.resize(num_process);
            inited = true;

            //reserve send_buffer and receive_buffer according to the size matrix
            for(int i = 0; i < num_process; i++){
                send_buffer[i].resize(mpi_size_matrix_all[my_id * num_process + i]);
                receive_buffer[i].resize(mpi_size_matrix_all[i * num_process + my_id]);
            }

            int counter_send = 0;
            for(auto& i: send_buffer)
                if(i.size()!=0) counter_send++;
            request_send.resize(counter_send);

            int counter_receive = 0;
            for(auto& i: receive_buffer)
                if(i.size()!=0) counter_receive++;
            request_receive.resize(counter_receive);
        }

        void init(){
            send_buffer.resize(num_process);
            receive_buffer.resize(num_process);
            inited = true;
        }

        void clearCommMatrix()
        {
            mpi_size_matrix_all.clear();
        }

        std::vector<int> getCommMatrix() const
        {
            return mpi_size_matrix_all;
        }

        bool is_inited() {
            return inited;
        }

        void callMPI(bool checkSize = false){

            if(inited == false)throw std::runtime_error("not initialized!");
            if(mpi_size_matrix_all.size()==0) // NO size mat
            {
                MPI_array_wrapper<dataType> temp_mpiw;
                temp_mpiw.init();
                for (int i = 0; i < (int)send_buffer.size(); i++)
                {
                    temp_mpiw.send_buffer[i] = send_buffer[i];
                }
                temp_mpiw.callMPI();
                for (int i = 0; i < (int)send_buffer.size(); i++)
                {
                    receive_buffer[i] = temp_mpiw.receive_buffer[i];
                }
                std::vector<std::vector<int>> sizemat(num_process);
                for (int i = 0; i < num_process; i++)
                {
                    sizemat[i].resize(num_process);
                    for (int j = 0; j < num_process; j++)
                    {
                        sizemat[i][j] = temp_mpiw.mpi_size_matrix_all[i * num_process + j];
                    }
                }
                init(sizemat);
            }

            if(checkSize)
            {
                for(int i = 0; i < num_process; i++){
                    if(mpi_size_matrix_all[my_id * num_process + i] != (int)send_buffer[i].size())
                        throw std::runtime_error("size of send buffer is wrong");
                    if(mpi_size_matrix_all[i * num_process + my_id] != (int)receive_buffer[i].size())
                        throw std::runtime_error("size of receive buffer is wrong");
                }
            }

            int id_send = 0;
            int id_receive = 0;
            for(int i = 0; i < num_process; i++){
                if(mpi_size_matrix_all[my_id * num_process + i] != 0){
                    rc_check(psum_mpi_isend(
                        send_buffer[i].data(),
                        (int)(mpi_size_matrix_all[my_id * num_process + i] * sizeof(dataType)),
                        PSUM_MPI_BYTE,
                        i,
                        0,
                        &request_send[id_send]
                    ));
                    id_send++;
                }
                if(mpi_size_matrix_all[i * num_process + my_id] != 0)
                {
                    rc_check(psum_mpi_irecv(
                        receive_buffer[i].data(),
                        (int)(mpi_size_matrix_all[i * num_process + my_id] * sizeof(dataType)),
                        PSUM_MPI_BYTE,
                        i,
                        0,
                        &request_receive[id_receive]
                    ));
                    id_receive++;
                }
            }

            for(int i = 0; i < (int)request_receive.size(); i++){
                int receive_index = 0;
                rc_check(psum_mpi_waitany(
                    (int)request_receive.size(),
                    request_receive.data(),
                    &receive_index,
                    PSUM_MPI_STATUS_IGNORE
                ));
            }

            rc_check(psum_mpi_waitall((int)request_send.size(), request_send.data()));

            for(auto& r : request_receive) rc_check(psum_mpi_request_free(&r));
            for(auto& r : request_send)    rc_check(psum_mpi_request_free(&r));

            for(int i = 0; i < num_process; i++){
                send_buffer[i].resize(0);
            }
        }

    };

}

}

#endif