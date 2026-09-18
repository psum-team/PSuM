#ifndef PSUM_MPI_INTERFACE_H
#define PSUM_MPI_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif

    #define PSUM_MPI_SUCCESS 0
    #define PSUM_MPI_ERR_HANDLE 0x40000000
    #define PSUM_MPI_ERR_INIT   0x40000001

    // just index and compile unit should map psum_mpi_datatype to MPI_Datatype
    enum psum_mpi_datatype {
        PSUM_MPI_CHAR  = 0,
        PSUM_MPI_BYTE  = 1,
        PSUM_MPI_INT   = 2,
        PSUM_MPI_INT64 = 3,
        PSUM_MPI_FLOAT = 4,
        PSUM_MPI_DOUBLE = 5
    };

    // just index and compile unit should map psum_mpi_op to MPI_Op
    enum psum_mpi_op {
        PSUM_MPI_SUM = 0,
        PSUM_MPI_MAX = 1,
        PSUM_MPI_MIN = 2
    };

    typedef struct psum_mpi_status {
        int source;
        int tag;
        int error;
    } psum_mpi_status_t;

    typedef struct psum_mpi_request psum_mpi_request_t;

    #define PSUM_MPI_ANY_SOURCE (-1)
    #define PSUM_MPI_ANY_TAG    (-1)
    #define PSUM_MPI_STATUS_IGNORE ((psum_mpi_status_t*)0)

    int psum_mpi_init(int* argc, char*** argv);
    int psum_mpi_finalize(void);
    int psum_mpi_comm_rank(int* rank);
    int psum_mpi_comm_size(int* size);
    int psum_mpi_abort(int error_code);

    double psum_mpi_wtime(void);

    int psum_mpi_send(const void* buf, int count, int datatype, int dest, int tag);
    int psum_mpi_recv(void* buf, int count, int datatype, int src, int tag, psum_mpi_status_t* status);
    int psum_mpi_isend(const void* buf, int count, int datatype, int dest, int tag, psum_mpi_request_t** req);
    int psum_mpi_irecv(void* buf, int count, int datatype, int src, int tag, psum_mpi_request_t** req);
    int psum_mpi_waitany(int count, psum_mpi_request_t** reqs, int* index, psum_mpi_status_t* status);
    int psum_mpi_wait(psum_mpi_request_t** req);
    int psum_mpi_waitall(int count, psum_mpi_request_t** reqs);
    int psum_mpi_request_free(psum_mpi_request_t** req);
    int psum_mpi_barrier(void);
    int psum_mpi_allreduce(const void* sendbuf, void* recvbuf, int count, int datatype, int op);
    int psum_mpi_bcast(void* buf, int count, int datatype, int root);

#ifdef __cplusplus
}

#endif

#endif
