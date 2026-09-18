#include "interface.h"
#include <mpi.h>
#include <stdlib.h>

static const MPI_Datatype k_dtype[] = {
    MPI_CHAR, MPI_BYTE, MPI_INT, MPI_LONG_LONG_INT, MPI_FLOAT, MPI_DOUBLE
};
static const MPI_Op k_op[] = { MPI_SUM, MPI_MAX, MPI_MIN };

struct psum_mpi_request {
    MPI_Request r;
};

static int ensure_dtype(int datatype, MPI_Datatype* out)
{
    if (datatype < 0 || (size_t)datatype >= sizeof(k_dtype)/sizeof(k_dtype[0]))
        return PSUM_MPI_ERR_HANDLE;
    *out = k_dtype[datatype];
    return PSUM_MPI_SUCCESS;
}

static int new_req(psum_mpi_request_t** req)
{
    if (!req) return PSUM_MPI_ERR_HANDLE;
    *req = (psum_mpi_request_t*)malloc(sizeof(psum_mpi_request_t));
    if (!*req) return PSUM_MPI_ERR_HANDLE;
    return PSUM_MPI_SUCCESS;
}

int psum_mpi_init(int* argc, char*** argv)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) return PSUM_MPI_SUCCESS;
    return MPI_Init(argc, argv);
}

int psum_mpi_finalize(void)
{
    return MPI_Finalize();
}

int psum_mpi_comm_rank(int* rank)  { return MPI_Comm_rank(MPI_COMM_WORLD, rank); }
int psum_mpi_comm_size(int* size)  { return MPI_Comm_size(MPI_COMM_WORLD, size); }
int psum_mpi_abort(int code)       { return MPI_Abort(MPI_COMM_WORLD, code); }
double psum_mpi_wtime(void)        { return MPI_Wtime(); }

int psum_mpi_send(const void* buf, int count, int datatype, int dest, int tag)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    return MPI_Send(buf, count, t, dest, tag, MPI_COMM_WORLD);
}

int psum_mpi_recv(void* buf, int count, int datatype, int src, int tag,
                  psum_mpi_status_t* status)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    MPI_Status st;
    rc = MPI_Recv(buf, count, t, src, tag, MPI_COMM_WORLD, &st);
    if (status) {
        status->source = st.MPI_SOURCE;
        status->tag = st.MPI_TAG;
        status->error = rc;
    }
    return rc;
}

int psum_mpi_isend(const void* buf, int count, int datatype, int dest, int tag,
                   psum_mpi_request_t** req)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    rc = new_req(req);
    if (rc) return rc;
    rc = MPI_Isend(buf, count, t, dest, tag, MPI_COMM_WORLD, &(*req)->r);
    if (rc != MPI_SUCCESS) { free(*req); *req = NULL; }
    return rc;
}

int psum_mpi_irecv(void* buf, int count, int datatype, int src, int tag,
                   psum_mpi_request_t** req)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    rc = new_req(req);
    if (rc) return rc;
    rc = MPI_Irecv(buf, count, t, src, tag, MPI_COMM_WORLD, &(*req)->r);
    if (rc != MPI_SUCCESS) { free(*req); *req = NULL; }
    return rc;
}

int psum_mpi_waitany(int count, psum_mpi_request_t** reqs, int* index,
                     psum_mpi_status_t* status)
{
    if (count < 0) return PSUM_MPI_ERR_HANDLE;
    if (index) *index = -1;                       /* count==0: no completion */
    if (count == 0) return PSUM_MPI_SUCCESS;      /* zero requests: no-op, allows NULL */
    if (!reqs) return PSUM_MPI_ERR_HANDLE;
    MPI_Request* raw = (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!raw) return PSUM_MPI_ERR_HANDLE;
    for (int i = 0; i < count; i++)
        raw[i] = reqs[i] ? reqs[i]->r : MPI_REQUEST_NULL;
    MPI_Status st;
    int rc = MPI_Waitany(count, raw, index, &st);
    for (int i = 0; i < count; i++)
        if (reqs[i]) reqs[i]->r = raw[i];
    if (status) {
        status->source = st.MPI_SOURCE;
        status->tag = st.MPI_TAG;
        status->error = rc;
    }
    free(raw);
    return rc;
}

int psum_mpi_wait(psum_mpi_request_t** req)
{
    if (!req || !*req) return PSUM_MPI_ERR_HANDLE;
    return MPI_Wait(&(*req)->r, MPI_STATUS_IGNORE);
}

int psum_mpi_waitall(int count, psum_mpi_request_t** reqs)
{
    if (count < 0) return PSUM_MPI_ERR_HANDLE;
    if (count == 0) return PSUM_MPI_SUCCESS;      /* zero requests: no-op, allows NULL */
    if (!reqs) return PSUM_MPI_ERR_HANDLE;
    MPI_Request* raw = (MPI_Request*)malloc((size_t)count * sizeof(MPI_Request));
    if (!raw) return PSUM_MPI_ERR_HANDLE;
    for (int i = 0; i < count; i++)
        raw[i] = reqs[i] ? reqs[i]->r : MPI_REQUEST_NULL;
    int rc = MPI_Waitall(count, raw, MPI_STATUSES_IGNORE);
    for (int i = 0; i < count; i++)
        if (reqs[i]) reqs[i]->r = raw[i];
    free(raw);
    return rc;
}

int psum_mpi_request_free(psum_mpi_request_t** req)
{
    if (!req || !*req) return PSUM_MPI_ERR_HANDLE;
    if ((*req)->r == MPI_REQUEST_NULL) {
        free(*req);
        *req = NULL;
        return PSUM_MPI_SUCCESS;
    }
    int rc = MPI_Request_free(&(*req)->r);
    free(*req);
    *req = NULL;
    return rc;
}

int psum_mpi_barrier(void)
{
    return MPI_Barrier(MPI_COMM_WORLD);
}

int psum_mpi_allreduce(const void* sendbuf, void* recvbuf, int count,
                       int datatype, int op)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    if (op < 0 || (size_t)op >= sizeof(k_op)/sizeof(k_op[0]))
        return PSUM_MPI_ERR_HANDLE;
    return MPI_Allreduce(sendbuf, recvbuf, count, t, k_op[op], MPI_COMM_WORLD);
}

int psum_mpi_bcast(void* buf, int count, int datatype, int root)
{
    MPI_Datatype t;
    int rc = ensure_dtype(datatype, &t);
    if (rc) return rc;
    return MPI_Bcast(buf, count, t, root, MPI_COMM_WORLD);
}
