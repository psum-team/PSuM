// test_mpi_edges — MPI edge cases: empty / one-way / repeated communication.
// Rank-count agnostic (run with -np 2/3/4); complements test_comm_wrapper.
#include "../../src/mpi/comm_wrapper.hpp"
#include <cstdio>

using namespace psum::mpi;

int main(int argc, char** argv)
{
    if (psum_mpi_init(&argc, &argv) != PSUM_MPI_SUCCESS) { printf("psum_mpi_init failed\n"); return 1; }
    int rank = 0, P = 1;
    psum_mpi_comm_rank(&rank);
    psum_mpi_comm_size(&P);
    int fails = 0;

    // 1) empty communication: nobody sends; callMPI must not throw
    {
        MPI_array_wrapper<int> w;
        w.init();
        w.callMPI();
        for (int i = 0; i < P; i++)
            if (!w.receive_buffer[i].empty()) {
                printf("FAIL rank %d: empty-comm recv[%d].size=%zu\n", rank, i, w.receive_buffer[i].size());
                fails++;
            }
    }

    // 2) one-way on the SAME wrapper across rounds: non-empty -> empty -> different length
    {
        MPI_array_wrapper<int> w;
        w.init();
        const int sizes[3] = { -1, 0, 2 };   // -1 = rank-dependent count
        const int offs[3] = { 0, 0, 50 };
        for (int round = 0; round < 3; round++) {
            if (rank == 0) {
                for (int i = 1; i < P; i++) {
                    int n = (sizes[round] < 0) ? i : sizes[round];
                    w.send_buffer[i].resize(n);
                    for (int j = 0; j < n; j++) w.send_buffer[i][j] = offs[round] + i * 100 + j;
                }
            }
            w.callMPI();
            if (rank != 0) {
                const std::vector<int>& r0 = w.receive_buffer[0];
                int expect_n = (sizes[round] < 0) ? rank : sizes[round];
                if ((int)r0.size() != expect_n) {
                    printf("FAIL rank %d round %d: recv size=%zu expect %d\n", rank, round, r0.size(), expect_n);
                    fails++;
                } else {
                    for (int j = 0; j < expect_n; j++)
                        if (r0[j] != offs[round] + rank * 100 + j) {
                            printf("FAIL rank %d round %d: val[%d]=%d expect %d\n", rank, round, j, r0[j], offs[round] + rank * 100 + j);
                            fails++;
                        }
                }
            }
        }
    }

    // 3) fixed wrapper, empty
    {
        MPI_fixed_array_wrapper<double> f;
        f.init();
        f.callMPI(true);
        for (int i = 0; i < P; i++)
            if (!f.receive_buffer[i].empty()) {
                printf("FAIL rank %d: fixed empty-comm recv[%d] non-empty\n", rank, i);
                fails++;
            }
    }

    int total_fails = 0;
    psum_mpi_allreduce(&fails, &total_fails, 1, PSUM_MPI_INT, PSUM_MPI_SUM);
    psum_mpi_finalize();
    if (total_fails) { printf("rank %d/%d: MPI EDGES FAIL (total_fails=%d)\n", rank, P, total_fails); return 1; }
    printf("rank %d/%d: MPI EDGES ALL PASS\n", rank, P);
    return 0;
}
