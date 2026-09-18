#include "../../src/mpi/comm_wrapper.hpp"

using namespace psum::mpi;

int main(int argc, char** argv) {
    if (psum_mpi_init(&argc, &argv) != PSUM_MPI_SUCCESS) { printf("psum_mpi_init failed\n"); return 1; }
    // 1) array_wrapper：0→1, sending 3 integers. 1→0, sending 5 integers
    MPI_array_wrapper<int> w;
    w.init();
    if (w.my_id == 0) w.send_buffer[1] = {1, 2, 3};
    if (w.my_id == 1) w.send_buffer[0] = {9, 8, 7, 6, 5};
    w.callMPI();
    if (w.my_id == 0 && (w.receive_buffer[1].size() != 5 || w.receive_buffer[1][4] != 5)) { printf("FAIL array r0\n"); return 1; }
    if (w.my_id == 1 && (w.receive_buffer[0].size() != 3 || w.receive_buffer[0][2] != 3)) { printf("FAIL array r1\n"); return 1; }

    // 2) fixed_wrapper
    MPI_fixed_array_wrapper<double> f;
    f.init();
    if (f.my_id == 0) f.send_buffer[1].resize(2, 3.5);
    if (f.my_id == 1) f.send_buffer[0].resize(4, 1.25);
    // size matrix will be automatically calculated by callMPI and frozen
    f.callMPI(true);
    if (f.my_id == 0 && (f.receive_buffer[1].size() != 4 || f.receive_buffer[1][0] != 1.25)) { printf("FAIL fixed r0\n"); return 1; }
    if (f.my_id == 1 && (f.receive_buffer[0].size() != 2 || f.receive_buffer[0][0] != 3.5)) { printf("FAIL fixed r1\n"); return 1; }

    printf("rank %d/%d: COMM WRAPPER ALL PASS\n", w.my_id, w.num_process);
    psum_mpi_finalize();
    return 0;
}
