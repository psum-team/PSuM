#include "../../src/mpi/distributed_la_obj.hpp"

using namespace psum::mpi;

static inline double next_rand(unsigned long long& s)
{
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((s >> 11) & 0x1FFFFFFFFFFFFFULL) / (double)(1ULL << 53);
}

int main(int argc, char* argv[])
{
    psum_mpi_init(&argc, &argv);

    int my_id = 0, num_proc = 1;
    psum_mpi_comm_rank(&my_id);
    psum_mpi_comm_size(&num_proc);

    const int I = 100, J = 100, N = I * J;

    Eigen::SparseMatrix<double> LaplaceMat;
    LaplaceMat.resize(N, N);
    std::vector<Eigen::Triplet<double>> trips;
    {
        unsigned long long rs = 0;
        for (int i = 1; i < I - 1; i++)
            for (int j = 1; j < J - 1; j++)
            {
                int index = i * J + j;
                double c1 = 0.5 * next_rand(rs) + 0.5;
                double c2 = 0.5 * next_rand(rs) + 0.5;
                double c3 = 0.5 * next_rand(rs) + 0.5;
                double c4 = 0.5 * next_rand(rs) + 0.5;
                trips.push_back(Eigen::Triplet<double>(index, index - 1, c1 / 4));
                trips.push_back(Eigen::Triplet<double>(index, index + 1, c2 / 4));
                trips.push_back(Eigen::Triplet<double>(index, index - J, c3 / 4));
                trips.push_back(Eigen::Triplet<double>(index, index + J, c4 / 4));
                trips.push_back(Eigen::Triplet<double>(index, index, 1 - (c1 + c2 + c3 + c4) / 4));
            }
    }
    for (int i = 0; i < I; i++)
    {
        trips.push_back(Eigen::Triplet<double>(i * J + 0, i * J + 0, 1.0));
        trips.push_back(Eigen::Triplet<double>(i * J + J - 1, i * J + J - 1, 1.0));
    }
    for (int j = 1; j < J - 1; j++)
    {
        trips.push_back(Eigen::Triplet<double>(0 * J + j, 0 * J + j, 1.0));
        trips.push_back(Eigen::Triplet<double>((I - 1) * J + j, (I - 1) * J + j, 1.0));
    }
    LaplaceMat.setFromTriplets(trips.begin(), trips.end());

    // use distributed vector and distributed sparse matrix
    std::vector<std::size_t> owner(N);
    distributed_EigenVector<double> vec;
    distributed_EigenSpMat<double> Mat_d;
    for (int i = 0; i < N; i++) owner[i] = (std::size_t)(i % vec.num_process());
    vec.resize(N);
    vec.setOwner(owner);
    Mat_d.set(LaplaceMat, vec, vec);

    {
        unsigned long long rs = (unsigned long long)vec.my_id();
        for (int i = 0; i < N; i++)
            if (owner[i] == (std::size_t)vec.my_id())
                vec[i] = next_rand(rs);
    }

    // ---- 50 times SpMV ----
    for (int loop = 0; loop < 50; loop++)
        Mat_d.product(vec, vec);
    vec.sync();

    // ---- check in rank 0 ----
    int ok = 1;
    if (vec.my_id() == 0)
    {
        Eigen::VectorXd vecloc(N);
        for (int id = 0; id < vec.num_process(); id++)
        {
            unsigned long long rs = (unsigned long long)id;
            for (int i = 0; i < N; i++)
                if (owner[i] == (std::size_t)id)
                    vecloc[i] = next_rand(rs);
        }
        for (int loop = 0; loop < 50; loop++)
            vecloc = LaplaceMat * vecloc;

        double dis = (vecloc - vec).norm();
        printf("distributed vs dense: |diff| = %g\n", dis);
        if (dis == 0.0) { printf("rank 0: PASS\n"); }
        else { printf("rank 0: FAIL (validation failed)\n"); ok = 0; }
        std::cout << "Comm size:\n" << Mat_d.getCommMat() << std::endl;
    }

    int ok_all = 1;
    psum_mpi_allreduce(&ok, &ok_all, 1, PSUM_MPI_INT, PSUM_MPI_MIN);
    psum_mpi_finalize();
    if (!ok_all) { printf("rank %d/%d: DISTRIBUTEDVEC FAIL\n", my_id, num_proc); return 1; }
    printf("rank %d/%d: DISTRIBUTEDVEC ALL PASS\n", my_id, num_proc);
    return 0;
}
