#include <psum/field.hpp>

using namespace psum::field;

template<int Dim>
void test_field_interp(){
    typedef host_field<Dim, var_loc::nodeCentered, double, 1> nf;
    typedef host_field<Dim, var_loc::nodeCentered, double, 3> vf;
    std::vector<double> lower_bound(Dim, 0.0);
    std::vector<double> upper_bound(Dim, 1.0);
    std::vector<int> coarse_size(Dim, 15);
    std::vector<int> refine_size(Dim, 50);
    simple_grid<Dim> grid_c(lower_bound, upper_bound, coarse_size);
    simple_grid<Dim> grid_r(lower_bound, upper_bound, refine_size);

    nf A(grid_c);
    vf B(grid_r);
    vf C(grid_c);
    vf D(grid_c);

    A.for_each(
        [&](size_t i, typename nf::Value& val) {
            val = i * i;
        }
    );

    B.for_each(
        [&](size_t i, typename vf::Value &val, const Eigen::RowVector<double, Dim> &p) {
            val[0] = interp_nearest(p, A);
            val[1] = interp(p, A);
            val[2]= i;
        });

    C.for_each(
        [&](size_t i, typename vf::Value& val, const Eigen::RowVector<double, Dim>& p ) {
            val = interp(p, B);
        }
    );

    D.for_each(
        [&](size_t i, typename vf::Value& val, const Eigen::RowVector<double, Dim>& p ) {
            val = interp_nearest(p, B);
        }
    );

    B.plot("test" + std::to_string(Dim) + "_interp_host.plt");
    C.plot("test" + std::to_string(Dim) + "_re_interp_nearest_host.plt");
    D.plot("test" + std::to_string(Dim) + "_re_interp_nearest_A_host.plt");
}

int main(){
    test_field_interp<1>();
    test_field_interp<2>();
    test_field_interp<3>();
    return 0;
}