#include <psum/field.hpp>

using namespace psum::field;

template<int Dim>
void test_field_interp(){
    sycl::queue q{sycl::default_selector_v};
    typedef device_field<Dim, var_loc::nodeCentered, double, 1> nf;
    typedef device_field<Dim, var_loc::nodeCentered, double, 3> vf;
    std::vector<double> lower_bound(Dim, 0.0);
    std::vector<double> upper_bound(Dim, 1.0);
    std::vector<int> coarse_size(Dim, 15);
    std::vector<int> refine_size(Dim, 50);
    simple_grid<Dim> grid_c(lower_bound, upper_bound, coarse_size);
    simple_grid<Dim> grid_r(lower_bound, upper_bound, refine_size);

    nf A(q, grid_c);
    vf B(q, grid_r);
    vf C(q, grid_c);
    vf D(q, grid_c);

    A.for_each([&](sycl::handler &h) {
        return [=](size_t i, typename nf::Value& val) {
            val = i * i;
        };
    });

    B.for_each([&](sycl::handler &h) {
        auto A_acc = A.get_access(h);
        return [=](size_t i, typename vf::Value& val, const Eigen::RowVector<double, Dim>& p ) {
            val[0]= interp_nearest(p, A_acc);
            val[1]= interp(p, A_acc);
            val[2]= i;
        };
    });

    C.for_each([&](sycl::handler &h) {
        auto B_acc = B.get_access(h);
        return [=](size_t i, typename vf::Value& val, const Eigen::RowVector<double, Dim>& p ) {
            val = interp(p, B_acc);
        };
    });

    D.for_each([&](sycl::handler &h) {
        auto B_acc = B.get_access(h);
        return [=](size_t i, typename vf::Value& val, const Eigen::RowVector<double, Dim>& p ) {
            val = interp_nearest(p, B_acc);
        };
    });

    B.plot("test" + std::to_string(Dim) + "_interp.plt");
    C.plot("test" + std::to_string(Dim) + "_re_interp_nearest.plt");
    D.plot("test" + std::to_string(Dim) + "_re_interp_nearest_A.plt");
}

int main(){
    test_field_interp<1>();
    test_field_interp<2>();
    test_field_interp<3>();
    return 0;
}