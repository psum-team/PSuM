#include <psum/field.hpp>
#include "../../src/utils_sycl.hpp"
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace psum;
using namespace field;
using namespace interp_tools;
using namespace std;
using namespace utils_sycl;

auto R = [](){return rand()/double(RAND_MAX);};

bool testG1D(sycl::queue& q){
    typedef node_field1D<double> nf;

    grid1D g({0.0}, {1.0}, {10});

    nf A(q, g);

    A.for_each([&](sycl::handler& h) {
        int A_size = A.size();
        return [=](size_t i, nf::Value& v){
            v = i*i*1.0 / A_size;
        };
    });

    std::vector<double> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = R()*g.span<0>() + g.lowerBound<0>();
    }
    device_array<double> rand_pos_dev(q, rand_pos);

    double* err = shared_variable<double>(q);
    *err = 0.0;

    // just for shorter!
    using Vec1D = Eigen::Vector<double, 1>;

    for_each(q, 10000, [&](sycl::handler& h) {
        auto A_acc = A.get_access(h);
        auto p_acc = rand_pos_dev.get_access(h);
        return [=](size_t i){
            double pos = p_acc[i];
            auto val_1 = interp_diff(Vec1D{pos}, A_acc);
            double val_2 = (interp(Vec1D{pos + 1e-7}, A_acc) - interp(Vec1D{pos - 1e-7}, A_acc)) / 2e-7;
            atomic_add(*err, abs(val_1[0] - val_2));
        };
    });
    std::cout << "Accumulation error in Grid1D:" << *err;
    bool ok = *err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

bool testG2D(sycl::queue& q){
    typedef node_field2D<double> nf;

    grid2D g({0.0, 0.0}, {1.0, 1.0}, {10, 11});

    nf A(q, g);

    A.for_each([&](sycl::handler& h) {
        int A_size = A.size();
        return [=](size_t i, nf::Value& v){
            v = i*i*1.0 / A_size;
        };
    });

    std::vector<Eigen::Vector2d> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = {R()*g.span<0>() + g.lowerBound<0>(), R()*g.span<1>() + g.lowerBound<1>()};
    }
    device_array<Eigen::Vector2d> rand_pos_dev(q, rand_pos);

    double* err = shared_variable<double>(q);
    *err = 0.0;

    // just for shorter!
    using Vec2D = Eigen::Vector<double, 2>;

    for_each(q, 10000, [&](sycl::handler& h) {
        auto A_acc = A.get_access(h);
        auto p_acc = rand_pos_dev.get_access(h);
        return [=](size_t i){
            Eigen::Vector2d pos = p_acc[i];
            auto [val_1_x, val_1_y] = interp_diff(pos, A_acc);
            auto val_2_x = (interp(Vec2D{pos[0] + 1e-7, pos[1]}, A_acc) - interp(Vec2D{pos[0] - 1e-7, pos[1]}, A_acc)) / 2e-7;
            auto val_2_y = (interp(Vec2D{pos[0], pos[1] + 1e-7}, A_acc) - interp(Vec2D{pos[0], pos[1] - 1e-7}, A_acc)) / 2e-7;
            atomic_add(*err, abs(val_1_x - val_2_x) + abs(val_1_y - val_2_y));
        };
    });
    std::cout << "Accumulation error in Grid2D:" << *err;
    bool ok = *err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

bool testG3D(sycl::queue& q){
    typedef node_field3D<double> nf;

    grid3D g({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {4, 5, 6});

    nf A(q, g);

    A.for_each([&](sycl::handler& h) {
        int A_size = A.size();
        return [=](size_t i, nf::Value& v){
            v = i*i*1.0 / A_size;
        };
    });

    std::vector<Eigen::Vector3d> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = {R()*g.span<0>() + g.lowerBound<0>(), R()*g.span<1>() + g.lowerBound<1>(), R()*g.span<2>() + g.lowerBound<2>()};
    }
    device_array<Eigen::Vector3d> rand_pos_dev(q, rand_pos);

    double* err = shared_variable<double>(q);
    *err = 0.0;

    // just for shorter!
    using Vec3D = Eigen::Vector<double, 3>;

    for_each(q, 10000, [&](sycl::handler& h) {
        auto A_acc = A.get_access(h);
        auto p_acc = rand_pos_dev.get_access(h);
        return [=](size_t i){
            Eigen::Vector3d pos = p_acc[i];
            auto [val_1_x, val_1_y, val_1_z] = interp_diff(pos, A_acc);
            auto val_2_x = (interp(Vec3D{pos[0] + 1e-7, pos[1], pos[2]}, A_acc) - interp(Vec3D{pos[0] - 1e-7, pos[1], pos[2]}, A_acc)) / 2e-7;
            auto val_2_y = (interp(Vec3D{pos[0], pos[1] + 1e-7, pos[2]}, A_acc) - interp(Vec3D{pos[0], pos[1] - 1e-7, pos[2]}, A_acc)) / 2e-7;
            auto val_2_z = (interp(Vec3D{pos[0], pos[1], pos[2] + 1e-7}, A_acc) - interp(Vec3D{pos[0], pos[1], pos[2] - 1e-7}, A_acc)) / 2e-7;
            atomic_add(*err, abs(val_1_x - val_2_x) + abs(val_1_y - val_2_y) + abs(val_1_z - val_2_z));
        };
    });
    std::cout << "Accumulation error in Grid3D:" << *err;
    bool ok = *err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

int main()
{
    sycl::queue q{sycl::default_selector_v};
    int failures = 0;
    failures += testG1D(q) ? 0 : 1;
    failures += testG2D(q) ? 0 : 1;
    failures += testG3D(q) ? 0 : 1;
    return failures ? 1 : 0;
}