#include <psum/field.hpp>
#include <Eigen/Core>
#include <iostream>

using namespace Eigen;
using namespace psum;
using namespace field;
using namespace interp_tools;
using namespace std;

auto R = [](){return rand()/double(RAND_MAX);};

bool testG1D(){
    typedef host_field<1, var_loc::nodeCentered, double, 1> nf;

    simple_grid<1> g({0.0}, {1.0}, {10});

    nf A(g);

    A.for_each([&](size_t i, typename nf::Value& v){
        v = i*i*1.0 / g.contentSize(var_loc::nodeCentered);
    });

    std::vector<double> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = R()*g.get_spans()[0] + g.get_lower_bounds()[0];
    }

    double err = 0.0;

    using Vec1D = Eigen::Vector<double, 1>;

    for(int i=0;i<10000;i++){
        double pos = rand_pos[i];
        auto val_1 = interp_diff(Vec1D{pos}, A);
        double val_2 = (interp(Vec1D{pos + 1e-7}, A) - interp(Vec1D{pos - 1e-7}, A)) / 2e-7;
        err += abs(val_1[0] - val_2);
    }
    std::cout << "Accumulation error in Grid1D:" << err;
    bool ok = err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

bool testG2D(){
    typedef host_field<2, var_loc::nodeCentered, double, 1> nf;

    simple_grid<2> g({0.0, 0.0}, {1.0, 1.0}, {10, 11});

    nf A(g);

    A.for_each([&](size_t i, typename nf::Value& v){
        v = i*i*1.0 / g.contentSize(var_loc::nodeCentered);
    });

    std::vector<Eigen::Vector2d> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = {R()*g.get_spans()[0] + g.get_lower_bounds()[0], R()*g.get_spans()[1] + g.get_lower_bounds()[1]};
    }

    double err = 0.0;

    using Vec2D = Eigen::Vector<double, 2>;

    for(int i=0;i<10000;i++){
        Eigen::Vector2d pos = rand_pos[i];
        auto [val_1_x, val_1_y] = interp_diff(pos, A);
        auto val_2_x = (interp(Vec2D{pos[0] + 1e-7, pos[1]}, A) - interp(Vec2D{pos[0] - 1e-7, pos[1]}, A)) / 2e-7;
        auto val_2_y = (interp(Vec2D{pos[0], pos[1] + 1e-7}, A) - interp(Vec2D{pos[0], pos[1] - 1e-7}, A)) / 2e-7;
        err += abs(val_1_x - val_2_x) + abs(val_1_y - val_2_y);
    }
    std::cout << "Accumulation error in Grid2D:" << err;
    bool ok = err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

bool testG3D(){
    typedef host_field<3, var_loc::nodeCentered, double, 1> nf;

    simple_grid<3> g({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {4, 5, 6});

    nf A(g);

    A.for_each([&](size_t i, typename nf::Value& v){
        v = i*i*1.0 / g.contentSize(var_loc::nodeCentered);
    });

    std::vector<Eigen::Vector3d> rand_pos(10000);
    for(int i=0;i<10000;i++){
        rand_pos[i] = {R()*g.get_spans()[0] + g.get_lower_bounds()[0], R()*g.get_spans()[1] + g.get_lower_bounds()[1], R()*g.get_spans()[2] + g.get_lower_bounds()[2]};
    }

    double err = 0.0;

    using Vec3D = Eigen::Vector<double, 3>;

    for(int i=0;i<10000;i++){
        Eigen::Vector3d pos = rand_pos[i];
        auto [val_1_x, val_1_y, val_1_z] = interp_diff(pos, A);
        auto val_2_x = (interp(Vec3D{pos[0] + 1e-7, pos[1], pos[2]}, A) - interp(Vec3D{pos[0] - 1e-7, pos[1], pos[2]}, A)) / 2e-7;
        auto val_2_y = (interp(Vec3D{pos[0], pos[1] + 1e-7, pos[2]}, A) - interp(Vec3D{pos[0], pos[1] - 1e-7, pos[2]}, A)) / 2e-7;
        auto val_2_z = (interp(Vec3D{pos[0], pos[1], pos[2] + 1e-7}, A) - interp(Vec3D{pos[0], pos[1], pos[2] - 1e-7}, A)) / 2e-7;
        err += abs(val_1_x - val_2_x) + abs(val_1_y - val_2_y) + abs(val_1_z - val_2_z);
    }
    std::cout << "Accumulation error in Grid3D:" << err;
    bool ok = err / 10000 < 1e-6;
    if (ok) std::cout << " ✅" << std::endl;
    else std::cout << " ❌" << std::endl;
    std::cout << "    (this value should be much smaller than 1)" << std::endl;
    return ok;
}

int main()
{
    int failures = 0;
    failures += testG1D() ? 0 : 1;
    failures += testG2D() ? 0 : 1;
    failures += testG3D() ? 0 : 1;
    return failures ? 1 : 0;
}
