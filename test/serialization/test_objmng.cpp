#include <psum/tag.hpp>
#include <psum/serialization.hpp>
#include <iostream>

using namespace std;
using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::serialization;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>
>;

auto check_same = [](Particle& p1, Particle& p2) {
    try {
        assert(get<position>(p1) == get<position>(p2));
        assert(get<velocity>(p1) == get<velocity>(p2));
        assert(get<weight>(p1) == get<weight>(p2));
        return true;
    } catch (const std::exception& e) {
        return false;
    }
};

int main() {

    object_manager mng("tmp.mas");

    auto& matrix = mng.obj<Eigen::MatrixXd>("my_matrix");
    auto& config = mng.obj<std::map<std::string, double>>("config");
    auto& counter = mng.obj<int>("counter");
    auto& p1 = mng.obj<Particle>("p1");

    matrix = Eigen::MatrixXd::Random(3, 3);
    config["threshold"] = 0.5;
    config["max_iter"] = 100;
    config["alpha"] = 0.1;
    counter = 999;
    get<position>(p1) = Eigen::RowVector3d(1, 2, 3);
    get<velocity>(p1) = Eigen::RowVector3d(4, 5, 6);
    get<weight>(p1) = 0.1;

    mas_file file("tmp_saved.mas", mas_file::replaceMode);
    mng.save(file);

    object_manager mng2("tmp_saved.mas");
    if (mng2.obj<Eigen::MatrixXd>("my_matrix") == matrix) cout << "S/L matrix: pass\n"; else cout << "S/L matrix: fail\n";
    if (mng2.obj<std::map<std::string, double>>("config") == config) cout << "S/L config: pass\n"; else cout << "S/L config: fail\n";
    if (mng2.obj<int>("counter") == counter) cout << "S/L counter: pass\n"; else cout << "S/L counter: fail\n";
    if (check_same(mng2.obj<Particle>("p1"), p1)) cout << "S/L p1: pass\n"; else cout << "S/L p1: fail\n";

    return 0;
}