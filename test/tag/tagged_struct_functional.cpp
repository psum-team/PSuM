#include <psum/tag.hpp>
#include "../../src/timer.hpp"
#include <atomic>
#include <Eigen/Core>
#include <iostream>

using namespace psum::tag;
using namespace psum::tag::property;
using namespace std;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>
>;

int main() {
    Particle p;
    get<velocity>(p) << 0, 0, 0;
    get<position>(p) << 1, 2, 3;
    get<weight>(p) = 4;

    ostream_settings::set_like_json();
    cout << p << endl;

    std::vector<Particle> ps_vec;
Tic("insert_vec")
    for (int i = 0; i < 1e7; i++)
    {
        Particle p;
        get<position>(p) << i, 0, 0;
        ps_vec.push_back(p);
    }

TocTic("insert_soa")
    for (int i = 0; i < 1e7; i++)
    {
        Particle p;
        get<position>(p) << i, 0, 0;
    }
Toc

    atomic_int counter_vec{}, counter_soa{};
Tic("loop_vec")
#pragma omp parallel for num_threads(6)
    for(auto& i: ps_vec)
    {
        if (get<position>(i).x()>1e6&&get<position>(i).x()<1.1e6)
            counter_vec++;
    }
Toc
PrintTimer
    cout << counter_vec << "," << counter_soa << endl;

    return 0;
}