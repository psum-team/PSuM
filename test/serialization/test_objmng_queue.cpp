#include <psum/tag.hpp>
#include <psum/serialization.hpp>
#include <psum/particle_container.hpp>
#include <iostream>

using namespace std;
using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::serialization;
using namespace psum::particle_container;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>
>;

using pGroup = particle_group<Particle, pos_x_nan_is_invalid>;

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

    sycl::queue q;
    object_manager mng("tmp.mas", q);

    auto& p1 = mng.obj<Particle>("p1");

    get<position>(p1) = Eigen::RowVector3d(1, 2, 3);
    get<velocity>(p1) = Eigen::RowVector3d(4, 5, 6);
    get<weight>(p1) = 0.1;

    auto& int_vector = mng.obj<device_vector<int>>("int_vector");
    int_vector.copy(vector<int>{7, 42});

    auto& pgroup = mng.obj<pGroup>("pgroup");
    pgroup.insert(vector<Particle>{p1, p1});

    mas_file file("tmp_saved.mas", mas_file::replaceMode);
    mng.save(file);

    object_manager mng2("tmp_saved.mas", q);
    if (check_same(mng2.obj<Particle>("p1"), p1)) cout << "S/L p1: pass\n"; else cout << "S/L p1: fail\n";

    auto& loaded_int_vector = mng2.obj<device_vector<int>>("int_vector");
    auto loaded_host = loaded_int_vector.to_host();
    if (loaded_host.size() == 2 && loaded_host[0] == 7 && loaded_host[1] == 42) cout << "S/L int_vector: pass\n"; else cout << "S/L int_vector: fail\n";
    device_vector<Particle> loaded_particles(q);
    mng2.obj<pGroup>("pgroup").for_each([&](sycl::handler& h) {
        auto loaded_particles_acc = loaded_particles.get_access(h);
        return [=](const Particle& p) {
            loaded_particles_acc.push_back(p);
        };
    });
    auto loaded_particles_host = loaded_particles.to_host();
    if (mng2.obj<pGroup>("pgroup").size() == 2 && check_same(loaded_particles_host[0], p1) && check_same(loaded_particles_host[1], p1)) cout << "S/L pgroup: pass\n"; else cout << "S/L pgroup: fail\n";

    object_manager mng_noq("tmp_saved.mas");
    bool noq_pass = false;
    try {
        auto& bad = mng_noq.obj<device_vector<int>>("int_vector");
        (void)bad;
    } catch (const std::runtime_error& e) {
        noq_pass = string(e.what()).find("queue") != string::npos;
    }
    if (noq_pass) cout << "queue-less manager rejection: pass\n"; else cout << "queue-less manager rejection: fail\n";

    return 0;
}