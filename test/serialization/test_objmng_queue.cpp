#include <psum/tag.hpp>
#include <psum/serialization.hpp>
#include <psum/particle_container.hpp>
#include <psum/field.hpp>
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
    int failures = 0;


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

    auto& dfield = mng.obj<psum::field::cell_field1D<double>>("dfield");
    dfield = psum::field::cell_field1D<double>(q, psum::field::grid1D({0.0}, {1.0}, {16}));
    dfield.for_each([](sycl::handler& h) {
        return [](size_t i, double& v) { v = 0.125 * i; };
    });

    auto& darr = mng.obj<psum::field::device_array<double>>("darr");
    vector<double> arr_src(123, 0.0);
    arr_src.front() = 1.5; arr_src.back() = -7.25;
    darr = psum::field::device_array<double>(q, arr_src);

    auto& grid_out = mng.obj<psum::field::grid1D>("grid1");
    grid_out = psum::field::grid1D({0.0}, {1.0}, {16});

    using MPG = psum::field::multi_patch::multi_patch_grid<2>;
    auto& mpg_out = mng.obj<MPG>("mpg1");
    mpg_out = MPG(q, {0.0, 0.0}, {1.0, 1.0}, {2, 2}, {2, 2, 3, 2});

    mas_file file("tmp_saved.mas", mas_file::replaceMode);
    mng.save(file);

    object_manager mng2("tmp_saved.mas", q);
    if (check_same(mng2.obj<Particle>("p1"), p1)) cout << "S/L p1: pass\n"; else { cout << "S/L p1: fail\n"; failures++; }

    auto& loaded_int_vector = mng2.obj<device_vector<int>>("int_vector");
    auto loaded_host = loaded_int_vector.to_host();
    if (loaded_host.size() == 2 && loaded_host[0] == 7 && loaded_host[1] == 42) cout << "S/L int_vector: pass\n"; else { cout << "S/L int_vector: fail\n"; failures++; }
    device_vector<Particle> loaded_particles(q);
    mng2.obj<pGroup>("pgroup").for_each([&](sycl::handler& h) {
        auto loaded_particles_acc = loaded_particles.get_access(h);
        return [=](const Particle& p) {
            loaded_particles_acc.push_back(p);
        };
    });
    auto loaded_particles_host = loaded_particles.to_host();
    if (mng2.obj<pGroup>("pgroup").size() == 2 && check_same(loaded_particles_host[0], p1) && check_same(loaded_particles_host[1], p1)) cout << "S/L pgroup: pass\n"; else { cout << "S/L pgroup: fail\n"; failures++; }

    auto& loaded_dfield = mng2.obj<psum::field::cell_field1D<double>>("dfield");
    auto loaded_dfield_host = loaded_dfield.getContent().to_host();
    bool dfield_ok = loaded_dfield.has_grid() && loaded_dfield.size() == 16;
    for (size_t i = 0; dfield_ok && i < loaded_dfield.size(); i++)
        dfield_ok = dfield_ok && (loaded_dfield_host[i] == 0.125 * i);
    if (dfield_ok) cout << "S/L dfield: pass\n"; else { cout << "S/L dfield: fail\n"; failures++; }

    auto& loaded_grid = mng2.obj<psum::field::grid1D>("grid1");
    if (loaded_grid.get_cell_num() == std::array<int, 1>{16} && 
        loaded_grid.get_lower_bounds() == std::array<double, 1>{0.0} && 
        loaded_grid.get_upper_bounds() == std::array<double, 1>{1.0}
    ) 
        cout << "S/L grid1: pass\n"; else { cout << "S/L grid1: fail\n"; failures++; }

    auto& loaded_mpg = mng2.obj<MPG>("mpg1");
    {
        auto lp = loaded_mpg.get_patch_num();
        auto rp = loaded_mpg.get_resolution_exponents();
        bool mpg_ok = loaded_mpg.get_lower_bounds() == std::array<double, 2>{0.0, 0.0}
            && loaded_mpg.get_upper_bounds() == std::array<double, 2>{1.0, 1.0}
            && lp == std::array<int, 2>{2, 2}
            && rp == std::vector<int>{2, 2, 3, 2};
        if (mpg_ok) cout << "S/L mpg1: pass\n"; else { cout << "S/L mpg1: fail\n"; failures++; }
    }

    auto& loaded_darr = mng2.obj<psum::field::device_array<double>>("darr");
    auto loaded_darr_host = loaded_darr.to_host();
    if (loaded_darr_host == arr_src) cout << "S/L darr: pass\n"; else { cout << "S/L darr: fail\n"; failures++; }

    object_manager mng_noq("tmp_saved.mas");
    bool noq_pass = false;
    try {
        auto& bad = mng_noq.obj<device_vector<int>>("int_vector");
        (void)bad;
    } catch (const std::runtime_error& e) {
        noq_pass = string(e.what()).find("queue") != string::npos;
        try {
            auto& bad2 = mng_noq.obj<MPG>("mpg1");
            (void)bad2;
        } catch (const std::runtime_error& e2) {
            noq_pass = noq_pass && string(e2.what()).find("queue") != string::npos;
        }
        try {
            auto& bad3 = mng_noq.obj<psum::field::cell_field1D<double>>("dfield");
            (void)bad3;
        } catch (const std::runtime_error& e3) {
            noq_pass = noq_pass && string(e3.what()).find("queue") != string::npos;
        }
    }
    if (noq_pass) cout << "queue-less manager rejection: pass\n"; else { cout << "queue-less manager rejection: fail\n"; failures++; }

    if (failures) cout << "test_objmng_queue: " << failures << " FAILURES" << endl;
    else cout << "test_objmng_queue: all checks passed" << endl;
    return failures ? 1 : 0;
}