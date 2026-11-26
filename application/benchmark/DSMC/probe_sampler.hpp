#pragma once

#include <psum/psum.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

class N2ProbeSampler {
public:
    N2ProbeSampler(sycl::queue q, const std::filesystem::path& path, double dt)
        : sample_steps(int(std::llround(1e-7 / dt))), lookup(q, make_lookup(selected)),
          sums(q, 8 * selected.size()) {
        if (sample_steps < 1 || std::abs(sample_steps * dt - 1e-7) > 1e-15)
            throw std::runtime_error("dt must divide probe sample period");
        out.open(path / "local-moments.csv");
        if (!out) throw std::runtime_error("cannot create local moments");
        out << std::setprecision(17) << "step,ix,iy,N,sx,sy,sz,sxx,syy,szz,se\n";
        sums.fill(0.0);
    }

    const int sample_steps;

    template<class ParticleGroup>
    void sample(int step, const ParticleGroup& group) {
        const auto* cells = lookup.data();
        auto* dst = sums.data();
        using namespace psum::tag;
        const size_t extent = group.extent();
        if (extent) sums.get_queue().submit([&](sycl::handler& h) {
            const auto particles = group.get_content().get_access(h);
            h.parallel_for(sycl::range<1>(extent), [=](sycl::id<1> id) {
                const auto& particle = particles[id[0]];
                if (!ParticleGroup::validator::is_valid(particle)) return;
                const auto& pos = get<property::position>(particle);
                int i = int(sycl::floor(pos.x() / .0005));
                int j = int(sycl::floor((pos.y() + .075) / .0005));
                if (i < 0 || i >= 350 || j < 0 || j >= 300) return;
                const int cell = cells[j * 350 + i];
                if (cell < 0) return;
                const auto& v = get<property::velocity>(particle);
                const double moments[] = {1., v.x(), v.y(), v.z(), v.x()*v.x(), v.y()*v.y(),
                                          v.z()*v.z(), get<property::internal_energy>(particle)};
                using Atomic = sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                                sycl::access::address_space::global_space>;
                for (int k = 0; k < 8; ++k) Atomic(dst[8 * cell + k]).fetch_add(moments[k]);
            });
        }).wait_and_throw();
        const auto host = sums.to_host();
        for (size_t cell = 0; cell < selected.size(); ++cell) {
            out << step << ',';
            out << selected[cell] % 350 << ',' << selected[cell] / 350;
            for (int k = 0; k < 8; ++k) out << ',' << host[8 * cell + k];
            out << '\n';
        }
        if (step % 1000 == 0) out.flush();
        sums.fill(0.0);
    }

private:
    static std::vector<int> make_lookup(std::vector<int>& selected) {
        constexpr double centers[][2] = {{.04,0},{.0525,.005},{.1,.005},{.1,.0125},
                                         {.1475,.005},{.16,0},{.1,-.005}};
        std::vector<int> cells(350 * 300, -1);
        for (int j = 0; j < 300; ++j) {
            for (int i = 0; i < 350; ++i) {
                const double x = (i + .5) * .0005, y = -.075 + (j + .5) * .0005;
                bool keep = false;
                for (const auto& c : centers)
                    if (std::abs(x-c[0]) < .0025-1e-12 && std::abs(y-c[1]) < .0015-1e-12)
                        keep = true;
                if (keep) {
                    cells[j * 350 + i] = int(selected.size());
                    selected.push_back(j * 350 + i);
                }
            }
        }
        return cells;
    }

    std::ofstream out;
    std::vector<int> selected;
    psum::field::device_array<int> lookup;
    psum::field::device_array<double> sums;
};
