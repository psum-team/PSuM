#include "../../../application/benchmark/DSMC/custom_dpmcc.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace psum::prelude;

namespace {
using Particle = tagged_struct<
    tag_bind<property::position, Eigen::Vector3d>,
    tag_bind<property::velocity, Eigen::Vector3d>,
    tag_bind<property::internal_energy, double>,
    tag_bind<property::random_seed, uint32_t>
>;
using Channel = N2_dsmc_model::col_N2_to_N2_vhs;
constexpr double mass = 4.652e-26;
constexpr double boltzmann = 1.380649e-23;
constexpr double density_value = 5e21;
constexpr double beam_speed = 500;
constexpr double coarse_dt = 1e-8;
constexpr double temperature_scale = mass * beam_speed * beam_speed / boltzmann;
constexpr std::array checkpoints{0, 10, 25, 50, 100, 200, 400};
constexpr int bins = 31;
constexpr double bin_width = 100;
constexpr std::array axes{"x", "y", "z"};
constexpr std::array methods{"reference", "dpmcc", "dpmcc_half_dt"};

struct Snapshot {
    std::array<double, 4> temperature{}; // Three velocity variances and rotational temperature.
    std::array<std::array<double, bins>, 3> histogram{};
};

void accumulate(Snapshot& total, const std::vector<Particle>& particles, int repeats) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (const auto& p : particles) mean += get<property::velocity>(p);
    mean /= double(particles.size());
    const double weight = 1. / (particles.size() * repeats);
    for (const auto& p : particles) {
        const auto& v = get<property::velocity>(p);
        const double internal = get<property::internal_energy>(p);
        if (!v.allFinite() || !std::isfinite(internal) || internal < 0)
            throw std::runtime_error("Invalid particle state in relaxation trace");
        for (int axis = 0; axis < 3; ++axis) {
            const double centered = v[axis] - mean[axis];
            total.temperature[axis] += weight * mass / boltzmann * centered * centered;
            const double index = std::floor(v[axis] / bin_width + .5) + bins / 2;
            const int bin = int(std::clamp(index, 0., double(bins - 1)));
            total.histogram[axis][bin] += weight;
        }
        total.temperature[3] += weight * internal / boltzmann;
    }
}

using Trace = std::array<Snapshot, checkpoints.size()>;

std::vector<Particle> initial_state(int count, int seed) {
    std::mt19937 rng(seed);
    std::vector<Particle> particles(count);
    for (int i = 0; i < count; ++i) {
        auto& p = particles[i];
        get<property::position>(p).setConstant(.5);
        get<property::velocity>(p) = Eigen::Vector3d(i < count / 2 ? beam_speed : -beam_speed, 0, 0);
        get<property::internal_energy>(p) = 0;
        get<property::random_seed>(p) = rng();
    }
    return particles;
}

double pair_rate(double relative_speed_squared) {
    return n2::vhs_cross_section(relative_speed_squared) * std::sqrt(relative_speed_squared);
}

void reference(Trace& trace, std::vector<Particle> particles, int seed, int repeats) {
    const int count = int(particles.size());
    // Energy bounds g^2 by 4 E / m. The VHS rate increases with g.
    const double majorant = pair_rate(2. * count * beam_speed * beam_speed) * (1. + 1e-9);
    const double proposal_rate = density_value * (count - 1) / 2. * majorant;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> first(0, count - 1), other(0, count - 2);
    std::uniform_real_distribution<double> uniform;
    constexpr int context = 0;
    struct Pair { Particle* first; Particle* second; };
    accumulate(trace[0], particles, repeats);
    for (size_t sample = 1; sample < trace.size(); ++sample) {
        const double interval = (checkpoints[sample] - checkpoints[sample-1]) * coarse_dt;
        std::poisson_distribution<int> proposals(proposal_rate * interval);
        const int attempts = proposals(rng);
        for (int attempt = 0; attempt < attempts; ++attempt) {
            const int i = first(rng), j = other(rng);
            Pair pair{&particles[i], &particles[j + (j >= i)]};
            const auto relative = get<property::velocity>(*pair.first) - get<property::velocity>(*pair.second);
            const double rate = pair_rate(relative.squaredNorm());
            if (rate > majorant) throw std::runtime_error("Reference rate exceeded the majorant");
            if (uniform(rng) * majorant < rate) Channel::pair_collide(pair, context);
        }
        accumulate(trace[sample], particles, repeats);
    }
}

void dpmcc(Trace& trace, const std::vector<Particle>& initial, sycl::queue& q,
           int substeps, int repeats) {
    particle_group<Particle, pos_x_nan_is_invalid> group(q);
    cell_field3D<double> density(q, grid3D({0.,0.,0.}, {1.,1.,1.}, {1,1,1}));
    density.setConstant(density_value);
    group.insert<false>(initial);
    auto collide = N2_dsmc_model::make(group, density);
    accumulate(trace[0], initial, repeats);
    for (size_t sample = 1; sample < trace.size(); ++sample) {
        const int steps = substeps * (checkpoints[sample] - checkpoints[sample-1]);
        for (int i = 0; i < steps; ++i) collide(coarse_dt / substeps);
        accumulate(trace[sample], group.get_content().to_host(), repeats);
    }
}

bool compare(const std::array<Trace, 3>& traces, bool study) {
    const double max_temperature_error = study ? .045 : .075;
    const double max_histogram_distance = study ? .065 : .11;
    bool pass = true;
    for (size_t method = 1; method < traces.size(); ++method) {
        double temperature_error = 0, histogram_distance = 0;
        for (size_t sample = 0; sample < checkpoints.size(); ++sample) {
            const auto& expected = traces[0][sample];
            const auto& measured = traces[method][sample];
            for (size_t i = 0; i < measured.temperature.size(); ++i) {
                const auto error = std::abs(measured.temperature[i] - expected.temperature[i]);
                temperature_error = std::max(temperature_error, error / temperature_scale);
            }
            for (size_t axis = 0; axis < axes.size(); ++axis) {
                double difference = 0;
                for (int bin = 0; bin < bins; ++bin)
                    difference += std::abs(measured.histogram[axis][bin] - expected.histogram[axis][bin]);
                histogram_distance = std::max(histogram_distance, difference / 2.);
            }
        }
        const bool passed = temperature_error <= max_temperature_error &&
                            histogram_distance <= max_histogram_distance;
        std::cout << methods[method] << " max_temperature_error=" << temperature_error
                  << " max_histogram_tv=" << histogram_distance << " pass=" << passed << '\n';
        pass &= passed;
    }
    return pass;
}

void write_results(const std::filesystem::path& output, const std::array<Trace, 3>& traces) {
    if (!std::filesystem::create_directory(output)) throw std::runtime_error("Output directory already exists");
    std::ofstream temperatures(output / "temperatures.csv"), histograms(output / "histograms.csv");
    temperatures.exceptions(std::ios::badbit | std::ios::failbit);
    histograms.exceptions(std::ios::badbit | std::ios::failbit);
    temperatures << std::setprecision(15) << "method,time_s,tx_k,ty_k,tz_k,trot_k\n";
    histograms << std::setprecision(15) << "method,time_s,axis,bin,center_mps,probability\n";
    for (size_t method = 0; method < traces.size(); ++method) {
        for (size_t sample = 0; sample < checkpoints.size(); ++sample) {
            const auto& data = traces[method][sample];
            const double time = checkpoints[sample] * coarse_dt;
            temperatures << methods[method] << ',' << time;
            for (double value : data.temperature) temperatures << ',' << value;
            temperatures << '\n';
            for (size_t axis = 0; axis < axes.size(); ++axis)
                for (int bin = 0; bin < bins; ++bin)
                    histograms << methods[method] << ',' << time << ',' << axes[axis] << ','
                               << bin << ',' << (bin - bins / 2) * bin_width << ','
                               << data.histogram[axis][bin] << '\n';
        }
    }
}
}

int main(int argc, char** argv) {
    const bool study = argc == 3 && std::string_view(argv[1]) == "--study";
    if (argc != 1 && !study) {
        std::cerr << "Usage: test_n2_relaxation [--study NEW_OUTPUT_DIRECTORY]\n";
        return 2;
    }
    try {
        sycl::queue q;
        const int count = study ? 4096 : 2048;
        const int repeats = study ? 6 : 2;
        std::array<Trace, 3> traces{};
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto initial = initial_state(count, 7151 + repeat);
            reference(traces[0], initial, 91801 + repeat, repeats);
            dpmcc(traces[1], initial, q, 1, repeats);
            dpmcc(traces[2], initial, q, 2, repeats);
        }
        std::cout << q.get_device().get_info<sycl::info::device::name>() << " N=" << count
                  << " repeats=" << repeats << '\n';
        if (study) write_results(argv[2], traces);
        return compare(traces, study) ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
