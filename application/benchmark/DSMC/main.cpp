#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <filesystem>
#include <optional>
#include <cstdlib>
#include <string_view>
#include <psum/psum.hpp>
#include "custom_dpmcc.hpp"
#include "custom_material.hpp"
#include "probe_sampler.hpp"

using namespace psum::prelude;
using property::position;
using property::velocity;
using property::internal_energy;
using property::random_seed;

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector3d>,
    tag_bind<velocity, Eigen::Vector3d>,
    tag_bind<internal_energy, double>,
    tag_bind<random_seed, uint32_t>
>;

using ParticleGroup = particle_container::particle_group<Particle, particle_container::pos_x_nan_is_invalid>;

using Field = field::cell_field3D<double>;

using MaterialSet = simple_material_set<Particle, pos_x_nan_is_invalid>;
using material_type = MaterialSet::material_type;
constexpr double half_height = 0.0025;
constexpr double plate_left = 0.050;
constexpr double plate_length = 0.1;
constexpr double T_plate = 290.0;
constexpr double pmass = 4.652E-26;

void compute_density_distribution(ParticleGroup& N2, Field& dens, double weight) {
    dens.setZero();
    double cell_v = dens.getGrid().del<0>() * dens.getGrid().del<1>() * dens.getGrid().del<2>();
    N2.for_each([&](sycl::handler& h) {
        auto dens_acc = dens.get_access(h);
        return [=](Particle& p) {
            if (dens_acc.getGrid().inGrid(get<position>(p)))
                add_back_nearest(get<position>(p), weight / cell_v, dens_acc);
            else
                ParticleGroup::validator::make_invalid(p);
        };
    });
}

#ifdef N2_CUDA_MEMORY
#include <cuda_runtime_api.h>

void record_gpu_memory(std::ofstream& out, int step) {
    size_t free_bytes, total_bytes;
    const auto error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    out << step << ',' << total_bytes - free_bytes << ',' << total_bytes << '\n';
}
#endif

#ifdef N2_ANALYTIC_BOUNDARY
class Boundary {
    MaterialSet::acc_type wall{T_plate, pmass, 0.8};
    double xmax, ymax, dz;
    static constexpr double plate_right = plate_left + plate_length;

public:
    Boundary(sycl::queue&, const grid3D& g)
        : xmax(g.upperBound<0>()), ymax(g.upperBound<1>()), dz(g.upperBound<2>()) {}

    Boundary get_access(sycl::handler&) const { return *this; }

    void operator()(const Eigen::Vector3d& start, const Eigen::Vector3d& end, Particle& p) const {
        // All physical boundaries are axis-aligned. Test swept segments,
        // keeping the first hit (the router also handles one impact per step).
        double first = 2.0;
        int axis = 0;
        material_type material = material_type::absorb;
        auto face = [&](int dim, double plane, double lo, double hi,
                        double zlo, double zhi, material_type kind) {
            if (!((start[dim] < plane && end[dim] >= plane) ||
                  (start[dim] > plane && end[dim] <= plane))) return;
            const double t = (plane - start[dim]) / (end[dim] - start[dim]);
            if (t >= first) return;
            const int other = 1 - dim;
            const double lateral = start[other] + t * (end[other] - start[other]);
            const double z = start.z() + t * (end.z() - start.z());
            if (lateral >= lo && lateral <= hi && z >= zlo && z <= zhi) {
                first = t;
                axis = dim;
                material = kind;
            }
        };
        if (end.x() <= 0 || end.x() >= xmax)
            face(0, end.x() <= 0 ? 0 : xmax, -ymax, ymax, 0, dz, material_type::absorb);
        if (end.y() <= -ymax || end.y() >= ymax)
            face(1, end.y() <= -ymax ? -ymax : ymax, 0, xmax, 0, dz, material_type::specular);
        if (sycl::fmax(start.x(), end.x()) >= plate_left &&
            sycl::fmin(start.x(), end.x()) <= plate_right &&
            sycl::fmax(start.y(), end.y()) >= -half_height &&
            sycl::fmin(start.y(), end.y()) <= half_height) {
            face(0, plate_left, -half_height, half_height, -1, 1, material_type::maxwell_reflect);
            face(0, plate_right, -half_height, half_height, -1, 1, material_type::maxwell_reflect);
            face(1, -half_height, plate_left, plate_right, -1, 1, material_type::maxwell_reflect);
            face(1, half_height, plate_left, plate_right, -1, 1, material_type::maxwell_reflect);
        }
        if (first <= 1) {
            const Eigen::Vector3d hit = start + first * (end - start);
            wall.action(material, p, hit.x(), hit.y(), hit.z(), axis == 0, axis == 1, 0);
        }
    }
};
#else
class Boundary {
    boundary_router<MaterialSet, Particle> router;

public:
    Boundary(sycl::queue& q, const grid3D& g) : router(q) {
        auto leave_surf_x_lower = mesh_generator::axis_aligned_plane(
            g.lowerBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
            g.lowerBound<0>(), g.upperBound<1>(), g.upperBound<2>()
        );

        auto leave_surf_x_upper = mesh_generator::axis_aligned_plane(
            g.upperBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
            g.upperBound<0>(), g.upperBound<1>(), g.upperBound<2>()
        );

        auto leave_surf_y_lower = mesh_generator::axis_aligned_plane(
            g.lowerBound<0>(), g.lowerBound<1>(), g.lowerBound<2>(),
            g.upperBound<0>(), g.lowerBound<1>(), g.upperBound<2>()
        );

        auto leave_surf_y_upper = mesh_generator::axis_aligned_plane(
            g.lowerBound<0>(), g.upperBound<1>(), g.lowerBound<2>(),
            g.upperBound<0>(), g.upperBound<1>(), g.upperBound<2>()
        );

        auto truncate_surf = mesh_generator::axis_aligned_cube(
            plate_left, -half_height, -1,
            plate_left + plate_length, half_height, 1
        );

        std::vector<triangle_mesh_with_id<material_type>> meshes = {
            {leave_surf_x_lower, material_type::absorb},
            {leave_surf_x_upper, material_type::absorb},
            {leave_surf_y_lower, material_type::specular},
            {leave_surf_y_upper, material_type::specular},
            {truncate_surf, material_type::maxwell_reflect}
        };

        router.set(meshes, 80, 80, 4);
        router.get_material_set().set_default_temperature(T_plate);
        router.get_material_set().set_default_mass(pmass);
    }

    auto get_access(sycl::handler& h) {
        auto acc = router.get_access(h);
        return [=](const Eigen::Vector3d& start, const Eigen::Vector3d& end, Particle& p) {
            acc.deal(start.x(), start.y(), start.z(), end.x(), end.y(), end.z(), p);
        };
    }
};
#endif

void run_n2_flat_plate(int max_iterations = 50000, unsigned seed = 20261021,
                       double weight = 1e10, const std::filesystem::path& output_dir = "output") {
    constexpr double dt = 2e-8;
    const char* setting = std::getenv("N2_DIAGNOSTICS");
    const bool diagnostics = !setting || std::string_view(setting) != "0";
    sycl::queue q{sycl::default_selector{}};
    std::cout << "Running on " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    ParticleGroup N2(q);

    int I = 350;
    int J = 150;
    double dx = 0.5e-3;
    double dz = 5e-3;

    double boltz = 1.380649e-23;
    double dens = 3.716E20;
    double T = 13.32;
    double v_inf = 1503;

    grid3D g({0, -J * dx, 0}, {I * dx, J * dx, dz}, {I, J * 2, 1});
    Field f_N2(q, g);
    auto model = N2_dsmc_model::make(N2, f_N2);
    Boundary boundary(q, g);

    std::optional<N2ProbeSampler> probes;
#ifdef N2_CUDA_MEMORY
    std::ofstream memory;
#endif
    if (diagnostics) {
        std::filesystem::create_directories(output_dir);
        probes.emplace(q, output_dir, dt);
#ifdef N2_CUDA_MEMORY
        if (q.get_device().get_backend() == sycl::backend::cuda) {
            cudaPointerAttributes attributes{};
            auto error = cudaPointerGetAttributes(&attributes, f_N2.getContent().data());
            if (error == cudaSuccess) error = cudaSetDevice(attributes.device);
            if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
            memory.open(output_dir / "gpu-memory.csv");
            if (!memory) throw std::runtime_error("cannot create GPU memory diagnostics");
            memory << "step,used_bytes,total_bytes\n";
            record_gpu_memory(memory, 0);
        }
#endif
    }

    const int inlet_cells = 2 * J;
    const double scale = sqrt(2 * boltz * T / pmass);
    const double s = v_inf / scale;
    const double root = sqrt(s * s + 2);
    const double pi = acos(-1.0);
    const double flux = scale * (exp(-s * s) + sqrt(pi) * s * (1 + erf(s))) / (2 * sqrt(pi));
    const double particles_per_cell = dens * dx * dz * flux * dt / weight;
    const auto uniform = [](std::mt19937& rng) { return (double(rng()) + 0.5) / 4294967296.0; };

    std::mt19937 count_rng(seed);
    std::vector<std::mt19937> cell_rng;
    cell_rng.reserve(inlet_cells);
    for (int cell = 0; cell < inlet_cells; ++cell) {
        std::seed_seq stream{seed, static_cast<unsigned>(cell)};
        cell_rng.emplace_back(stream);
    }
    std::vector<std::size_t> offsets(inlet_cells + 1);
    std::vector<Particle> new_particles;

    q.wait_and_throw();
    const auto start_time = std::chrono::steady_clock::now();
    for (int iter = 1; iter <= max_iterations; ++iter) {
        N2.for_each([&](sycl::handler& h) {
            auto hit_wall = boundary.get_access(h);
            return [=](Particle& p) {
                const auto start = get<position>(p);
                get<position>(p) += get<velocity>(p) * dt;
                hit_wall(start, get<position>(p), p);
                g.wrap_on_dim<2>(get<position>(p));
            };
        });

        // Round each cell independently and give it a disjoint output range.
        for (int cell = 0; cell < inlet_cells; ++cell) {
            const auto count = static_cast<std::size_t>(particles_per_cell + uniform(count_rng));
            offsets[cell + 1] = offsets[cell] + count;
        }
        new_particles.resize(offsets.back());
        bool reaches_plate = false;
        #pragma omp parallel for schedule(static) reduction(|| : reaches_plate)
        for (int cell = 0; cell < inlet_cells; ++cell) {
            auto& rng = cell_rng[cell];
            for (auto i = offsets[cell]; i < offsets[cell + 1]; ++i) {
                auto& p = new_particles[i];
                double beta, accept;
                do {
                    do {
                        beta = 6 * uniform(rng) - 3;
                    } while (beta + s < 0);
                    accept = 2 * (beta + s) / (s + root)
                           * exp(0.5 + 0.5 * s * (s - root) - beta * beta);
                } while (accept < uniform(rng));

                const double angle = 2 * pi * uniform(rng);
                const double radius = scale * sqrt(-log(uniform(rng)));
                get<velocity>(p) = {v_inf + scale * beta, radius * sin(angle), radius * cos(angle)};
                get<position>(p) = {0, (cell + uniform(rng)) * dx + g.lowerBound<1>(), uniform(rng) * dz};
                get<position>(p) += get<velocity>(p) * (dt * uniform(rng));
                if (get<position>(p).x() >= plate_left) reaches_plate = true;
                for (int side : {-1, 1}) {
                    if (side * get<position>(p).y() > J * dx) {
                        get<position>(p).y() = 2 * side * J * dx - get<position>(p).y();
                        get<velocity>(p).y() *= -1;
                    }
                }
                g.wrap_on_dim<2>(get<position>(p));
                get<random_seed>(p) = rng();
                get<internal_energy>(p) = -boltz * T * log(uniform(rng));
            }
        }
        if (reaches_plate) throw std::runtime_error("inlet flight reaches plate");
        N2.insert<false>(new_particles);

        compute_density_distribution(N2, f_N2, weight);
        model(dt);

        if (probes && iter % probes->sample_steps == 0) probes->sample(iter, N2);

#ifdef N2_CUDA_MEMORY
        if (memory.is_open() && (iter % 100 == 0 || iter == max_iterations))
            record_gpu_memory(memory, iter);
#endif
    }
    q.wait_and_throw();
    if (!diagnostics) {
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_time;
        std::cout << "elapsed_seconds " << elapsed.count() << '\n';
    }
}

#ifndef N2_CASE_ARGUMENTS
#define N2_CASE_ARGUMENTS
#endif

int main() {
    try {
        run_n2_flat_plate(N2_CASE_ARGUMENTS);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "n2-flat-plate: " << error.what() << '\n';
        return 2;
    }
}
