#ifndef PSUM_PARTICLE_COLLISION_COULOMB_COLLISION_SOLVER_HPP
#define PSUM_PARTICLE_COLLISION_COULOMB_COLLISION_SOLVER_HPP

#include <cmath>
#include <vector>
#include <algorithm>

#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include "../field/device_field.hpp"
#include "device_exclusive_scan.hpp"
#include "../random.hpp"
#include "../timer.hpp"

namespace psum {

namespace particle_collision {

    inline constexpr double coulomb_data[61] = {
        100.500833331945,89.6260288267918,79.9338725741669,71.2957555491199,63.5970551868168,56.7356144107331,
        50.620386070291,45.1702248004616,40.3128102720307,35.9836875447076,32.1254117891478,28.6867860288506,
        25.6221817872208,22.8909336251724,20.4567995351288,18.2874800311344,16.3541895536384,14.6312745016094,
        13.0958728224622,11.7276106331936,10.5083317800815,9.42185641084597,8.4537641315958,7.59119569940812,
        6.82266499133022,6.13787304083164,5.52752238530729,4.98314313744729,4.49695404109696,4.06178045487348,
        3.67103360028937,3.31873285302973,2.99954091308376,2.70878583600612,2.44245746646594,2.19717892808232,
        1.97016115527622,1.75914998336237,1.56237340062892,1.3784934409123,1.20656409657113,1.07284174071623,
        0.957409915113605,0.856795635012106,0.768434805135889,0.690376480231811,0.62109756841458,0.559382414073523,
        0.504242062966007,0.45485864676795,0.410546145411193,0.370722104701031,0.33488684815973,0.302607918190841,
        0.273508231652444,0.247256916884656,0.22356211576823,0.202165246324745,0.182836365711845,0.165370373273384,
        0.149583863248078
    };

    inline double solve_A(double s) {
        if (s < 1e-2) {
            return 1.0 / s;
        } else if (s < 1.0) {
            const double si = (std::log10(s) + 2.0) / 0.05;
            const int i = static_cast<int>(si);
            const double w = si - i;
            return (1.0 - w) * coulomb_data[i] + w * coulomb_data[i + 1];
        } else if (s < 3.0) {
            const double si = (s - 1.0) / 0.1;
            const int i = static_cast<int>(si);
            const double w = si - i;
            return (1.0 - w) * coulomb_data[i + 40] + w * coulomb_data[i + 41];
        } else {
            return 3.0 * std::exp(-s);
        }
    }

    template <typename Rander>
    inline void scattering_NanBu(
        Eigen::Vector3d& v1, Eigen::Vector3d& v2,
        double mass1, double mass2, double charge1, double charge2,
        double density, double lambdaD, double dt,
        Rander& R
    ) {
        constexpr double H_BAR = 1.054571817e-34;
        constexpr double EPSILON0 = 8.854187817e-12;

        const auto u = v1 - v2;
        const double un = u.norm();
        if (un <= 1e-10) return;
        const auto u_unit = u / un;
        const double m12 = mass1 * mass2 / (mass1 + mass2);
        const double b_min = std::max(H_BAR / (m12 * un), std::abs(charge1 * charge2) / (un * un * 4 * M_PI * EPSILON0 * m12));
        const double lambda = std::max(lambdaD / b_min, 1.01);
        const double s = std::log(lambda) / 4 / M_PI *
            std::pow((charge1 * charge2) / (EPSILON0 * m12), 2) *
            density * std::pow(un, -3) * dt;

        double cos_theta;
        if (s == 0.0) return;
        else if (s <= 1e-2) {
            cos_theta = std::max(1 + s * std::log(R()), -1.0);
        } else {
            const double A = solve_A(s);
            const double random = R();
            if (A > 0.01) {
                cos_theta = 1.0 + std::log(random + (1.0 - random) * std::exp(-2.0 * A)) / A;
            } else if (A != 0.0) {
                const double center_r = 2 * random - 1;
                cos_theta = center_r - A * (center_r * center_r - 1.0) * 0.5 +
                    std::pow(A, 2) * (std::pow(center_r, 3) - center_r) / 3.0;
            } else {
                cos_theta = 1.0 - 2.0 * random;
            }
        }

        const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
        const Eigen::Vector3d helper = psum::random::RandFunction3D::RandV_spherical(R);
        const Eigen::Vector3d tangent = u_unit.cross(helper).normalized();
        const Eigen::Vector3d u_new = un * (cos_theta * u_unit + sin_theta * tangent);
        const Eigen::Vector3d delta_u = u_new - u;
        v1 += m12 / mass1 * delta_u;
        v2 -= m12 / mass2 * delta_u;
    }

template<typename Particle, template<typename> typename Validator, int Dim>
class coulomb_collision_solver {
    static constexpr double EPSILON0 = 8.854187817e-12;
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;

    using ParticleGroup = psum::particle_container::particle_group<Particle, Validator>;
    using Grid = psum::field::simple_grid<Dim>;
    using pos_tag = psum::tag::property::position;
    using random_seed_tag = psum::tag::property::random_seed;

    sycl::queue q_;
    Grid grid_;
    field::device_field<Dim, field::var_loc::cellCentered, double> lambdaD_;
    size_t num_cells_;
    device_exclusive_scan_context<size_t> scan_ctx_;
    field::device_array<size_t> counters_temp_;

    struct species_data {
        field::device_array<size_t> sorted_indices;
        field::device_array<size_t> cell_offsets;
        field::device_array<size_t> cell_counts;
        field::device_array<uint32_t> particle_cell_index;
        field::device_array<double> density;
        size_t capacity = 0;

        species_data(sycl::queue& q, size_t n_cells, size_t n_particles)
            : sorted_indices(q, std::max(n_particles, size_t(1))),
              cell_offsets(q, n_cells + 1),
              cell_counts(q, n_cells),
              particle_cell_index(q, std::max(n_particles, size_t(1))),
              density(q, n_cells),
              capacity(std::max(n_particles, size_t(1))) {}

        void ensure_capacity(sycl::queue& q, size_t n_particles) {
            if (n_particles <= capacity) return;
            size_t new_cap = std::max(n_particles * 2, size_t(8192));
            sorted_indices = field::device_array<size_t>(q, new_cap);
            particle_cell_index = field::device_array<uint32_t>(q, new_cap);
            capacity = new_cap;
            q.wait();
        }
    };

    std::vector<ParticleGroup*> particles_vec_;
    std::vector<std::unique_ptr<species_data>> cached_species_;

    void build_species_index(species_data& sd, ParticleGroup& pg) {
        auto& data_content = pg.get_content();
        size_t n_data = data_content.size();
        size_t n_valid = pg.size();

        Tic("   build_index_fill_zero")
        sd.cell_counts.fill(0);
        counters_temp_.fill(0);
        q_.wait();
        Toc_("   build_index_fill_zero")

        if (n_data == 0 || n_valid == 0) {
            sd.cell_offsets.fill(0);
            sd.density.fill(0.0);
            q_.wait();
            return;
        }

        sd.ensure_capacity(q_, n_valid);

        Tic("   build_index_count_kernel")
        q_.submit([&](sycl::handler& h) {
            auto data_acc = data_content.get_access(h);
            auto counts_ptr = sd.cell_counts.data();
            auto g = grid_;
            h.parallel_for(sycl::range<1>(n_data), [=](sycl::id<1> idx) {
                Particle& p = data_acc[idx];
                if (!Validator<Particle>::is_valid(p)) return;
                auto pos = psum::tag::get<pos_tag>(p);
                if (g.inGrid(pos)) {
                    uint32_t cid = static_cast<uint32_t>(g.c2i(g.nC(pos)));
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref(counts_ptr[cid]);
                    ref.fetch_add(1);
                }
            });
        }).wait();
        Toc_("   build_index_count_kernel")

        Tic("   build_index_exclusive_scan")
        size_t total = exclusive_scan_device<size_t>(
            scan_ctx_, sd.cell_counts.data(), sd.cell_offsets.data(), num_cells_);

        q_.memcpy(sd.cell_offsets.data() + num_cells_, &total, sizeof(size_t)).wait();
        Toc_("   build_index_exclusive_scan")

        Tic("   build_index_fill_zero2")
        counters_temp_.fill(0);
        q_.wait();
        Toc_("   build_index_fill_zero2")

        Tic("   build_index_sort_kernel")
        q_.submit([&](sycl::handler& h) {
            auto data_acc = data_content.get_access(h);
            auto sorted_acc = sd.sorted_indices.get_access(h);
            auto pci_acc = sd.particle_cell_index.get_access(h);
            auto offsets_ptr = sd.cell_offsets.data();
            auto ctr_ptr = counters_temp_.data();
            auto g = grid_;
            h.parallel_for(sycl::range<1>(n_data), [=](sycl::id<1> idx) {
                Particle& p = data_acc[idx];
                if (!Validator<Particle>::is_valid(p)) return;
                auto pos = psum::tag::get<pos_tag>(p);
                if (g.inGrid(pos)) {
                    uint32_t cid = static_cast<uint32_t>(g.c2i(g.nC(pos)));
                    sycl::atomic_ref<size_t, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> ref(ctr_ptr[cid]);
                    size_t local_idx = ref.fetch_add(1);
                    size_t sorted_pos = offsets_ptr[cid] + local_idx;
                    sorted_acc[sorted_pos] = idx;
                    pci_acc[sorted_pos] = cid;
                }
            });
        }).wait();
        Toc_("   build_index_sort_kernel")
    }

    template<typename Num2Dens>
    void compute_density_on_device(species_data& sd, Num2Dens num2dens) {
        auto* counts_ptr = sd.cell_counts.data();
        auto* dens_ptr = sd.density.data();
        auto nc = num_cells_;
        q_.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(nc), [=](sycl::id<1> i) {
                dens_ptr[i[0]] = num2dens(i[0], counts_ptr[i[0]]);
            });
        }).wait();
    }

    void compute_debye_length_on_device(
        const std::vector<species_data*>& species_arr,
        const std::vector<double>& charges,
        const std::vector<const double*>& temp_dev_ptrs)
    {
        field::device_array<double> lambdaD_inv_sq(q_, num_cells_);
        lambdaD_inv_sq.fill(0.0);
        q_.wait();

        for (size_t sp = 0; sp < species_arr.size(); sp++) {
            double cc = charges[sp] * charges[sp];
            const double* dens_ptr = species_arr[sp]->density.data();
            const double* temp_ptr = temp_dev_ptrs[sp];

            q_.submit([&](sycl::handler& h) {
                auto ldis_ptr = lambdaD_inv_sq.data();
                auto nc = num_cells_;
                h.parallel_for(sycl::range<1>(nc), [=](sycl::id<1> i) {
                    double dens = dens_ptr[i[0]];
                    double T_val = temp_ptr[i[0]];
                    if (T_val > 0 && std::isfinite(dens) && dens > 0) {
                        double lbd_i = dens * cc / (EPSILON0 * ELECTRON_CHARGE * T_val);
                        if (std::isfinite(lbd_i))
                            ldis_ptr[i[0]] += lbd_i;
                    }
                });
            });
        }
        q_.wait();

        q_.submit([&](sycl::handler& h) {
            auto ld_ptr = lambdaD_.data();
            auto ldis_ptr = lambdaD_inv_sq.data();
            auto nc = num_cells_;
            h.parallel_for(sycl::range<1>(nc), [=](sycl::id<1> i) {
                double val = ldis_ptr[i[0]];
                ld_ptr[i[0]] = (val == 0.0) ? NAN : std::sqrt(1.0 / val);
            });
        }).wait();
    }

public:
    auto& lambdaD() { return lambdaD_; }

    coulomb_collision_solver(sycl::queue& q, const Grid& grid,
                             const std::vector<ParticleGroup*>& species)
        : q_(q), grid_(grid), lambdaD_(q, grid),
          num_cells_(grid.contentSize(field::var_loc::cellCentered)),
          scan_ctx_(), counters_temp_(q, grid.contentSize(field::var_loc::cellCentered)) {
        scan_ctx_.init(q_);
        for (auto* sp : species) {
            particles_vec_.push_back(sp);
        }
        for (size_t i = 0; i < particles_vec_.size(); i++) {
            cached_species_.push_back(std::make_unique<species_data>(
                q_, num_cells_, particles_vec_[i]->size()));
        }
    }

    coulomb_collision_solver(const coulomb_collision_solver&) = delete;
    coulomb_collision_solver& operator=(const coulomb_collision_solver&) = delete;

    template<typename P2V, typename V2P>
    double collide_in_species(
        ParticleGroup& pg, species_data& sd,
        double mass, double charge, double dt,
        P2V func, V2P v2p
    ) {
        size_t n = pg.size();
        if (n <= 1) return 0.0;

        auto& data_content = pg.get_content();

        q_.submit([&](sycl::handler& h) {
            auto data_acc = data_content.get_access(h);
            auto sorted_acc = sd.sorted_indices.get_access(h);
            auto pci_acc = sd.particle_cell_index.get_access(h);
            auto offset_ptr = sd.cell_offsets.data();
            auto count_ptr = sd.cell_counts.data();
            auto dens_ptr = sd.density.get_access(h);
            auto ld_ptr = lambdaD_.data();

            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> item) {
                size_t i = item[0];
                uint32_t cell_id = pci_acc[i];
                size_t pnum = count_ptr[cell_id];
                if (pnum <= 1) return;
                bool mode = (pnum % 2 == 0);
                size_t num_collision = pnum / 2;
                size_t j = i - offset_ptr[cell_id];
                if (j >= num_collision) return;
                size_t offset = offset_ptr[cell_id];
                double density = dens_ptr[cell_id];
                double ld = ld_ptr[cell_id];

                Particle *p1, *p2;
                if (mode) {
                    p1 = &data_acc[sorted_acc[offset + j * 2]];
                    p2 = &data_acc[sorted_acc[offset + j * 2 + 1]];
                } else {
                    p1 = &data_acc[sorted_acc[offset + j * 2 + 1]];
                    p2 = &data_acc[sorted_acc[offset + j * 2 + 2]];
                }

                if (mode || j >= 1) {
                    auto v1 = func(*p1);
                    auto v2 = func(*p2);
                    psum::random::rander R(psum::tag::get<random_seed_tag>(*p1));
                    scattering_NanBu(v1, v2, mass, mass, charge, charge,
                                     density, ld, dt, R);
                    v2p(*p1, v1);
                    v2p(*p2, v2);
                    psum::tag::get<random_seed_tag>(*p1) = R.get_engine()();
                } else {
                    for (int k = 0; k < 3; k++) {
                        p1 = &data_acc[sorted_acc[offset + k]];
                        p2 = &data_acc[sorted_acc[offset + (k + 1) % 3]];
                        auto v1 = func(*p1);
                        auto v2 = func(*p2);
                        psum::random::rander R(psum::tag::get<random_seed_tag>(*p1));
                        scattering_NanBu(v1, v2, mass, mass, charge, charge,
                                         density, ld, dt/2, R);
                        v2p(*p1, v1);
                        v2p(*p2, v2);
                        psum::tag::get<random_seed_tag>(*p1) = R.get_engine()();
                    }
                }
            });
        }).wait();
        return 0.0;
    }

    template<typename P2V, typename V2P>
    double collide_between_species(
        ParticleGroup& pg0, ParticleGroup& pg1,
        species_data& sd0, species_data& sd1,
        double mass0, double charge0, double mass1, double charge1,
        double dt, P2V func, V2P v2p
    ) {
        auto& data0 = pg0.get_content();
        auto& data1 = pg1.get_content();
        size_t n0 = pg0.size();
        if (n0 == 0) return 0.0;

        q_.submit([&](sycl::handler& h) {
            auto d0 = data0.get_access(h);
            auto d1 = data1.get_access(h);
            auto s0 = sd0.sorted_indices.get_access(h);
            auto s1 = sd1.sorted_indices.get_access(h);
            auto pci0 = sd0.particle_cell_index.get_access(h);
            auto o0 = sd0.cell_offsets.data();
            auto o1 = sd1.cell_offsets.data();
            auto c0 = sd0.cell_counts.data();
            auto c1 = sd1.cell_counts.data();
            auto dens0_ptr = sd0.density.data();
            auto dens1_ptr = sd1.density.data();
            auto ld_ptr = lambdaD_.data();
            auto nc = num_cells_;

            h.parallel_for(sycl::range<1>(n0), [=](sycl::id<1> item) {
                size_t i = item[0];
                if (i >= o0[nc]) return;

                uint32_t cid = pci0[i];
                size_t pnum0 = c0[cid];
                size_t pnum1 = c1[cid];
                if (pnum0 == 0 || pnum1 == 0) return;

                size_t off0 = o0[cid];
                size_t off1 = o1[cid];
                size_t j = i - off0;
                if (j >= pnum0 || j >= pnum1) return;

                double density = (dens0_ptr[cid] < dens1_ptr[cid])
                                 ? dens0_ptr[cid] : dens1_ptr[cid];
                double ld = ld_ptr[cid];

                if (pnum0 >= pnum1) {
                    size_t num_rounds = pnum0 / pnum1;
                    for (size_t round = 0; round <= num_rounds; round++){
                        size_t idx0 = j + round * pnum1;
                        if (idx0 >= pnum0) return;
                        Particle& pA = d0[s0[off0 + idx0]];
                        Particle& pB = d1[s1[off1 + j]];

                        auto v1 = func(pA);
                        auto v2 = func(pB);

                        psum::random::rander R(psum::tag::get<random_seed_tag>(pA));
                        scattering_NanBu(v1, v2, mass0, mass1, charge0, charge1,
                                        density, ld, dt, R);
                        psum::tag::get<random_seed_tag>(pA) = R.get_engine()();

                        v2p(pA, v1);
                        v2p(pB, v2);
                    }
                } else {
                    size_t num_rounds = pnum1 / pnum0;
                    for (size_t round = 0; round <= num_rounds; round++) {
                        size_t idx1 = j + round * pnum0;
                        if (idx1 >= pnum1) break;

                        Particle& pA = d0[s0[off0 + j]];
                        Particle& pB = d1[s1[off1 + idx1]];

                        auto v1 = func(pA);
                        auto v2 = func(pB);
                        psum::random::rander R(psum::tag::get<random_seed_tag>(pA));
                        scattering_NanBu(v1, v2, mass0, mass1, charge0, charge1,
                                         density, ld, dt, R);
                        psum::tag::get<random_seed_tag>(pA) = R.get_engine()();
                        v2p(pA, v1);
                        v2p(pB, v2);
                    }
                }
            });
        }).wait();
        return 0.0;
    }

    template<typename P2V, typename V2P, typename Num2Dens>
    double solve(
        double dt,
        Num2Dens num2dens,
        const std::vector<const double*>& temp_dev_ptrs,
        const std::vector<double>& masses,
        const std::vector<double>& charges,
        P2V func, V2P v2p
    ) {
        size_t n_species = particles_vec_.size();
        if (n_species != masses.size() || n_species != charges.size() || n_species != temp_dev_ptrs.size()) {
            throw std::runtime_error("Error: species count mismatch in coulomb_collision_solver::solve");
        }

        std::vector<species_data*> species_ptrs;

        Tic(" solve_total")
        for (size_t sp = 0; sp < n_species; sp++) {
            size_t n_valid = particles_vec_[sp]->size();
            if (cached_species_[sp]->capacity < n_valid + 16) {
                cached_species_[sp] = std::make_unique<species_data>(q_, num_cells_, n_valid);
            }
            species_data* sd = cached_species_[sp].get();
            species_ptrs.push_back(sd);

            Tic("  solve_build_index")
            build_species_index(*sd, *particles_vec_[sp]);
            Toc_("  solve_build_index")

            Tic("  solve_density")
            compute_density_on_device(*sd, num2dens);
            Toc_("  solve_density")
        }

        Tic("  solve_debye")
        compute_debye_length_on_device(species_ptrs, charges, temp_dev_ptrs);
        Toc_("  solve_debye")

        Tic("  solve_collide")
        for (size_t i = 0; i < n_species; i++) {
            double mass = masses[i];
            double charge = charges[i];
            Tic("   solve_collide_species")
            collide_in_species(*particles_vec_[i], *species_ptrs[i],
                               mass, charge, dt, func, v2p);
            Toc
            for (size_t j = i + 1; j < n_species; j++) {
                double mass1 = masses[j];
                double charge1 = charges[j];
                Tic("   solve_collide_between_species")
                collide_between_species(*particles_vec_[i], *particles_vec_[j],
                                        *species_ptrs[i], *species_ptrs[j],
                                        mass, charge, mass1, charge1,
                                        dt, func, v2p);
                Toc
            }
        }
        Toc_("  solve_collide")
        Toc_(" solve_total")

        return 0.0;
    }
};

}

}
#endif