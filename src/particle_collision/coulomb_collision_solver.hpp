#ifndef PSUM_PARTICLE_COLLISION_COULOMB_COLLISION_SOLVER_HPP
#define PSUM_PARTICLE_COLLISION_COULOMB_COLLISION_SOLVER_HPP

#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <vector>
#include <algorithm>

#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include "../../include/psum/particle_container.hpp"
#include "../field/device_field.hpp"
#include <AdaptiveCpp/algorithms/numeric.hpp>
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

    inline double coulomb_cross_section_limit(double max_density) {
        if (max_density == 0.0) return 0.0;
        // Limit the mean free path to the interparticle spacing of the denser species.
        return std::cbrt((4.0 * M_PI / 3.0) * max_density) / max_density;
    }

    template <typename Rander>
    inline void scattering_NanBu(
        Eigen::Vector3d& v1, Eigen::Vector3d& v2,
        double mass1, double mass2, double charge1, double charge2,
        double density, double sigma_max, double lambdaD, double dt,
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
        const double s_uncapped = std::log(lambda) / 4 / M_PI *
            std::pow((charge1 * charge2) / (EPSILON0 * m12), 2) *
            density * std::pow(un, -3) * dt;
        const double s = std::min(s_uncapped, sigma_max * density * un * dt);

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

    template <typename Rander>
    inline void scattering_NanBu(
        Eigen::Vector3d& v1, Eigen::Vector3d& v2,
        double mass1, double mass2, double charge1, double charge2,
        double density, double lambdaD, double dt, Rander& random
    ) {
        scattering_NanBu(v1, v2, mass1, mass2, charge1, charge2, density,
                         coulomb_cross_section_limit(density), lambdaD, dt, random);
    }

enum class coulomb_role { electron, ion };

template<typename Particle, template<typename> typename Validator, int Dim>
class coulomb_collision_solver {
    static constexpr double EPSILON0 = 8.854187817e-12;
    static constexpr double ELECTRON_CHARGE = 1.602176634e-19;

    using ParticleGroup = psum::particle_container::particle_group<Particle, Validator>;
    using TemperatureField = field::device_field<Dim, field::var_loc::cellCentered, double>;
    using Grid = psum::field::simple_grid<Dim>;
    using pos_tag = psum::tag::property::position;
    using random_seed_tag = psum::tag::property::random_seed;

    // SplitMix64 finalizer gives successive pool index builds distinct permutation keys.
    static std::uint64_t mix(std::uint64_t x) {
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }

    std::uint64_t pairing_epoch_;
    sycl::queue q_;
    Grid grid_;
    field::device_field<Dim, field::var_loc::cellCentered, double> lambdaD_;
    size_t num_cells_;
    acpp::algorithms::util::allocation_cache scan_cache_{acpp::algorithms::util::allocation_type::device};

public:
    // Pooling assumes equal macro-particle weights across the bound species.
    // The group and cell-centered temperature field (eV) must outlive the solver.
    struct species_binding {
        ParticleGroup& group;
        coulomb_role role;
        double mass, charge; // kg, C
        std::reference_wrapper<const TemperatureField> temperature;
    };

private:
    struct species_view {
        Particle* particles;
        const double* temperature;
        double mass, charge;
        size_t begin, end;
        coulomb_role role;
    };

    struct slot {
        size_t index;
        uint32_t species, cell;
    };

    struct pool {
        field::device_array<slot> slots;
        field::device_array<size_t> counts, offsets;
        field::device_array<double> density;
        size_t extent = 0, indexed = 0;

        pool(sycl::queue& q, size_t cells)
            : slots(q, 1), counts(q, cells + 1), offsets(q, cells + 1), density(q, cells) {}
    };

    std::vector<species_binding> bindings_;
    std::vector<species_view> host_views_;
    field::device_array<species_view> species_;
    field::device_array<size_t> species_counts_;
    field::device_array<double> species_density_;
    pool ions_, electrons_;

    using counter = sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>;

    template<typename Num2Dens>
    void prepare_species(Num2Dens num2dens) {
        ions_.extent = electrons_.extent = 0;
        species_counts_.fill(0);
        const auto grid = grid_;
        for (size_t s = 0; s < bindings_.size(); ++s) {
            const auto& [group, role, mass, charge, temperature] = bindings_[s];
            auto& extent = role == coulomb_role::ion ? ions_.extent : electrons_.extent;
            auto* data = group.get_content().data(); // Includes the host overflow check.
            const size_t n = group.extent();
            host_views_[s] = {data, temperature.get().data(), mass, charge, extent, extent + n, role};
            extent += n;
            if (!n) continue;
            auto* counts = species_counts_.data() + s * num_cells_;
            q_.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                const auto& particle = data[i[0]];
                if (!Validator<Particle>::is_valid(particle)) return;
                const auto pos = psum::tag::get<pos_tag>(particle);
                if (grid.inGrid(pos)) counter(counts[grid.c2i(grid.nC(pos))]).fetch_add(1);
            }).wait();
        }
        species_.copy(host_views_);
        const auto* views = species_.data();
        const auto* counts = species_counts_.data();
        auto* densities = species_density_.data();
        auto* ic = ions_.counts.data();
        auto* ec = electrons_.counts.data();
        auto* idens = ions_.density.data();
        auto* edens = electrons_.density.data();
        auto* ld = lambdaD_.data();
        const size_t cells = num_cells_, ns = bindings_.size();
        q_.parallel_for(sycl::range<1>(cells + 1), [=](sycl::id<1> item) {
            const size_t c = item[0];
            size_t ni = 0, ne = 0;
            double di = 0, de = 0, sum = 0;
            if (c < cells) {
                for (size_t s = 0; s < ns; ++s) {
                    const size_t at = s * cells + c;
                    const double n = num2dens(c, counts[at]);
                    densities[at] = n;
                    if (views[s].role == coulomb_role::ion) { ni += counts[at]; di += n; }
                    else { ne += counts[at]; de += n; }
                    const double t = views[s].temperature[c], q = views[s].charge;
                    if (t > 0 && std::isfinite(n) && n > 0) {
                        const double term = n * (q*q) / (EPSILON0 * ELECTRON_CHARGE * t);
                        if (std::isfinite(term)) sum += term;
                    }
                }
                idens[c] = di;
                edens[c] = de;
                ld[c] = sum == 0 ? NAN : std::sqrt(1.0 / sum);
            }
            ic[c] = ni;
            ec[c] = ne;
        }).wait();
    }

    void build_index(pool& p, coulomb_role role) {
        p.indexed = 0;
        if (!p.extent) return;
        auto* counts = p.counts.data();
        auto* offsets = p.offsets.data();
        acpp::algorithms::util::allocation_group scratch{&scan_cache_, q_.get_device()};
        auto scan = acpp::algorithms::exclusive_scan(q_, scratch, counts, counts + num_cells_ + 1,
                                                    offsets, size_t{0});
        q_.memcpy(&p.indexed, offsets + num_cells_, sizeof(size_t), scan).wait();
        if (!p.indexed) return;
        if (p.indexed > p.slots.size())
            p.slots = field::device_array<slot>(q_, std::max(p.indexed, p.slots.size() + p.slots.size()/2));
        p.counts.fill(0);
        // Consecutive keys can give several rounds the same set of hash inputs
        // for a small right half, biasing adjacent pairs after serial binning.
        const auto seed = mix(++pairing_epoch_);
        const particle_container::detail::slot_permutation order(p.extent,
            {mix(seed), mix(seed + 1), mix(seed + 2), mix(seed + 3)});
        auto* slots = p.slots.data();
        const auto* views = species_.data();
        const auto grid = grid_;
        const size_t ns = bindings_.size();
        q_.parallel_for(sycl::range<1>(p.extent), [=](sycl::id<1> item) {
            const size_t k = order(item[0]);
            for (uint32_t s = 0; s < ns; ++s) {
                const auto& view = views[s];
                // begin/end are offsets in this role's concatenated slot range.
                if (view.role != role || k >= view.end) continue;
                const size_t index = k - view.begin;
                const auto& particle = view.particles[index];
                if (!Validator<Particle>::is_valid(particle)) return;
                const auto pos = psum::tag::get<pos_tag>(particle);
                if (grid.inGrid(pos)) {
                    const auto c = static_cast<uint32_t>(grid.c2i(grid.nC(pos)));
                    slots[offsets[c] + counter(counts[c]).fetch_add(1)] = {index, s, c};
                }
                return;
            }
        }).wait();
    }

    // Both pools use the same within-pool matching. All callbacks see the real species.
    template<bool Within, typename P2V, typename V2P>
    void collide(const pool& first, const pool& second, double dt, P2V read, V2P write,
                 std::optional<sycl::event>& previous) {
        if (!first.indexed || !second.indexed || (Within && first.indexed < 2)) return;
        const auto *a = first.slots.data(), *b = second.slots.data();
        const auto *ao = first.offsets.data(), *bo = second.offsets.data();
        const auto *ac = first.counts.data(), *bc = second.counts.data();
        const auto *ad = first.density.data(), *bd = second.density.data();
        const auto* views = species_.data();
        const auto* densities = species_density_.data();
        const auto* debye = lambdaD_.data();
        const size_t cells = num_cells_;
        const auto key = mix(pairing_epoch_);
        previous = q_.submit([&](sycl::handler& h) {
            if (previous) h.depends_on(*previous);
            h.parallel_for(sycl::range<1>(first.indexed), [=](sycl::id<1> item) {
                const auto c = a[item[0]].cell;
                const size_t j = item[0] - ao[c], na = ac[c], nb = bc[c];
                double density = Within ? ad[c] : std::max(ad[c], bd[c]);
                // Each particle has na - 1 possible partners within its pool in this cell.
                if constexpr (Within) density *= static_cast<double>(na - 1) / na;
                auto scatter = [&](size_t i, size_t k, double step) {
                    const slot x = a[ao[c]+i], y = b[bo[c]+k];
                    const auto& sx = views[x.species];
                    const auto& sy = views[y.species];
                    const double limit = coulomb_cross_section_limit(std::max(
                        densities[x.species*cells+c], densities[y.species*cells+c]));
                    auto& p = sx.particles[x.index];
                    auto& partner = sy.particles[y.index];
                    auto v = read(p), w = read(partner);
                    auto& seed = psum::tag::get<random_seed_tag>(p);
                    psum::random::rander random(seed);
                    scattering_NanBu(v,w,sx.mass,sy.mass,sx.charge,sy.charge,density,limit,debye[c],step,random);
                    if (x.species != y.species) seed = random.get_engine()();
                    write(p,v);
                    write(partner,w);
                    if (x.species == y.species) seed = random.get_engine()();
                };
                if constexpr (Within) {
                    if (j >= na/2) return;
                    if (na%2 && j == 0) {
                        for (size_t k=0; k<3; ++k) scatter(k,(k+1)%3,dt/2);
                    } else {
                        const size_t k = 2*j + na%2;
                        scatter(k,k+1,dt);
                    }
                } else {
                    if (j >= nb) return;
                    // Take one distinct partner per endpoint, even for unequal pool sizes.
                    const auto shift = mix(key ^ mix(c));
                    scatter((j+shift%na)%na,(j+mix(shift)%nb)%nb,dt);
                }
            });
        });
    }

public:
    auto& lambdaD() { return lambdaD_; }

    coulomb_collision_solver(sycl::queue& q, const Grid& grid, const std::vector<species_binding>& species,
                             std::uint64_t pairing_seed = std::random_device{}())
        : pairing_epoch_(mix(pairing_seed)), q_(q), grid_(grid), lambdaD_(q,grid),
          num_cells_(grid.contentSize(field::var_loc::cellCentered)), bindings_(species),
          host_views_(species.size()), species_(q,std::max<size_t>(species.size(),1)),
          species_counts_(q,std::max<size_t>(species.size()*num_cells_,1)),
          species_density_(q,std::max<size_t>(species.size()*num_cells_,1)),
          ions_(q,num_cells_), electrons_(q,num_cells_) {
        for (size_t i=0; i<bindings_.size(); ++i)
            for (size_t j=0; j<i; ++j)
                if (&bindings_[i].group == &bindings_[j].group)
                    throw std::invalid_argument("duplicate particle group in Coulomb solver");
    }

    coulomb_collision_solver(const coulomb_collision_solver&) = delete;
    coulomb_collision_solver& operator=(const coulomb_collision_solver&) = delete;

    template<typename P2V, typename V2P, typename Num2Dens>
    double solve(double dt, Num2Dens num2dens, P2V read, V2P write) {
        if (bindings_.empty()) { lambdaD_.setConstant(NAN); return 0.0; }

        Tic(" solve_total")
        Tic("  solve_density")
        prepare_species(num2dens);
        Toc_("  solve_density")
        Tic("  solve_build_index")
        build_index(ions_,coulomb_role::ion);
        build_index(electrons_,coulomb_role::electron);
        Toc_("  solve_build_index")
        Tic("  solve_collide")
        std::optional<sycl::event> previous;
        collide<true>(ions_,ions_,dt,read,write,previous);
        collide<true>(electrons_,electrons_,dt,read,write,previous);
        collide<false>(electrons_,ions_,dt,read,write,previous);
        if (previous) previous->wait();
        Toc_("  solve_collide")
        Toc_(" solve_total")
        return 0.0;
    }
};
}
}
#endif
