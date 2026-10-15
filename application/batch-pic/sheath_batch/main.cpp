#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <algorithm>

#include <psum/psum.hpp>
#include <psum/batch_pic.hpp>

using namespace psum::prelude;
using namespace psum::batch_pic;
using namespace boundary_creator;
using property::position;
using property::velocity;
using property::case_id;
using namespace std;

using Particle = tagged_struct<
    tag_bind<position, Eigen::Vector<double, 1>>,
    tag_bind<velocity, Eigen::Vector<double, 1>>,
    tag_bind<case_id, int>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

using BNf = batch_node_field1D<double>;
using BHNf = batch_host_node_field1D<double>;

struct CaseParam {
    double dens;     // injected plasma density [m^-3]
    double Te_ev;
    double Ti_ev;
    double phi_R;    // right (wall) Dirichlet potential [V]
};

// merged injection for ALL cases: host-side RNG build, then ONE device
// insert per species covering every case
void init_particles(ParticleGroup& ps, const std::vector<CaseParam>& params, double weight, double dt, double mass_ele, double mass_ar, grid1D& c_domain, bool ion) {
    Tic("inj_build")
    // per-case RNG streams, built in parallel; the streams do not need to
    // match the serial order
    std::vector<std::vector<Particle>> per_case(params.size());
    unsigned nthreads = std::max(1u, std::min<unsigned>(params.size(), std::thread::hardware_concurrency()));
    auto case_body = [&](size_t k) {
        rander R(10007 * (int)k + (ion ? 1 : 2));
        auto& inj_list = per_case[k];
        inj_list.reserve(8);
        double dens = params[k].dens;
        double mass = ion ? mass_ar : mass_ele;
        double T = (ion ? params[k].Ti_ev : params[k].Te_ev) * 11700;
        double Num_to_deal = (dens * c_domain.del<0>() / weight);
        while (true) {
            auto v = psum::random::RandFunction3D::RandV_Maxwell(R, T, mass).x();
            double x_ = c_domain.nodePosition({c_domain.numCells<0>()}).x() + c_domain.del<0>() * R();
            if (v * dt + x_ < c_domain.nodePosition({c_domain.numCells<0>()}).x())
            {
                Particle p;
                get<position>(p).x() = v * dt + x_;
                get<velocity>(p).x() = v;
                get<case_id>(p) = k;
                inj_list.push_back(p);
            }
            Num_to_deal--;
            if((R() < Num_to_deal)==false) break;
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned t = 0; t < nthreads; t++)
        pool.emplace_back([&](unsigned t) {
            for (size_t k = t; k < params.size(); k += nthreads) case_body(k);
        }, t);
    for (auto& th : pool) th.join();
    size_t total = 0;
    for (auto& v : per_case) total += v.size();
    std::vector<Particle> inj_list;
    inj_list.reserve(total);
    for (auto& v : per_case) inj_list.insert(inj_list.end(), v.begin(), v.end());
    TocTic("inj_insert")
    ps.insert(inj_list);
    Toc
}

void particle_count(ParticleGroup& particles, BNf& dens, double weight) {
    dens.setZero();
    particles.for_each([&](sycl::handler& h) {
        auto dens_acc = dens.get_access(h);
        return [=](const Particle& p) {
            add_back(get<position>(p), weight, get<case_id>(p), dens_acc);
        };
    });
}

void push_particles(ParticleGroup& particles, BNf& phi, double q_m, double dt, double length) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        return [=](Particle& p) {
            auto [dPhi_dx] = interp_diff(get<position>(p), get<case_id>(p), phi_acc);
            double ex = -dPhi_dx;
            get<velocity>(p).x() += ex * q_m * dt;
            get<position>(p).x() += get<velocity>(p).x() * dt;
            double& new_pos = get<position>(p).x();
            if (new_pos < 0 || new_pos >= length) {
                ParticleGroup::validator::make_invalid(p);
            }
        };
    });
}

std::vector<CaseParam> load_cases(const std::string& json_fname) {
    json_loader jl;
    if (!jl.load_json(json_fname))
        throw std::runtime_error("json_loader reported type errors: " + json_fname);
    std::vector<CaseParam> params;
    const size_t n_cases = jl.in("cases").idxs().size();
    for (int idx = 0; idx < n_cases; idx++) {
        CaseParam p;
        auto cc = jl.in("cases").at(idx);
        cc.set("dens", p.dens);
        cc.set("Te_ev", p.Te_ev);
        cc.set("Ti_ev", p.Ti_ev);
        cc.set("phi_R", p.phi_R);
        params.push_back(p);
    }
    if (params.empty())
        throw std::runtime_error("no cases found in " + json_fname + " (expected a non-empty 'cases' array)");
    return params;
}

int main(int argc, char** argv) {
    int total_iters = 200000;
    if (argc > 1) total_iters = atoi(argv[1]);
    // quiet: no console/file io in the loop, for clean performance timings
    bool quiet = (argc > 2 && string(argv[2]) == "quiet");
    std::string config = (argc > 3) ? argv[3] : "sheath_batch.json";
    const std::vector<CaseParam> params = load_cases(config);
    const size_t n_cases = params.size();

    double dx = 0.04/100;
    double dt = 1.3e-10;
    int I = 100;
    double length = dx * I;

    sycl::queue q{sycl::default_selector()};
    grid1D grid({0.0}, {length}, {I});

    BNf ele_dens(q, grid, n_cases), ion_dens(q, grid, n_cases), phi(q, grid, n_cases);
    BHNf ele_dens_host(grid, n_cases), ion_dens_host(grid, n_cases), phi_host(grid, n_cases), qdens_host(grid, n_cases);
    const size_t N_nodes = grid.contentSize(var_loc::nodeCentered);
    std::vector<double> source_packing(N_nodes * n_cases), phi_packing(N_nodes * n_cases);

    ParticleGroup ele(q, 512);
    ParticleGroup ion(q, 512);

    batch_poisson_solver_1d solver(n_cases);
    for (size_t c = 0; c < n_cases; c++) {
        solver.init(c, grid,
            {},
            {
                Dirichlet_point(grid, boundary_direction_1d::L) = 0.0,
                Dirichlet_point(grid, boundary_direction_1d::R) = params[c].phi_R
            }
        );
    }
    solver.init_batch();

    double mass_ele = 9.1e-31;
    double mass_ar = mass_ele * 39.9 * 1836;
    double weight = 6250000;

    if (!quiet) { ofstream fp("numline_xy.plt"); fp.close(); }

    for (int iter = 0; iter < total_iters; iter++) {
        Tic("deposit")
        particle_count(ele, ele_dens, weight);
        particle_count(ion, ion_dens, weight);
        TocTic("solve")
        batch_copy(ele_dens, ele_dens_host);
        batch_copy(ion_dens, ion_dens_host);
        #pragma omp parallel for
        for (size_t c = 0; c < n_cases; c++) {
            double volume = dx;
            for (size_t i = 0; i < qdens_host.at(c).size(); i++) {
                ele_dens_host.at(c)(i) = ele_dens_host.at(c)(i) / volume;
                ion_dens_host.at(c)(i) = ion_dens_host.at(c)(i) / volume;
                qdens_host.at(c)(i) = (ion_dens_host.at(c)(i) - ele_dens_host.at(c)(i)) * 1.602e-19;
            }
        }
        packing(qdens_host, source_packing);
        solver.solve(phi_packing.data(), source_packing.data());
        unpacking(phi_packing, phi_host);
        batch_copy(phi_host, phi);
        TocTic("push")
        push_particles(ele, phi, -1.602e-19 / mass_ele, dt, length);
        push_particles(ion, phi, 1.602e-19 / mass_ar, dt, length);
        Toc

        init_particles(ion, params, weight, dt, mass_ele, mass_ar, grid, true);
        init_particles(ele, params, weight, dt, mass_ele, mass_ar, grid, false);

        if (iter % 50 == 0) {
            Tic("compress")
            if(ele.size() < 0.8 * ele.get_content().size() && ele.size() > 2000)
                ele.compress();
            if(ion.size() < 0.8 * ion.get_content().size() && ion.size() > 2000)
                ion.compress();
            Toc

            if (!quiet) {
                cout << iter << ",\t" << ele.size() << ",\t" << ion.size() << endl;
                ofstream fp("numline_xy.plt", ios::app);
                fp << iter << "," << ele.size() << "," << ion.size() << endl;
                fp.close();
            }
        }
        if (!quiet && iter % 5000 == 0) {
            phi_host.plot("output/c%d_" + to_string(iter) + "Phi_xy.plt", "phi", iter * dt);
            ele_dens_host.plot("output/c%d_" + to_string(iter) + "edens_xy.plt", "ne", iter * dt);
            ion_dens_host.plot("output/c%d_" + to_string(iter) + "idens_xy.plt", "ni", iter * dt);
        }
    }
    printf("# %d iters, %zu cases, particles ele=%zu ion=%zu\n",
           total_iters, n_cases, ele.size(), ion.size());
    PrintTimer
    return 0;
}