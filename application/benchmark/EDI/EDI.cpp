#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <sycl/sycl.hpp>
#include <iostream>
#include <psum/psum.hpp>

using namespace psum::prelude;
using namespace boundary_creator;
using property::position;
using property::velocity;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector2d>,
    tag_bind<velocity, Eigen::RowVector2d>
>;

using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

using Nf = node_field2D<double>;
using Nf1d = node_field1D<double>;
using Nf1d_host = host_node_field1D<double>;

inline Nf1d_host create_magnetic_field_profile(const grid1D& g) {
    double xmax = 0.025;
    double xBmax = 0.0075;
    double Bmax = 0.01;
    double Bl = 0.006;
    double Br = 0.001;
    double sigma1 = 0.00625;
    double sigma2 = 0.00625;
    double ak1 = (Bmax - Bl) / (1 - std::exp(-0.5 * std::pow(xBmax / sigma1, 2)));
    double ak2 = (Bmax - Br) / (1 - std::exp(-0.5 * std::pow((xmax - xBmax) / sigma2, 2)));
    double bk1 = Bl - std::exp(-0.5 * std::pow(xBmax / sigma1, 2)) * ak1;
    double bk2 = Br - std::exp(-0.5 * std::pow((xmax - xBmax) / sigma1, 2)) * ak1;

    Nf1d_host Br_f(g);
    Br_f.for_each([&](size_t i, double& v, Nf1d_host::Position pos) {
        double x = pos(0);
        if (x < xBmax)
            v = std::exp(-std::pow(x - xBmax, 2) / (2 * sigma1 * sigma1)) * ak1 + bk1;
        else
            v = std::exp(-std::pow(x - xBmax, 2) / (2 * sigma2 * sigma2)) * ak2 + bk2;
    });
    return Br_f;
}

inline void compute_Efield(
    Nf& Exfield,
    Nf& Eyfield,
    Nf& Phi
)
{
    auto grid = Phi.getGrid();
    double span_x = grid.span<0>();
    double span_y = grid.span<1>();

    Exfield.for_each([&](sycl::handler& h) {
        auto Phi_acc = Phi.get_access(h);
        return [=](size_t i, double& v, Nf::Position pos) {
            pos = pos * (1 - 1e-10);
            pos(0) = std::fmod(pos(0) + span_x, span_x);
            auto grad = interp_diff(pos, Phi_acc);
            v = -grad[0];
        };
    });

    Eyfield.for_each([&](sycl::handler& h) {
        auto Phi_acc = Phi.get_access(h);
        return [=](size_t i, double& v, Nf::Position pos) {
            pos = pos * (1 - 1e-10);
            pos(1) = std::fmod(pos(1) + span_y, span_y);
            auto grad = interp_diff(pos, Phi_acc);
            v = -grad[1];
        };
    });
}

inline void compute_rectify_Phi(
    Nf& Phi,
    double rectify_phi
)
{
    Phi.for_each([&](sycl::handler& h) {
        return [=](size_t i, double& v, Nf::Position pos) {
            v -= rectify_phi / 0.024 * pos(0);
        };
    });
}

inline void injParticle_ioni(
    std::vector<Particle>& ele,
    std::vector<Particle>& ion,
    const grid2D& grid,
    double weight,
    double dt
)
{
    ele.clear();
    ion.clear();

    double x1 = 0.0025;
    double x2 = 0.01;
    double xm = 0.00625;
    double S0 = 5.23e23;

    int num_cells_x = grid.numCells<0>();
    for (int i = 0; i < num_cells_x; i++)
    {
        double x = grid.lowerBound<0>() + (i + 0.5) * grid.del<0>();
        double S;
        if (x < x1 || x > x2)
            S = 0;
        else
        {
            S = S0 * std::cos(M_PI * (x - xm) / (x2 - x1));
        }
        double num_inj = S * grid.del<0>() * grid.span<1>() * dt;
        num_inj /= weight;
        rander R;
        int num_insert = std::floor(num_inj) + (R() < num_inj - std::floor(num_inj));
        while (num_insert > 0)
        {
            double pos_x = x + (R() - 0.5) * grid.del<0>();
            double pos_y = grid.span<1>() * R() + grid.lowerBound<1>();

            auto v_ele = RandFunction2D::RandV_Maxwell(R, 10 * 11700, 9.11e-31);
            auto v_ion = RandFunction2D::RandV_Maxwell(R, 0.5 * 11700, 1.66e-27 * 131.29);

            Particle p_ele;
            get<position>(p_ele) = {pos_x, pos_y};
            get<velocity>(p_ele) = v_ele;
            ele.push_back(p_ele);

            Particle p_ion;
            get<position>(p_ion) = {pos_x, pos_y};
            get<velocity>(p_ion) = v_ion;
            ion.push_back(p_ion);
            num_insert--;
        }
    }
}

inline void injParticle_cathode(
    std::vector<Particle>& ele,
    const grid2D& grid,
    double num_in
)
{
    double x_inj = 0.024;
    rander R;
    while (R() < num_in)
    {
        double pos_y = grid.span<1>() * R() + grid.lowerBound<1>();
        auto v_ele = RandFunction2D::RandV_Maxwell(R, 10 * 11700, 9.11e-31);

        Particle p;
        get<position>(p) = {x_inj, pos_y};
        get<velocity>(p) = v_ele;
        ele.push_back(p);
        num_in--;
    }
}

inline void injParticle_init(
    ParticleGroup& ele,
    ParticleGroup& ion,
    const grid2D& grid,
    double weight
)
{
    double num_inj = 5e16 * grid.span<0>() * grid.span<1>();
    num_inj /= weight;

    rander R;
    std::vector<Particle> add_list_ele, add_list_ion;
    while (R() < num_inj)
    {
        double x = grid.span<0>() * R() + grid.lowerBound<0>();
        double y = grid.span<1>() * R() + grid.lowerBound<1>();

        auto v_ele = RandFunction2D::RandV_Maxwell(R, 10 * 11700, 9.11e-31);
        auto v_ion = RandFunction2D::RandV_Maxwell(R, 0.5 * 11700, 1.66e-27 * 131.29);

        Particle p_ele;
        get<position>(p_ele) = {x, y};
        get<velocity>(p_ele) = v_ele;
        add_list_ele.push_back(p_ele);

        Particle p_ion;
        get<position>(p_ion) = {x, y};
        get<velocity>(p_ion) = v_ion;
        add_list_ion.push_back(p_ion);
        num_inj -= 1;
    }

    ele.insert(add_list_ele);
    ion.insert(add_list_ion);
}

inline void particleCount(
    const ParticleGroup& ps,
    Nf& dens,
    double weight
)
{
    dens.setZero();
    ps.for_each([&](sycl::handler& h) {
        auto dens_acc = dens.get_access(h);
        return [weight, dens_acc](const Particle& p) {
            add_back(get<position>(p), weight, dens_acc);
        };
    });
}

inline void particlePush(
    sycl::queue& q,
    ParticleGroup& ps,
    Nf& Phi,
    Nf1d& Br_field,
    double q_mass_rate,
    double dt,
    int* absorbed_counter
)
{
    double xmin = Phi.getGrid().lowerBound<0>();
    double xmax = Phi.getGrid().upperBound<0>();
    double ymin = Phi.getGrid().lowerBound<1>();
    double ymax = Phi.getGrid().upperBound<1>();
    double span_y = ymax - ymin;
    ps.for_each([&](sycl::handler& h) {
        auto Phi_acc = Phi.get_access(h);
        auto Br_acc = Br_field.get_access(h);
        return [=](Particle& p) {
            auto pos_old = get<position>(p);
            if (!Phi_acc.getGrid().inGrid(pos_old)) {
                pos_x_nan_is_invalid<Particle>::make_invalid(p);
                return;
            }
            auto [Ex, Ey] = interp_diff(get<position>(p), Phi_acc);
            double B = interp(Nf1d::Position{get<position>(p).x()}, Br_acc);
            Eigen::RowVector2d Evec = {-Ex, -Ey};
            double tao = B * q_mass_rate * dt * 0.5;
            double s = tao * (2 / (1 + tao*tao));
            double kk = q_mass_rate * dt / 2;
            Eigen::RowVector2d v_minus = get<velocity>(p);
            v_minus = v_minus + Evec * kk;
            Eigen::RowVector2d v_plus(v_minus.x() * (1 - s * tao) + s * v_minus.y(), 
                                     -s * v_minus.x() + v_minus.y() * (1 - s * tao));
            get<velocity>(p) = v_plus + Evec * kk;
            get<position>(p) = get<position>(p) + get<velocity>(p) * dt;

            if (get<position>(p).x() < xmin) {
                atomic_add(*absorbed_counter, 1);
                pos_x_nan_is_invalid<Particle>::make_invalid(p);
            } else if (get<position>(p).x() > xmax) {
                pos_x_nan_is_invalid<Particle>::make_invalid(p);
            }
            if (get<position>(p).y() < ymin) {
                get<position>(p).y() += span_y;
            } else if (get<position>(p).y() > ymax) {
                get<position>(p).y() -= span_y;
            }
        };
    });
}

int main()
{
    std::cout << "EDI Simulation - Aligned with ref_EDI.cpp" << std::endl;

    sycl::queue q{sycl::default_selector_v};
    std::cout << "Using device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    grid2D grid({0.0, 0.0}, {0.025, 0.0128}, {500, 256});
    grid2D grid_ex({-grid.del<0>() / 2, 0}, {0.025 + grid.del<0>() / 2, 0.0128}, {grid.numCells<0>() + 1, grid.numCells<1>()});
    grid2D grid_ey({0, -grid.del<1>() / 2}, {0.025, 0.0128 + grid.del<1>() / 2}, {grid.numCells<0>(), grid.numCells<1>() + 1});
    grid1D grid1d({0.0}, {0.025}, {11});



    std::cout << "[Init] Creating device fields..." << std::endl;
    Nf Phi(q, grid);
    Nf q_dens(q, grid);
    Nf e_dens(q, grid);
    Nf i_dens(q, grid);
    Nf Ex(q, grid_ex);
    Nf Ey(q, grid_ey);
    Nf1d_host Br_field_host = create_magnetic_field_profile(grid1d);
    Nf1d Br(q, Br_field_host.getGrid());
    Br.copy(Br_field_host.getContent());

    std::cout << "[Init] Creating Poisson solver..." << std::endl;
    Poisson_solver_2d psolver;
    psolver.init(
        "cuda_sparselu_gpu",
        grid, 
        Poisson_solver_2d::Cartesian,
        {
            Dirichlet_line(grid, boundary_direction_2d::W) = 200,
            Dirichlet_line(grid, boundary_direction_2d::E) = 0
        }
    );
    double* phi_right_sum = shared_variable<double>(q);

    int* counter_ion = shared_variable<int>(q);
    int* counter_ele = shared_variable<int>(q);

    ParticleGroup ele(q), ion(q);
    std::vector<Particle> ele_buf, ion_buf;
    double weight = 1.7e6;
    double dt = 5e-12;

    int iter0 = 0;
    {
        // load checkpoint
        injParticle_init(ele, ion, grid, weight);
    }
    particleCount(ele, e_dens, weight / (grid.del<0>() * grid.del<1>()));
    particleCount(ion, i_dens, weight / (grid.del<0>() * grid.del<1>()));

    
    std::cout << "[Init] Starting simulation..." << std::endl;
    for (int iter = iter0; iter <= 6e6; iter++)
    {
Tic("Psolve")
        q_dens.for_each([&](sycl::handler& h) {
            auto e_dens_acc = e_dens.get_access(h);
            auto i_dens_acc = i_dens.get_access(h);
            return [=](size_t i, double& v) {
                v = (i_dens_acc(i) - e_dens_acc(i)) * 1.602e-19;
            };
        });
        psolver.solve(Phi.data(), q_dens.data());
        auto sample_i = grid.nN(Nf::Position{0.024, 0.0}).indices[0];
        *phi_right_sum = 0;
        Phi.for_each([&](sycl::handler& h) {
            return [=](size_t i, double& v) {
                auto node = grid.i2n(i);
                if (node.indices[0] == sample_i)
                    atomic_add(*phi_right_sum, v);
            };
        });
        double rectify_phi = *phi_right_sum / (grid.numCells<1>() + 1);
        compute_rectify_Phi(Phi, rectify_phi);

TocTic("pPush")
        // push
        *counter_ele = 0;
        *counter_ion = 0;
        particlePush(q, ele, Phi, Br, -1.602e-19 / 9.11e-31, dt, counter_ele);
        particlePush(q, ion, Phi, Br, 1.602e-19 / (1.66e-27 * 131.29), dt, counter_ion);
        
        double anode_net_particle_flow = 0;
        anode_net_particle_flow = *counter_ele - *counter_ion;
TocTic("pJnj")
        // inject
        injParticle_ioni(ele_buf, ion_buf, grid, weight, dt);
        injParticle_cathode(ele_buf, grid, anode_net_particle_flow);
        ele.insert(ele_buf);
        ion.insert(ion_buf);
        if (iter % 50 == 0)
        {
            ele.compress();
            ion.compress();
        }
TocTic("pCount")
        // count
        particleCount(ele, e_dens, weight / (grid.del<0>() * grid.del<1>()));
        particleCount(ion, i_dens, weight / (grid.del<0>() * grid.del<1>()));
Toc
        int ele_size = ele.size();
        int ion_size = ion.size();
        if (iter % 100 == 0)
        {
            std::ofstream of("numline.plt", std::ios::app);
            double current = anode_net_particle_flow * weight / dt * 1.602e-19 / grid.span<1>();
            std::stringstream ss;
            ss << iter << "," << ele_size << "," << ion_size << "," << current << std::endl;
            of << ss.str();
            of.close();
            std::cout << ss.str();
        }
        if (iter % 5000 == 0)
        {
            WriteTimer("output/"+std::to_string(iter) + "_timeused.txt");
            e_dens.plot("output/"+std::to_string(iter) + "_e_dens.plt", "ne");
            i_dens.plot("output/"+std::to_string(iter) + "_i_dens.plt", "ni");
            Phi.plot("output/"+std::to_string(iter) + "_phi.plt", "phi");
            compute_Efield(Ex, Ey, Phi);
            Ex.plot("output/"+std::to_string(iter) + "_Ex.plt", "Ex");
            Ey.plot("output/"+std::to_string(iter) + "_Ey.plt", "Ey");
        }
        if(iter % 20000 == 0)
        {
            // save checkpoint
        }
    }

    return 0;
}