#include "types.hpp"
#include "species.hpp"
#include "config_ops.hpp"
#include "field_ops.hpp"
#include "particle_ops.hpp"
#include "data_collection.hpp"
#include "moment_output.hpp"
#include "checkpoint.hpp"

#include <psum/psum.hpp>

#include <memory>

using namespace psum::prelude;
using namespace psum::field::interp_tools;
using namespace psum::random::RandFunction3D;
using namespace magnet_nozzle_2d;
using property::position;
using property::velocity;
using property::random_seed;

int main() {
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    psum::serialization::json_loader param;
    param.load_json("config.json");
    std::filesystem::create_directories("output");

    int simulate_step = param.obj<int>("simulate_step");
    int print_step = param.obj<int>("print_step");
    int collect_step = param.obj<int>("collect_step");
    int output_step = param.obj<int>("output_step");
    int save_step = param.obj<int>("save_step");
    int checkpoint = param.obj<int>("checkpoint");

    double capacitance = param.obj<double>("capacitance");
    double Te = param.obj<double>("Te");
    double Ti = param.obj<double>("Ti");
    double reference = param.obj<double>("reference");
    double threshold_init = param.obj<double>("threshold_init");
    double ppc = param.obj<double>("ppc");
    double time_coef = param.obj<double>("time_coef");
    double grid_coef = param.contains<double>("grid_coef") ? param.obj<double>("grid_coef") : 1.0;
    double gamma = load_gamma(param);
    const bool output_moments10_axisymmetric =
        param.contains<bool>("output_moments10_axisymmetric") && param.obj<bool>("output_moments10_axisymmetric");
    const std::string moment10_source_sha =
        param.contains<std::string>("moment10_source_sha") ? param.obj<std::string>("moment10_source_sha") : "unprovided";
    const std::string moment10_binary_sha =
        param.contains<std::string>("moment10_binary_sha") ? param.obj<std::string>("moment10_binary_sha") : "unprovided";
    const std::string moment10_config_sha =
        param.contains<std::string>("moment10_config_sha") ? param.obj<std::string>("moment10_config_sha") : "unprovided";

    double start_pos = param.obj<double>("start_pos");
    double injection_pos = param.obj<double>("injection_pos");
    double end_pos = param.obj<double>("end_pos");
    double r_length = param.obj<double>("r_length");
    double r_inlet = param.obj<double>("r_inlet");

    std::string field_type = param.contains<std::string>("field_type") ? param.obj<std::string>("field_type") : "file";
    double coil_B_max = param.obj<double>("coil_B_max");
    double coil_x_max = param.obj<double>("coil_x_max");
    double coil_L = param.obj<double>("coil_L");

    const double lambda_preview = std::sqrt(effective_epsilon_0(gamma) * Te / reference / Q);
    const size_t I_preview = std::max(
        static_cast<size_t>((end_pos - start_pos) / (0.5 * lambda_preview) * grid_coef),
        static_cast<size_t>(2)
    );
    const Grid grid(
        {start_pos, 0.0},
        {end_pos, r_length},
        {static_cast<int>(I_preview), static_cast<int>(
            std::max(
                static_cast<size_t>(r_length / (0.5 * lambda_preview) * grid_coef),
                static_cast<size_t>(2)
            )
        )}
    );
    DNf<2> B(q, grid);
    HNf<2> B_host(grid);
    double B_max;
    if (field_type == "coil") {
        B_max = create_coil_field(B, B_host, coil_B_max, coil_x_max, coil_L);
    } else {
        throw std::runtime_error("Unsupported field type: " + field_type);
    }

    const DerivedParameters2D derived = compute_derived_parameters(
        gamma,
        Te,
        Ti,
        reference,
        ppc,
        time_coef,
        grid_coef,
        start_pos,
        end_pos,
        r_length,
        r_inlet,
        B_max
    );
    double lambda = derived.lambda;
    double T_p_rad = derived.plasma_period_scale;
    size_t I = derived.I;
    size_t J = derived.J;
    double dx = derived.dx;
    double particle_weight = derived.particle_weight;
    double T_be = derived.electron_gyro_period;
    double dt = derived.dt;
    double num_to_inject = derived.ion_injection_expectation;
    {
        std::cout << "Start simulation with parameters:" << std::endl;
        std::cout << "  simulate_step: " << simulate_step << std::endl;
        std::cout << "  print_step: " << print_step << std::endl;
        std::cout << "  collect_step: " << collect_step << std::endl;
        std::cout << "  output_step: " << output_step << std::endl;
        std::cout << "  save_step: " << save_step << std::endl;
        std::cout << "  checkpoint: " << checkpoint << std::endl;
        std::cout << "  capacitance: " << capacitance << std::endl;
        std::cout << "  Te: " << Te << std::endl;
        std::cout << "  Ti: " << Ti << std::endl;
        std::cout << "  reference: " << reference << std::endl;
        std::cout << "  threshold_init: " << threshold_init << std::endl;
        std::cout << "  gamma: " << gamma << std::endl;
        std::cout << "  epsilon_eff: " << derived.epsilon << std::endl;
        std::cout << "  ppc: " << ppc << std::endl;
        std::cout << "  time_coef: " << time_coef << std::endl;
        std::cout << "  grid_coef: " << grid_coef << std::endl;
        std::cout << "  start_pos: " << start_pos << std::endl;
        std::cout << "  injection_pos: " << injection_pos << std::endl;
        std::cout << "  end_pos: " << end_pos << std::endl;
        std::cout << "  r_length: " << r_length << std::endl;
        std::cout << "  r_inlet: " << r_inlet << std::endl;
        std::cout << "  field_type: " << field_type << std::endl;
        std::cout << "  coil_B_max: " << coil_B_max << std::endl;
        std::cout << "  coil_x_max: " << coil_x_max << std::endl;
        std::cout << "  coil_L: " << coil_L << std::endl;

        std::cout << "  B_max: " << B_max << std::endl;
        std::cout << "  lambda: " << lambda << std::endl;
        std::cout << "  T_p_rad: " << T_p_rad << std::endl;
        std::cout << "  I: " << I << std::endl;
        std::cout << "  J: " << J << std::endl;
        std::cout << "  dx: " << dx << std::endl;
        std::cout << "  dt: " << dt << std::endl;
        std::cout << "  particle_weight: " << particle_weight << std::endl;
        std::cout << "  num_to_inject: " << num_to_inject << std::endl;
    }
    
    Species ele(q, m_e, -Q, particle_weight), ion(q, m_H, Q, particle_weight);

    auto ele_creator = [&](uint32_t seed) -> Particle {
        rander R(seed);
        Particle p;
        double r = sqrt(R()) * r_inlet;
        double alpha = 2.0 * M_PI * R();

        Eigen::RowVector2d norm_zr = interp(Eigen::RowVector2d(injection_pos, r), B_host);
        norm_zr = norm_zr.normalized();
        const Eigen::RowVector3d norm = {norm_zr.x(), norm_zr.y()*cos(alpha), norm_zr.y()*sin(alpha)};
        const Eigen::Vector3d ve = RandFunction3D::RandV_halfMaxw(R, norm, Te*11600, m_e);
        Eigen::RowVector3d pos = Eigen::RowVector3d(injection_pos, r*cos(alpha), r*sin(alpha)) + ve.dot(norm) * norm * dt * 0.5;
        get<position>(p) = pos;
        get<velocity>(p) = ve;
        get<random_seed>(p) = psum::random::global_random::rand_uint();
        return p;
    };

    auto ion_creator = [&](uint32_t seed) -> Particle {
        rander R(seed);
        Particle p;
        double r = sqrt(R()) * r_inlet;
        double alpha = 2.0 * M_PI * R();
        Eigen::RowVector2d norm_zr = interp(Eigen::RowVector2d(injection_pos, r), B_host);
        norm_zr = norm_zr.normalized();
        const Eigen::RowVector3d norm = {norm_zr.x(), norm_zr.y()*cos(alpha), norm_zr.y()*sin(alpha)};
        const Eigen::Vector3d vi = RandFunction3D::RandV_halfMaxw(R, norm, Ti*11600, m_H);
        Eigen::RowVector3d pos = Eigen::RowVector3d(injection_pos, r*cos(alpha), r*sin(alpha)) + vi.dot(norm) * norm * dt * 0.5;
        get<position>(p) = pos;
        get<velocity>(p) = vi;
        get<random_seed>(p) = psum::random::global_random::rand_uint();
        return p;
    };

    const double inject_detect_range = dx;
    grid1D inject_detect_grid({injection_pos}, {injection_pos + inject_detect_range}, {5});
    cell_field1D<double> inject_num_field(q, inject_detect_grid);
    size_t ion_input = 0, ele_input = 0;

    DCf<7> ele_data(q, grid), ion_data(q, grid);
    std::unique_ptr<DCf<10>> ele_moment_data;
    std::unique_ptr<DCf<10>> ion_moment_data;
    DNf<1> phi(q, grid), charge_density(q, grid);

    int *charge_out_num = sycl::malloc_device<int>(1, q);
    int charge_out_num_host = static_cast<int>(threshold_init * capacitance / (Q * particle_weight));
    double reflect_threshold_host = 0;
    q.copy(&charge_out_num_host, charge_out_num, 1).wait();
    auto compute_reflect_threshold_host = [&charge_out_num_host, particle_weight, capacitance, &reflect_threshold_host] () {
        reflect_threshold_host = charge_out_num_host * Q * particle_weight / capacitance;
    };

    Poisson_solver_2d solver;
    auto reflect_threshold_bc_east = std::function<double(const std::array<double, 2>&)>(
        [&charge_out_num_host, particle_weight, capacitance](const std::array<double, 2>&) {
            return charge_out_num_host * Q * particle_weight / capacitance;
        }
    );
    auto reflect_threshold_bc_north = std::function<double(const std::array<double, 2>&)>(
        [&charge_out_num_host, particle_weight, capacitance](const std::array<double, 2>&) {
            return charge_out_num_host * Q * particle_weight / capacitance;
        }
    );
    init_poisson_solver(solver, grid, r_length, derived.epsilon, reflect_threshold_bc_east, reflect_threshold_bc_north);
    charge_density.setZero();
    solver.solve(phi.data(), charge_density.data());

    mean_smoother<7> ele_data_smooth(grid), ion_data_smooth(grid);
    std::unique_ptr<mean_smoother<10>> ele_moment_smooth;
    std::unique_ptr<mean_smoother<10>> ion_moment_smooth;
    if (output_moments10_axisymmetric) {
        ele_moment_data = std::make_unique<DCf<10>>(q, grid);
        ion_moment_data = std::make_unique<DCf<10>>(q, grid);
        ele_moment_smooth = std::make_unique<mean_smoother<10>>(grid);
        ion_moment_smooth = std::make_unique<mean_smoother<10>>(grid);
    }

    auto compress_if_needed = [](Species& ps) {
        const size_t total_size = ps.get_content().size();
        if (total_size == 0) return;
        const double invalid_ratio = static_cast<double>(ps.get_invalid_indexes().size()) / static_cast<double>(total_size);
        if (invalid_ratio > 0.15) {
            ps.compress();
        }
    };

    int iter = 0;
    double physical_time = 0.0;
    if (checkpoint) {
        int charge_out_num_loaded = 0;
        load_checkpoint(ele, ion, charge_out_num_loaded, physical_time,
                        "output/checkpoint" + std::to_string(checkpoint) + ".bin");
        charge_out_num_host = charge_out_num_loaded;
        q.copy(&charge_out_num_host, charge_out_num, 1).wait();
        iter = checkpoint;
    }

    for (iter++; iter <= checkpoint + simulate_step; iter++) {
        Tic("total")
        q.copy(charge_out_num, &charge_out_num_host, 1).wait();
        compute_reflect_threshold_host();
        Tic("count_charge")
        count_charge(ele, ion, charge_density);
        TocTic("solve")
        solver.solve(phi.data(), charge_density.data());

        TocTic("ele_push")

        double elementary_charge = Q;
        ele.for_each([&](sycl::handler& h){
            auto phi_acc = phi.get_access(h);
            auto B_acc = B.get_access(h);
            return [=](Particle &p) {
                double old_x = get<position>(p).x();
                const double old_y = get<position>(p).y();
                const double old_z = get<position>(p).z();
                double old_r = sqrt(old_y * old_y + old_z * old_z);
                push_particles(p, phi_acc, B_acc, -elementary_charge / m_e, dt, start_pos, end_pos, r_length);
                const double py = get<position>(p).y();
                const double pz = get<position>(p).z();
                double r = sqrt(py * py + pz * pz);
                double px = get<position>(p).x();
                if (!std::isfinite(px) || !std::isfinite(r)) return;

                if (px > end_pos)
                {
                    double interp_x = std::min(std::max(old_x, start_pos + 1e-10), end_pos - 1e-10);
                    double interp_r = std::min(std::max(old_r, 1e-10), r_length - 1e-10);
                    const double particle_phi = interp_nearest(Eigen::RowVector2d{interp_x, interp_r}, phi_acc);
                    const double energy = 0.5 * m_e * get<velocity>(p).squaredNorm();
                    const double reflect_threshold = *charge_out_num * elementary_charge * particle_weight / capacitance;
                    if (energy > (particle_phi - reflect_threshold) * elementary_charge)
                    {
                        Species::validator::make_invalid(p);
                        atomic_add(*charge_out_num, -1);
                        return;
                    }
                    else
                    {
                        get<position>(p).x() = end_pos * 2 - px;
                        get<velocity>(p) *= -1;
                    }
                }
                else if (r > r_length)
                {
                    double interp_x = std::min(std::max(old_x, start_pos + 1e-10), end_pos - 1e-10);
                    const double particle_phi = interp_nearest(Eigen::RowVector2d{interp_x, r_length - 1e-10}, phi_acc);
                    const double energy = 0.5 * m_e * get<velocity>(p).squaredNorm();
                    const double reflect_threshold = *charge_out_num * elementary_charge * particle_weight / capacitance;
                    if (energy > (particle_phi - reflect_threshold) * elementary_charge)
                    {
                        Species::validator::make_invalid(p);
                        atomic_add(*charge_out_num, -1);
                    }
                    else
                    {
                        Eigen::Vector2d pos_yz = {get<position>(p).y(), get<position>(p).z()};
                        Eigen::Vector2d norm_yz = pos_yz / r;
                        get<position>(p).y() = (r_length * 2 - r) * norm_yz.x();
                        get<position>(p).z() = (r_length * 2 - r) * norm_yz.y();
                        get<velocity>(p) *= -1;
                    }
                }
            };
        });
        TocTic("ion_push")
        ion.for_each([&](sycl::handler& h){
            auto phi_acc = phi.get_access(h);
            auto B_acc = B.get_access(h);
            return [=](Particle &p) {
                push_particles(p, phi_acc, B_acc, elementary_charge / m_H, dt, start_pos, end_pos, r_length);
                const double py = get<position>(p).y();
                const double pz = get<position>(p).z();
                double r = sqrt(py * py + pz * pz);
                if (get<position>(p).x() > end_pos || r > r_length)
                {
                    Species::validator::make_invalid(p);
                    atomic_add(*charge_out_num, 1);
                }
            };
        });
        TocTic("inject")
        std::tie(ion_input, ele_input) = inject_particles(
            ele, ion, ion_creator, ele_creator, num_to_inject, inject_num_field, r_inlet
        );
        Toc
        if (iter % collect_step == 0) {
            Tic("compress")
            compress_if_needed(ele);
            compress_if_needed(ion);
            TocTic("collect_data")
            if (output_moments10_axisymmetric) {
                collect_data_with_moments<true>(ele, ele_data, ele_data_smooth, ele_moment_data.get(), ele_moment_smooth.get());
                collect_data_with_moments<true>(ion, ion_data, ion_data_smooth, ion_moment_data.get(), ion_moment_smooth.get());
            } else {
                collect_data(ele, ele_data, ele_data_smooth);
                collect_data(ion, ion_data, ion_data_smooth);
            }
            Toc
        }
        Toc_("total")

        physical_time += dt;

        if (iter % print_step == 0) {
            auto phi_host = phi.getContent().to_host();
            double phi_min = *std::min_element(phi_host.begin(), phi_host.end());
            double phi_max = *std::max_element(phi_host.begin(), phi_host.end());
            std::cout << "  phi range: [" << phi_min << ", " << phi_max << "]" << std::endl;
            std::cout << "iter: " << iter << " time: " << physical_time
                      << " ele_num: " << ele.size() << " ion_num: " << ion.size()
                      << " ion_input: " << ion_input << " ele_input: " << ele_input
                      << " reflect_threshold: " << reflect_threshold_host << std::endl;

            const bool need_record_header =
                !std::filesystem::exists("output/record.plt") || (checkpoint == 0 && iter == print_step);
            if (need_record_header) {
                std::ofstream record("output/record.plt", std::ios::out);
                record << "variables=time,i,ele_num,ion_num,ion_input,ele_input,reflect_threshold,phi_min,phi_max\n";
            }
            std::ofstream record_file("output/record.plt", std::ios::app);
            record_file << physical_time << " " << iter << " " << ele.size()
                        << " " << ion.size() << " " << ion_input << " " << ele_input
                        << " " << reflect_threshold_host
                        << " " << phi_min << " " << phi_max << std::endl;
        }

        if (iter % output_step == 0) {
            Tic("output")
            HNf<1> phi_host(grid);
            HNf<2> B_out_host(grid);
            phi_host.copy(phi.getContent().to_host());
            B_out_host.copy(B.getContent().to_host());
            HCf<17> data_out(grid);
            HCf<7> ele_data_temp = pop_data(ele, ele_data_smooth);
            HCf<7> ion_data_temp = pop_data(ion, ion_data_smooth);
            for (size_t i = 0; i < data_out.size(); i++) {
                const auto pos = grid.cellCenter(grid.i2c(i));
                const auto phi_val = interp(pos, phi_host);
                const auto B_val = interp(pos, B_out_host);
                Eigen::RowVector<double, 17> out_row;
                out_row << phi_val, B_val.x(), B_val.y(),
                           ion_data_temp(i), ele_data_temp(i);
                data_out(i) = out_row;
            }
            data_out.plot(
                "output/data_out_" + std::to_string(iter) + ".plt",
                "phi,Bx,Br,ni,vix,viy,viz,Ti_parallel,Ti_perp,Ti_anisotropy,ne,vex,vey,vez,Te_parallel,Te_perp,Te_anisotropy"
            );
            if (output_moments10_axisymmetric) {
                const size_t ele_samples = ele_moment_smooth->count();
                const size_t ion_samples = ion_moment_smooth->count();
                HCf<10> ele_moment_temp = ele_moment_smooth->pop();
                HCf<10> ion_moment_temp = ion_moment_smooth->pop();
                Moment10Metadata ele_meta{
                    moment10_source_sha, moment10_binary_sha, moment10_config_sha, "electron",
                    ele.mass(), ele.charge(), ele.weight(), collect_step, output_step, ele_samples
                };
                Moment10Metadata ion_meta{
                    moment10_source_sha, moment10_binary_sha, moment10_config_sha, "ion",
                    ion.mass(), ion.charge(), ion.weight(), collect_step, output_step, ion_samples
                };
                write_moment10_axisymmetric("output/moments10_axisymmetric_electron_" + std::to_string(iter) + ".plt", ele_moment_temp, ele_meta);
                write_moment10_axisymmetric("output/moments10_axisymmetric_ion_" + std::to_string(iter) + ".plt", ion_moment_temp, ion_meta);
            }
            WriteTimer("output/timeused.txt")
            Toc_("output")
        }

        if (checkpoint_output_enabled(save_step, iter)) {
            Tic("checkpoint")
            q.copy(charge_out_num, &charge_out_num_host, 1).wait();
            save_checkpoint(ele, ion, charge_out_num_host, physical_time,
                          "output/checkpoint" + std::to_string(iter) + ".bin");
            Toc_("checkpoint")
        }
    }

    sycl::free(charge_out_num, q);
    return 0;
}
