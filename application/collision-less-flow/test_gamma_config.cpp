#include "config_ops.hpp"
#include "field_ops.hpp"
#include "data_collection.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>

using namespace magnet_nozzle_2d;

namespace {

void require_close(double actual, double expected, double rel_tol, const std::string& label) {
    const double scale = std::max(1.0, std::abs(expected));
    if (std::abs(actual - expected) > rel_tol * scale) {
        throw std::runtime_error(
            label + " mismatch: actual=" + std::to_string(actual)
            + " expected=" + std::to_string(expected)
        );
    }
}

void write_gamma_config(const std::string& path, bool include_gamma, double gamma) {
    std::ofstream out(path);
    out << "{\n";
    if (include_gamma) out << "  \"gamma\": " << std::fixed << std::setprecision(6) << gamma << "\n";
    out << "}\n";
}

void test_gamma_default_and_validation() {
    write_gamma_config("test_gamma_absent.json", false, 1.0);
    write_gamma_config("test_gamma_one.json", true, 1.0);
    write_gamma_config("test_gamma_bad.json", true, 0.0);

    psum::serialization::json_loader absent;
    absent.load_json("test_gamma_absent.json");
    psum::serialization::json_loader explicit_one;
    explicit_one.load_json("test_gamma_one.json");
    psum::serialization::json_loader bad;
    bad.load_json("test_gamma_bad.json");

    require_close(load_gamma(absent), 1.0, 0.0, "absent gamma default");
    require_close(load_gamma(explicit_one), 1.0, 0.0, "explicit gamma=1");

    bool rejected = false;
    try {
        (void)load_gamma(bad);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("gamma<=0 was not rejected");

    std::remove("test_gamma_absent.json");
    std::remove("test_gamma_one.json");
    std::remove("test_gamma_bad.json");
}

void test_moment10_contract_helpers() {
    const Eigen::RowVector3d velocity{1.0, 2.0, 3.0};
    const Moment10 value = make_weighted_moment10(2.0, velocity, 0.0, 1.0);
    require_close(value(0), 2.0, 1.0e-12, "2D moment10 M0 weight");
    require_close(value(2), 6.0, 1.0e-12, "2D moment10 radial first moment");
    require_close(value(3), -4.0, 1.0e-12, "2D moment10 tangent first moment");
    require_close(value(9), -12.0, 1.0e-12, "2D moment10 radial-tangent cross term");
}

DerivedParameters2D derived_for_gamma(double gamma) {
    return compute_derived_parameters(
        gamma,
        30.0,
        3.0,
        5.0e15,
        200.0,
        1.0,
        1.0,
        -0.2,
        0.4,
        0.15,
        0.05,
        1.0
    );
}

void test_derived_gamma_scaling_and_weight_volume() {
    const auto gamma1 = derived_for_gamma(1.0);
    const auto gamma10 = derived_for_gamma(10.0);

    require_close(gamma1.epsilon, epsilon_0, 1.0e-15, "gamma=1 epsilon");
    require_close(gamma10.epsilon, 100.0 * epsilon_0, 1.0e-15, "gamma=10 epsilon");
    require_close(gamma10.lambda, 10.0 * gamma1.lambda, 1.0e-12, "gamma lambda scaling");
    require_close(
        gamma10.plasma_period_scale,
        10.0 * gamma1.plasma_period_scale,
        1.0e-12,
        "gamma plasma-period scaling"
    );

    const double inlet_swept_volume = gamma10.dx * M_PI * 0.05 * 0.05;
    const double expected_weight = 5.0e15 * inlet_swept_volume / 200.0;
    require_close(gamma10.particle_weight, expected_weight, 1.0e-12, "2D inlet-volume weight");
}

void test_checkpoint_output_guard() {
    if (checkpoint_output_enabled(0, 1)) {
        throw std::runtime_error("save_step=0 should disable checkpoint output");
    }
    if (checkpoint_output_enabled(-10, 10)) {
        throw std::runtime_error("negative save_step should not enable checkpoint output");
    }
    if (checkpoint_output_enabled(100, 50)) {
        throw std::runtime_error("checkpoint output enabled before save_step interval");
    }
    if (!checkpoint_output_enabled(100, 100)) {
        throw std::runtime_error("checkpoint output not enabled at save_step interval");
    }
}

void fill_charge(DNf<1>& field, double value) {
    field.for_each([&](sycl::handler&) {
        return [=](size_t, double& val, const auto&) {
            val = value;
        };
    });
}

double max_abs_difference(const HNf<1>& a, const HNf<1>& b, double scale = 1.0) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(result, std::abs(a(i) - scale * b(i)));
    }
    return result;
}

void test_fixed_poisson_gamma_regression() {
    sycl::queue queue(sycl::default_selector_v);
    const Grid grid({0.0, 0.0}, {0.02, 0.01}, {4, 3});
    DNf<1> rho(queue, grid);
    DNf<1> phi_default(queue, grid);
    DNf<1> phi_explicit(queue, grid);
    DNf<1> phi_gamma10(queue, grid);
    fill_charge(rho, 1.0e-8);
    phi_default.setZero();
    phi_explicit.setZero();
    phi_gamma10.setZero();
    queue.wait();

    auto zero_east = [](const std::array<double, 2>&) { return 0.0; };
    auto zero_north = [](const std::array<double, 2>&) { return 0.0; };

    Poisson_solver_2d solver_default;
    Poisson_solver_2d solver_explicit;
    Poisson_solver_2d solver_gamma10;
    init_poisson_solver(solver_default, grid, 0.01, effective_epsilon_0(1.0), zero_east, zero_north);
    init_poisson_solver(solver_explicit, grid, 0.01, effective_epsilon_0(1.0), zero_east, zero_north);
    init_poisson_solver(solver_gamma10, grid, 0.01, effective_epsilon_0(10.0), zero_east, zero_north);

    solver_default.solve(phi_default.data(), rho.data());
    solver_explicit.solve(phi_explicit.data(), rho.data());
    solver_gamma10.solve(phi_gamma10.data(), rho.data());
    queue.wait();

    HNf<1> default_host(grid);
    HNf<1> explicit_host(grid);
    HNf<1> gamma10_host(grid);
    default_host.copy(phi_default.getContent().to_host());
    explicit_host.copy(phi_explicit.getContent().to_host());
    gamma10_host.copy(phi_gamma10.getContent().to_host());

    if (max_abs_difference(default_host, explicit_host) > 1.0e-9) {
        throw std::runtime_error("gamma=1 fixed Poisson solve changed");
    }
    if (max_abs_difference(default_host, gamma10_host, 100.0) > 1.0e-6) {
        throw std::runtime_error("gamma=10 fixed Poisson solve did not scale by epsilon");
    }
    require_close(default_host(0), explicit_host(0), 1.0e-12, "boundary phi regression");
}

}

int main() {
    test_gamma_default_and_validation();
    test_derived_gamma_scaling_and_weight_volume();
    test_checkpoint_output_guard();
    test_fixed_poisson_gamma_regression();
    test_moment10_contract_helpers();
    std::cout << "[test] 2D gamma config tests passed" << std::endl;
    return 0;
}
