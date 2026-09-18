#include <iostream>
#include <vector>
#include "backend.hpp"

using namespace psum::field_solver;

int main() {
    bool success = true;
    std::cout << "Testing eigen_sparselu_cpu solver..." << std::endl;
    try {
        solver_backend f1("eigen_sparselu_cpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing cuda_sparselu_gpu solver..." << std::endl;
    try {
        solver_backend f2("cuda_sparselu_gpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing cuda_schurcomplement_gpu solver..." << std::endl;
    try {
        solver_backend f2s("cuda_schurcomplement_gpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing eigen_multigrid_cpu solver..." << std::endl;
    try {
        solver_backend f3("eigen_multigrid_cpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing eigen_schurcomplement_cpu solver..." << std::endl;
    try {
        solver_backend f4("eigen_schurcomplement_cpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing sycl_multigrid_gpu solver..." << std::endl;
    try {
        solver_backend f5("sycl_multigrid_gpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    std::cout << "Testing cuda_multigrid_gpu solver..." << std::endl;
    try {
        solver_backend f6("cuda_multigrid_gpu");
        std::cout << "  Solver created successfully" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "  Solver creation failed: " << e.what() << std::endl;
        success = false;
    }

    if (success) {
        std::cout << "All solvers initialized successfully!" << std::endl;
    }
    else {
        std::cout << "Some solvers failed to initialize. Please check error messages." << std::endl;
    }
    return 0;
}
