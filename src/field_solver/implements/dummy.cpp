// Placeholder backend: solve() always throws. Used by batch wrappers to hold
// per-case solvers without paying any backend memory or setup cost.
#include "../register/interface.h"
#include <stdexcept>

namespace psum {

namespace field_solver {

namespace implements {

namespace dummy {

    psum_field_solver_handle init() {
        return new int(0);
    }

    void set_matrix(psum_field_solver_handle, unsigned long long, unsigned long long, unsigned long long, unsigned long long*, unsigned long long*, double*) {
    }

    void solve(psum_field_solver_handle, double*, double*) {
        throw std::runtime_error("dummy backend: solve is not available");
    }

    void set_options(psum_field_solver_handle, const char*) {
    }

    void set_source_replace(psum_field_solver_handle, unsigned long long, unsigned long long*, double*) {
        throw std::runtime_error("dummy backend: set_source_replace is not available");
    }

    void set_source_addback(psum_field_solver_handle, unsigned long long, unsigned long long*, double*) {
        throw std::runtime_error("dummy backend: set_source_addback is not available");
    }

    void solver_free(psum_field_solver_handle h) {
        delete static_cast<int*>(h);
    }

    static int _ = []() {
        func_table table;
        table.init = init;
        table.set_matrix = set_matrix;
        table.solve = solve;
        table.set_options = set_options;
        table.set_source_replace = set_source_replace;
        table.set_source_addback = set_source_addback;
        table.free = solver_free;
        register_solver("dummy", table);
        return 0;
    }();

}

}

}

}
