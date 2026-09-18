#ifndef PSUM_FIELD_SOLVER_BACKEND_HPP
#define PSUM_FIELD_SOLVER_BACKEND_HPP

#include "register/interface.h"
#include <stdexcept>

namespace psum {

namespace field_solver {

class solver_backend {
    psum_field_solver_handle handle;
    func_table funcs;

    template<typename T>
    void raise_nullptr(T ptr) const {
        if (ptr == nullptr) {
            throw std::runtime_error("Error: null pointer");
        }
    }

public:
    inline solver_backend() {
        handle = nullptr;
        funcs.init = nullptr;
        funcs.set_matrix = nullptr;
        funcs.solve = nullptr;
        funcs.set_options = nullptr;
        funcs.set_source_replace = nullptr;
        funcs.set_source_addback = nullptr;
        funcs.free = nullptr;
    }
    inline solver_backend(const char* kind) {
        funcs = get_solver_funcs(kind);
        raise_nullptr(funcs.init);
        raise_nullptr(funcs.free);
        handle = funcs.init();
        if (handle == nullptr) {
            throw std::runtime_error("Error: failed to create solver.");
        }
    }

    inline solver_backend(const solver_backend&) = delete;
    inline solver_backend& operator=(const solver_backend&) = delete;

    inline solver_backend(solver_backend&& other) noexcept
        : handle(other.handle), funcs(other.funcs) {
        other.handle = nullptr;
        other.funcs.init = nullptr;
        other.funcs.set_matrix = nullptr;
        other.funcs.solve = nullptr;
        other.funcs.set_options = nullptr;
        other.funcs.set_source_replace = nullptr;
        other.funcs.set_source_addback = nullptr;
        other.funcs.free = nullptr;
    }

    inline solver_backend& operator=(solver_backend&& other) noexcept {
        if (this != &other) {
            if (handle != nullptr) {
                raise_nullptr(funcs.free);
                funcs.free(handle);
            }
            handle = other.handle;
            funcs = other.funcs;
            other.handle = nullptr;
            other.funcs.init = nullptr;
            other.funcs.set_matrix = nullptr;
            other.funcs.solve = nullptr;
            other.funcs.set_options = nullptr;
            other.funcs.set_source_replace = nullptr;
            other.funcs.set_source_addback = nullptr;
            other.funcs.free = nullptr;
        }
        return *this;
    }

    inline ~solver_backend() {
        if (handle != nullptr) {
            raise_nullptr(funcs.free);
            funcs.free(handle);
        }
    }
    inline void set_matrix(unsigned long long n_row, unsigned long long n_col, unsigned long long nnz, unsigned long long* rows, unsigned long long* cols, double* vals) {
        raise_nullptr(funcs.set_matrix);
        funcs.set_matrix(handle, n_row, n_col, nnz, rows, cols, vals);
    }
    inline void solve(double* b, double* x) {
        raise_nullptr(funcs.solve);
        funcs.solve(handle, b, x);
    }
    inline void set_options(const char* options) {
        raise_nullptr(funcs.set_options);
        funcs.set_options(handle, options);
    }
    inline void set_source_replace(unsigned long long size, unsigned long long* idxs, double* new_values) {
        raise_nullptr(funcs.set_source_replace);
        funcs.set_source_replace(handle, size, idxs, new_values);
    }
    inline void set_source_addback(unsigned long long size, unsigned long long* idxs, double* add_values) {
        raise_nullptr(funcs.set_source_addback);
        funcs.set_source_addback(handle, size, idxs, add_values);
    }
    inline bool is_null() const {
        return handle == nullptr;
    }
};

}

}

#endif