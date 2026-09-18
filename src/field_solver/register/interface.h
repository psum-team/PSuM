#ifndef PSUM_FIELD_SOLVER_REGISTER_INTERFACE_H
#define PSUM_FIELD_SOLVER_REGISTER_INTERFACE_H

extern "C" {
    typedef void* psum_field_solver_handle;
    typedef psum_field_solver_handle (*solver_init_func)();
    typedef void (*solver_set_matrix_func)(psum_field_solver_handle, unsigned long long, unsigned long long, unsigned long long, unsigned long long*, unsigned long long*, double*);
    typedef void (*solver_solve_func)(psum_field_solver_handle, double*, double*);
    typedef void (*solver_set_options_func)(psum_field_solver_handle, const char*);
    typedef void (*solver_set_source_replace_func)(psum_field_solver_handle, unsigned long long, unsigned long long*, double*);
    typedef void (*solver_set_source_addback_func)(psum_field_solver_handle, unsigned long long, unsigned long long*, double*);
    typedef void (*solver_free_func)(psum_field_solver_handle);

    struct func_table
    {
        solver_init_func init;
        solver_set_matrix_func set_matrix;
        solver_solve_func solve;
        solver_set_options_func set_options;
        solver_set_source_replace_func set_source_replace;
        solver_set_source_addback_func set_source_addback;
        solver_free_func free;
    };

    #define EXPORT __attribute__((visibility("default")))
    EXPORT void register_solver(const char *name, func_table table);
    EXPORT func_table get_solver_funcs(const char* name);
}

#endif