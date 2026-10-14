#ifndef PSUM_BATCH_PIC_BATCH_FIELD_SOLVER_REGISTER_INTERFACE_H
#define PSUM_BATCH_PIC_BATCH_FIELD_SOLVER_REGISTER_INTERFACE_H

extern "C" {
    typedef void* psum_batch_field_solver_handle;
    typedef psum_batch_field_solver_handle (*batch_solver_init_func)();
    typedef void (*batch_solver_set_matrix_func)(psum_batch_field_solver_handle, unsigned long long, unsigned long long, unsigned long long, unsigned long long*, unsigned long long*, double*);
    typedef void (*batch_solver_solve_func)(psum_batch_field_solver_handle, double*, double*);
    typedef void (*batch_solver_set_options_func)(psum_batch_field_solver_handle, const char*);
    typedef void (*batch_solver_set_source_replace_func)(psum_batch_field_solver_handle, unsigned long long, unsigned long long, unsigned long long*, double*);
    typedef void (*batch_solver_set_source_addback_func)(psum_batch_field_solver_handle, unsigned long long, unsigned long long, unsigned long long*, double*);
    typedef void (*batch_solver_free_func)(psum_batch_field_solver_handle);

    struct batch_func_table
    {
        batch_solver_init_func init;
        batch_solver_set_matrix_func set_matrix;
        batch_solver_solve_func solve;
        batch_solver_set_options_func set_options;
        batch_solver_set_source_replace_func set_source_replace;
        batch_solver_set_source_addback_func set_source_addback;
        batch_solver_free_func free;
    };

    #define EXPORT __attribute__((visibility("default")))
    EXPORT void register_batch_solver(const char *name, batch_func_table table);
    EXPORT batch_func_table get_batch_solver_funcs(const char* name);
}

#endif
