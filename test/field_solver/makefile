-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

test: check_solvers.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_app check_solvers.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
		./test_app
	rm test_app

test_fb: test_fixed_boundary.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_fb test_fixed_boundary.cpp $(LIBS) $(LDFLAGS)
	./test_fb
	rm test_fb

test_mb: test_mixed_boundary.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_mb test_mixed_boundary.cpp $(LIBS) $(LDFLAGS)
	./test_mb
	rm test_mb

test_ghost_mng: test_ghost_mng.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_ghost_mng test_ghost_mng.cpp $(LIBS) $(LDFLAGS)
	./test_ghost_mng
	rm test_ghost_mng

testpsolver2d:test_psolver_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_psolver_2d.cpp -o testpsolver2d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./testpsolver2d
	rm testpsolver2d

testpsolver3d:test_psolver_3d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_psolver_3d.cpp -o testpsolver3d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./testpsolver3d
	rm testpsolver3d

testpsolver1d:test_psolver_1d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_psolver_1d.cpp -o testpsolver1d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./testpsolver1d
	rm testpsolver1d

test_hall_thruster: test_hall_thruster_case.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_hall_thruster_case.cpp -o test_hall_thruster $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_hall_thruster
	rm test_hall_thruster

test_mg_core: test_multigrid_core.cpp
	g++ -std=c++20 -O3 -o test_mg_core test_multigrid_core.cpp -fopenmp -lstdc++ $(INCLUDES)
	./test_mg_core
	rm test_mg_core

test_conv_matvec: test_conv_sparse_matrix_matvec.cpp
	g++ -std=c++20 -O3 -o test_conv_matvec test_conv_sparse_matrix_matvec.cpp -fopenmp -lstdc++ $(INCLUDES)
	./test_conv_matvec
	rm test_conv_matvec

test_spmv_calculator: test_spmv_calculator.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_spmv_calculator.cpp -o test_spmv_calculator $(LIBS) $(LDFLAGS)
	./test_spmv_calculator
	rm test_spmv_calculator

test_eigen_schurcomplement_cpu: test_eigen_schurcomplement_cpu.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_eigen_schurcomplement_cpu.cpp -o test_eigen_schurcomplement_cpu $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_eigen_schurcomplement_cpu
	rm test_eigen_schurcomplement_cpu

test_native_vs_mg_2d:test_native_vs_multigrid_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_native_vs_multigrid_2d.cpp -o test_native_vs_mg_2d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_native_vs_mg_2d
	rm test_native_vs_mg_2d

testpsolver3d_multigrid:test_psolver_3d_multigrid.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_psolver_3d_multigrid.cpp -o testpsolver3d_multigrid $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./testpsolver3d_multigrid
	rm testpsolver3d_multigrid

test_manufactured_solution_2d: test_manufactured_solution_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_manufactured_solution_2d.cpp -o test_manufactured_solution_2d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_manufactured_solution_2d
	rm test_manufactured_solution_2d

test_fem_manufactured: test_fem_manufactured.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_fem_manufactured.cpp -o test_fem_manufactured $(USE_BACKENDS) $(LIBS) $(LDFLAGS) -I ../../src/field_solver/fem_solver
	./test_fem_manufactured
	rm test_fem_manufactured

test_native_vs_mg_2d_manufactured: test_native_vs_mg_2d_manufactured.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_native_vs_mg_2d_manufactured.cpp -o test_native_vs_mg_2d_manufactured $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_native_vs_mg_2d_manufactured
	rm test_native_vs_mg_2d_manufactured

test_native_vs_mg_3d:test_native_vs_multigrid_3d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_native_vs_multigrid_3d.cpp -o test_native_vs_mg_3d $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_native_vs_mg_3d
	rm test_native_vs_mg_3d

test_fem_poisson_2d: test_fem_poisson_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_fem_poisson_2d.cpp -o test_fem_poisson_2d $(USE_BACKENDS) $(LIBS) $(LDFLAGS) -I ../../src/field_solver/fem_solver
	./test_fem_poisson_2d
	rm test_fem_poisson_2d

test_mpg_to_trimesh: test_mpg_to_trimesh.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_mpg_to_trimesh.cpp -o test_mpg_to_trimesh $(USE_BACKENDS) $(LIBS) $(LDFLAGS) -I ../../src/field_solver/fem_solver
	./test_mpg_to_trimesh
	rm test_mpg_to_trimesh

test_mpg_solver_compare: test_mpg_solver_compare.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_mpg_solver_compare.cpp -o test_mpg_solver_compare $(USE_BACKENDS) $(LIBS) $(LDFLAGS) -I ../../src/field_solver/fem_solver
	./test_mpg_solver_compare
	rm test_mpg_solver_compare

# Aggregate target (not a standalone executable test): runs all multigrid
# cases in one go. Declared via AGGREGATE_TARGETS so that runCheck.sh can
# exclude it automatically (its output name would otherwise collide with the
# first prerequisite and inflate the total count).
AGGREGATE_TARGETS = test_multigrid

test_schur_core: test_schur_complement_core.cpp
	g++ -std=c++20 -O3 -o test_schur_core test_schur_complement_core.cpp -fopenmp -lstdc++ $(INCLUDES)
	./test_schur_core
	rm test_schur_core

test_schur_state_save_load: test_schur_state_save_load.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_schur_state_save_load.cpp -o test_schur_state_save_load $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_schur_state_save_load
	rm test_schur_state_save_load

test_device_mg_compare: test_device_multigrid_compare.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_device_multigrid_compare.cpp -o test_device_mg_compare $(USE_BACKENDS) $(LIBS) $(LDFLAGS)

test_multigrid: test_mg_core test_conv_matvec test_native_vs_mg_2d test_native_vs_mg_3d
