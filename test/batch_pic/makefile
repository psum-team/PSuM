-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

test_batch_field_consistency: test_batch_field_consistency.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_batch_field_consistency test_batch_field_consistency.cpp $(LIBS) $(LDFLAGS)
	./test_batch_field_consistency
	rm test_batch_field_consistency

test_host_batch_field_consistency: test_host_batch_field_consistency.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_host_batch_field_consistency test_host_batch_field_consistency.cpp $(LIBS) $(LDFLAGS)
	./test_host_batch_field_consistency
	rm test_host_batch_field_consistency

AGGREGATE_TARGETS += test_cuda_sparselu_mrhs

test_batch_poisson_solver_1d: test_batch_poisson_solver_1d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_batch_poisson_solver_1d test_batch_poisson_solver_1d.cpp $(LIBS) $(LDFLAGS)
	./test_batch_poisson_solver_1d
	rm -f test_batch_poisson_solver_1d

test_batch_poisson_solver_2d: test_batch_poisson_solver_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_batch_poisson_solver_2d test_batch_poisson_solver_2d.cpp $(USE_BACKENDS) $(USE_BATCH_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_batch_poisson_solver_2d
	rm -f test_batch_poisson_solver_2d

test_cuda_sparselu_mrhs: test_batch_poisson_solver_2d.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_cuda_sparselu_mrhs test_batch_poisson_solver_2d.cpp $(USE_BACKENDS) $(USE_BATCH_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_cuda_sparselu_mrhs cuda_sparselu_gpu cuda_sparselu_mrhs_gpu
	rm -f test_cuda_sparselu_mrhs

clean:
	rm -f test_batch_field_consistency test_host_batch_field_consistency test_cuda_sparselu_mrhs test_batch_poisson_solver_2d test_batch_poisson_solver_1d
