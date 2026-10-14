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

# CUDA-backed solver test: needs a GPU, excluded from CI
BFS = ../../src/batch_pic/batch_field_solver
FSL = ../../src/field_solver
AGGREGATE_TARGETS += test_cuda_sparselu_mrhs
MRHS_TEST_SRCS = test_cuda_sparselu_mrhs.cpp \
	$(BFS)/register/registry.cpp \
	$(FSL)/register/registry.cpp \
	$(BFS)/implements/cuda_sparselu_mrhs_gpu.cu \
	$(FSL)/implements/cuda_sparselu_gpu.cu

test_cuda_sparselu_mrhs: $(MRHS_TEST_SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -I /usr/include/suitesparse \
		-o test_cuda_sparselu_mrhs $(MRHS_TEST_SRCS) $(LIBS) $(LDFLAGS)
	./test_cuda_sparselu_mrhs
	rm test_cuda_sparselu_mrhs

clean:
	rm -f test_batch_field_consistency test_host_batch_field_consistency test_cuda_sparselu_mrhs
