-include ../../../config.mk


BATCH_FIELD_SOLVER_BIN = $(PROJECT_ROOT)/src/batch_pic/batch_field_solver/bin

.PHONY: all impls
all:
	$(MAKE) clean
	$(MAKE) $(BATCH_FIELD_SOLVER_BIN)/registry.o
	$(MAKE) $(BATCH_FIELD_SOLVER_BIN)/impls.o

$(BATCH_FIELD_SOLVER_BIN)/registry.o: register/registry.cpp
	mkdir -p bin
	$(ACPP) $(COMMON_FLAGS) -c register/registry.cpp -o bin/registry.o

$(BATCH_FIELD_SOLVER_BIN)/%.o: implements/%.cpp
	mkdir -p bin
	$(CXX) -O3 -std=c++20 -fopenmp -c $< -fPIC -o $@

$(BATCH_FIELD_SOLVER_BIN)/sycl_%.o: implements/sycl_%.cpp
	mkdir -p bin
	$(ACPP) $(COMMON_FLAGS) -c $< -fPIC -o $@

$(BATCH_FIELD_SOLVER_BIN)/%.o: implements/%.cu
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -c $< -o $@

$(BATCH_FIELD_SOLVER_BIN)/impls.o: $(PSUM_BATCH_SOLVER_BACKENDS)
	$(CXX) -nostdlib -r $^ -o $@
	rm -f $(PSUM_BATCH_SOLVER_BACKENDS)

clean:
	rm -rf $(BATCH_FIELD_SOLVER_BIN)

impls: $(BATCH_FIELD_SOLVER_BIN)/impls.o
