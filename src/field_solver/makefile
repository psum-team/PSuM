-include ../../config.mk

# clean must finish before test_backends builds; keep it serial under `make -j`.
.PHONY: all
all:
	$(MAKE) clean
	$(MAKE) test_backends

$(FIELD_SOLVER_BIN)/registry.o: register/registry.cpp
	mkdir -p bin
	$(ACPP) $(COMMON_FLAGS) -c register/registry.cpp -o bin/registry.o

$(FIELD_SOLVER_BIN)/%.o: implements/%.cpp
	mkdir -p bin
	$(CXX) -O3 -std=c++20 -fopenmp -c $< -fPIC -o $@

$(FIELD_SOLVER_BIN)/sycl_%.o: implements/sycl_%.cpp
	mkdir -p bin
	$(ACPP) $(COMMON_FLAGS) -I../../include -I../../src -c $< -fPIC -o $@

$(FIELD_SOLVER_BIN)/%.o: implements/%.cu
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fopenmp -c $< -o $@

$(FIELD_SOLVER_BIN)/impls.o: $(PSUM_FIELD_SOLVER_BACKENDS)
	$(CXX) -nostdlib -r $^ -o $@
	rm -f $(PSUM_FIELD_SOLVER_BACKENDS)

test_backends: test_backends.cpp $(FIELD_SOLVER_BIN)/impls.o $(FIELD_SOLVER_BIN)/registry.o
	$(ACPP) $(COMMON_FLAGS) -I../../include -I../../src test_backends.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS) \
	-o test_app
	./test_app
	rm -f test_app

clean:
	rm -f *.o test_app bin/*
