ACPP ?= /opt/AdaptiveCpp/bin/acpp
TARGETS ?= omp.library-only
CPU_CXX ?= /usr/lib/llvm-16/bin/clang++
CXXFLAGS ?= -O2 -std=c++20
BACKEND = --acpp-targets=$(TARGETS) --acpp-cpu-cxx=$(CPU_CXX)
SOURCES := main.cpp $(wildcard *.hpp) $(shell find ../../../include ../../../src -name '*.hpp')
BINS := $(addprefix build/n2-,flat-plate smoke reference flat-plate-gpu flat-plate-simple-gpu)

.PHONY: all dsmc run smoke reference-pilot gpu simple-gpu run-gpu clean
all dsmc: build/n2-flat-plate
gpu: build/n2-flat-plate-gpu
simple-gpu: build/n2-flat-plate-simple-gpu
build/n2-smoke: CASE_FLAGS = -DN2_CASE_ARGUMENTS=100
build/n2-reference: CASE_FLAGS = '-DN2_CASE_ARGUMENTS=2500,1101,1e11,"results/reference-pilot"'
build/n2-flat-plate-gpu build/n2-flat-plate-simple-gpu: BACKEND = --acpp-targets=generic -DN2_CUDA_MEMORY
build/n2-flat-plate-gpu build/n2-flat-plate-simple-gpu: CUDA_LIB = -lcudart
build/n2-flat-plate-simple-gpu: CASE_FLAGS = -DN2_ANALYTIC_BOUNDARY
$(BINS): $(SOURCES)
	mkdir -p build
	$(ACPP) $(BACKEND) $(CXXFLAGS) $(CASE_FLAGS) -fopenmp -I../../../include -I/usr/include/eigen3 $< $(CUDA_LIB) -lstdc++ -o $@

run: build/n2-flat-plate
smoke: build/n2-smoke
reference-pilot: export OMP_NUM_THREADS = 2
reference-pilot: build/n2-reference
run-gpu: export ACPP_VISIBILITY_MASK = cuda
run-gpu: export CUDA_VISIBLE_DEVICES = 0
run-gpu: build/n2-flat-plate-gpu
run smoke reference-pilot run-gpu:
	./$<

clean:
	rm -f $(BINS)

-include local/checks.mk
