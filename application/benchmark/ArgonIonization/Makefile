ACPP ?= /opt/AdaptiveCpp/bin/acpp
CPU_FLAGS ?= --acpp-targets=omp.library-only --acpp-cpu-cxx=/usr/lib/llvm-16/bin/clang++
GPU_FLAGS ?= --acpp-targets=generic
FLAGS = -std=c++20 -O2 -I../../../include -I/usr/include/eigen3
HEADERS := $(shell find ../../../include ../../../src -name '*.hpp')
SOURCES = main.cpp population.hpp argon.hpp $(HEADERS)

STEPS ?= 500000
START ?= 150000
PARTICLES ?= 6000
SEED ?= 1234
TD ?= 400
OUT ?= results/$(MODEL)-$(TD)-$(STEPS)-$(PARTICLES)-$(SEED).csv
MODEL = weighted-ar
export OMP_NUM_THREADS ?= 1

.PHONY: all run run-simple run-gpu clean
all: build/argon build/argon-simple
build/argon build/argon-gpu: $(SOURCES)
build/argon-simple: main_simple.cpp $(HEADERS)
build/argon-gpu: CPU_FLAGS = $(GPU_FLAGS)
build/argon build/argon-simple build/argon-gpu:
	mkdir -p build
	$(ACPP) $(CPU_FLAGS) $(FLAGS) $< -o $@

run: build/argon
run-simple: MODEL = simple
run-simple: build/argon-simple
run-gpu: export ACPP_VISIBILITY_MASK = cuda
run-gpu: build/argon-gpu
run run-simple run-gpu:
	mkdir -p "$(dir $(OUT))"
	./$< $(STEPS) $(START) $(PARTICLES) $(SEED) $(TD) "$(OUT)" > "$(OUT).log"

clean:
	rm -f build/argon build/argon-simple build/argon-gpu

-include local/checks.mk
