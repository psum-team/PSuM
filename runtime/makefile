# PSuM C runtime library: builds bin/psumrt.so from src/psum_runtime.cpp.
#   make            - build bin/psumrt.so
#   make UNIVERSE_FLAGS='-DPSUM_DIM_MASK=1 -DPSUM_E_ORDER=2' - custom universe
-include ../config.mk

INCLUDES = -I ../include -I ../src -I src

ACPP ?= acpp

UNIVERSE_FLAGS ?=

bin/psumrt.so: src/psum_runtime.cpp
	mkdir -p bin
	$(ACPP) $(COMMON_FLAGS) -fPIC $(INCLUDES) -shared -Wl,-soname,psumrt.so -o bin/psumrt.so src/psum_runtime.cpp \
		$(USE_BACKENDS) $(LIBS) $(LDFLAGS) $(UNIVERSE_FLAGS)

.PHONY: all
all: bin/psumrt.so
