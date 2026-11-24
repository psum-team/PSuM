-include ../../config.mk

INCLUDES = -I ../../include -I ../../src -I ../src

UNIVERSE_FLAGS ?=

# ---------- direct-header tests (no engine, acpp compile) ----------
test_push: test_push_variant.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_push test_push_variant.cpp $(LIBS) $(LDFLAGS)
	./test_push
	rm test_push

test_poisson: test_poisson_variant.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_poisson test_poisson_variant.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	./test_poisson
	rm test_poisson


# ---------- physics cases (C interface + psumrt.so) ----------

tsi: tsi.cpp ../bin/psumrt.so
	$(CXX) -o tsi tsi.cpp ../bin/psumrt.so -Wl,-rpath,$(abspath ../bin) -std=c++20 -O2 $(INCLUDES)

sheath: sheath.cpp ../bin/psumrt.so
	$(CXX) -o sheath sheath.cpp ../bin/psumrt.so -Wl,-rpath,$(abspath ../bin) -std=c++20 -O2 $(INCLUDES)

sphere_rz: sphere_rz.cpp ../bin/psumrt.so
	$(CXX) -o sphere_rz sphere_rz.cpp ../bin/psumrt.so -Wl,-rpath,$(abspath ../bin) -std=c++20 -O2 $(INCLUDES)

# ---------- aggregate ----------

check: tsi sheath sphere_rz test_push test_poisson
	./tsi && ./sheath 2000 && ./sphere_rz

.PHONY: check test_push test_poisson
