include ../../../config.mk

test9: test_coulomb_collision.cpp
test_coulomb_pool: test_coulomb_pool.cpp
test_coulomb_relaxation: $(if $(filter --full,$(COULOMB_RELAXATION_ARGS)),paper_temperature_relaxation.cpp,test_temperature_relaxation.cpp) calc_tau.hpp

targets = test9 test_coulomb_pool test_coulomb_relaxation
$(targets):
	$(ACPP) $(CPPFLAGS) -o $@ $< -std=c++20 -O3 -I ../../../include
	mkdir -p build
	mv $@ build/
	cd build && ./$@ $(if $(filter test_coulomb_relaxation,$@),$(COULOMB_RELAXATION_ARGS))

.PHONY: $(targets)
