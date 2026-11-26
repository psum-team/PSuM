include ../../../config.mk

dsmc_headers = ../../../application/benchmark/DSMC/custom_dpmcc.hpp ../../../application/benchmark/DSMC/n2_vhs.hpp

test6: test_dpmcc_context.cpp dpmcc_test_models.hpp
test_dpmcc_stages: test_dpmcc_stages.cpp dpmcc_test_models.hpp
test_dpmcc_requests: test_dpmcc_requests.cpp dpmcc_test_models.hpp
test_reaction_network: reaction_network/network.cpp
test7: test_simple_dsmc.cpp $(dsmc_headers)
test10: test_axisymmetric_dsmc.cpp
test_frozen_beams: test_frozen_beams.cpp frozen_beam_rates.hpp frozen_pair_order.hpp $(dsmc_headers)
test_n2_relaxation: test_n2_relaxation.cpp $(dsmc_headers)

targets = test6 test_dpmcc_stages test_dpmcc_requests test_reaction_network test7 test10 test_frozen_beams test_n2_relaxation
$(targets):
	$(ACPP) $(CPPFLAGS) -o $@ $< -std=c++20 -O3 -I ../../../include
	mkdir -p build
	mv $@ build/
	cd build && ./$@

.PHONY: $(targets)
