COULOMB = test9 test_coulomb_pool test_coulomb_relaxation
MCC_DPMCC = test6 test_dpmcc_stages test_dpmcc_requests test_reaction_network test7 test10 test_frozen_beams test_n2_relaxation
AGGREGATE_TARGETS = test6|test_dpmcc_stages|test_dpmcc_requests|test_reaction_network|test7|test9|test10|test_coulomb_relaxation|test_coulomb_pool|test_frozen_beams|test_n2_relaxation
.DEFAULT_GOAL := test6

all: $(MCC_DPMCC) $(COULOMB)

$(COULOMB):
	$(MAKE) -C coulomb_collision $@

$(MCC_DPMCC):
	$(MAKE) -C mcc_dpmcc $@

.PHONY: all $(COULOMB) $(MCC_DPMCC)
