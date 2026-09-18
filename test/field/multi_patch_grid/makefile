-include ../../../config.mk

INCLUDES = -I ../../../include -I ../../../src

TESTS = test_construction test_field test_index test_merge_on_duplicate test_density

test_construction: test_mpg_construction.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)
	./$@
	rm $@

test_field: test_mpg_field.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)
	./$@
	rm $@

test_index: test_mpg_index.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)
	./$@
	rm $@

test_merge_on_duplicate: test_mpg_merge_on_duplicate.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)
	./$@
	rm $@

test_density: test_mpg_density.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o $@ $< $(LIBS) $(LDFLAGS)
	./$@
	rm $@

all: $(TESTS)

.PHONY: all $(TESTS)
