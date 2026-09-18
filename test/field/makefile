-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

test_simplegrid: test_simplegrid.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_simplegrid test_simplegrid.cpp $(LIBS) $(LDFLAGS)
	./test_simplegrid
	rm test_simplegrid

test_simpleinterp: test_simpleinterp.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_simpleinterp test_simpleinterp.cpp $(LIBS) $(LDFLAGS)
	./test_simpleinterp
	rm test_simpleinterp

test_simpleinterpDiff: test_simpleinterpDiff.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_simpleinterpDiff test_simpleinterpDiff.cpp $(LIBS) $(LDFLAGS)
	./test_simpleinterpDiff
	rm test_simpleinterpDiff

test_field_interp: test_field_interp.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_field_interp test_field_interp.cpp $(LIBS) $(LDFLAGS)
	./test_field_interp
	rm test_field_interp

test_field_interpDiff: test_field_interpDiff.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_field_interpDiff test_field_interpDiff.cpp $(LIBS) $(LDFLAGS)
	./test_field_interpDiff
	rm test_field_interpDiff

test_host_field_interp: test_host_field_interp.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_host_field_interp test_host_field_interp.cpp $(LIBS) $(LDFLAGS)
	./test_host_field_interp
	rm test_host_field_interp

test_host_field_interpDiff: test_host_field_interpDiff.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test_host_field_interpDiff test_host_field_interpDiff.cpp $(LIBS) $(LDFLAGS)
	./test_host_field_interpDiff
	rm test_host_field_interpDiff

test_cov:
	./../coverage_utils/make_coverage_test.sh \
		test_simplegrid.cpp \
		test_simpleinterp.cpp \
		test_simpleinterpDiff.cpp \
		test_field_interp.cpp \
		test_field_interpDiff.cpp \
		test_host_field_interp.cpp \
		test_host_field_interpDiff.cpp