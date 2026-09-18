-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

test_dvec:test_dvec.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_dvec.cpp -o test_dvec $(LIBS) $(LDFLAGS)
	./test_dvec
	rm test_dvec

test_pg:test_pg.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_pg.cpp -o test_pg $(LIBS) $(LDFLAGS)
	./test_pg
	rm test_pg

test_sl:test_sl.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_sl.cpp -o test_sl $(LIBS) $(LDFLAGS)
	./test_sl
	rm test_sl

test_pg_complex:test_pg_complex.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_pg_complex.cpp -o test_pg_complex $(LIBS) $(LDFLAGS)
	./test_pg_complex
	rm test_pg_complex

test_pg_shuffle:test_pg_shuffle.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) test_pg_shuffle.cpp -o test_pg_shuffle $(LIBS) $(LDFLAGS)
	./test_pg_shuffle
	rm test_pg_shuffle

test_cov: 
	./../coverage_utils/make_coverage_test.sh \
	test_dvec.cpp \
	test_pg.cpp \
	test_sl.cpp