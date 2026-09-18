-include ../../config.mk

INCLUDES = -I ../../include -I ../../src -I ../../third_party

test1: test_mas.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test1 test_mas.cpp $(LIBS) $(LDFLAGS)
	./test1
	rm test1

test2: test_mas_complex.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test2 test_mas_complex.cpp $(LIBS) $(LDFLAGS)
	./test2
	rm test2

test3: test_tag_sl.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test3 test_tag_sl.cpp $(LIBS) $(LDFLAGS)
	./test3
	rm test3

test4: test_container_sl.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test4 test_container_sl.cpp $(LIBS) $(LDFLAGS)
	./test4
	rm test4

test5: test_objmng.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test5 test_objmng.cpp $(LIBS) $(LDFLAGS)
	./test5
	rm test5

test6: test_objmng_complex.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test6 test_objmng_complex.cpp $(LIBS) $(LDFLAGS)
	./test6
	rm test6

test7: test_json_loader.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o test7 test_json_loader.cpp $(LIBS) $(LDFLAGS)
	./test7
	rm test7

test_cov: 
	./../coverage_utils/make_coverage_test.sh \
	test_mas.cpp \
	test_mas_complex.cpp \
	test_tag_sl.cpp \
	test_container_sl.cpp \
	test_objmng.cpp \
	test_objmng_complex.cpp \
	test_json_loader.cpp