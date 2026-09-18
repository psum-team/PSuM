-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

test1: tagged_struct_functional.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) tagged_struct_functional.cpp -o test1 $(LIBS) $(LDFLAGS)
	./test1
	rm test1