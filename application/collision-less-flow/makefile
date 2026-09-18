-include ../../config.mk

.PHONY: run test clean

main: main.cpp *.hpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS) \
	-I ../../include \
	-o main 

test_gamma_config: test_gamma_config.cpp *.hpp
	$(ACPP) $(COMMON_FLAGS) test_gamma_config.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS) \
	-I ../../include \
	-o test_gamma_config

run: main config.json
	mkdir -p case_test
	cp main config.json case_test/
	cd case_test && ./main

test: test_gamma_config
	./test_gamma_config

clean:
	rm -f main test_gamma_config test_gamma_absent.json test_gamma_one.json test_gamma_bad.json
