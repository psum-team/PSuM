-include ../../../config.mk

dsmc: main.cpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o dsmc
	mkdir -p output
	./dsmc

clean:
	rm -f dsmc output/*.plt

.PHONY: dsmc clean
