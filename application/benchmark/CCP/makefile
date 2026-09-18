-include ../../../config.mk

ccp: main.cpp
	$(ACPP) $(COMMON_FLAGS) main.cpp $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o ccp
	mkdir -p output
	./ccp

clean:
	rm -f ccp output/*.plt numline_xy.plt

.PHONY: ccp clean
