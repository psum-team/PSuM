-include ../../../config.mk

edi: EDI.cpp
	$(ACPP) $(COMMON_FLAGS) EDI.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS) \
	-I ../../../include \
	-o edi
	mkdir -p output

clean:
	rm -f edi

.PHONY: clean edi 
