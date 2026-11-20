-include ../../config.mk

LZMA_VENDOR_O = $(PROJECT_ROOT)/src/serialization/bin/lzma_vendor.o

$(LZMA_VENDOR_O): lzma_vendor.c \
		../../third_party/lzma/LzmaEnc.c ../../third_party/lzma/LzmaDec.c \
		../../third_party/lzma/LzFind.c
	mkdir -p $(dir $(LZMA_VENDOR_O))
	$(CC) -O2 -fPIC -I../../third_party/lzma -c lzma_vendor.c -o $@

.PHONY: all
all: $(LZMA_VENDOR_O)
