#ifndef PSUM_SERIALIZATION_LZMA_COMPRESSOR_HPP
#define PSUM_SERIALIZATION_LZMA_COMPRESSOR_HPP

// LZMA compression core, built on the vendored public-domain LZMA SDK (third_party/lzma). 
// This file provides lzma_compressor, the lzma implementation of the abstract compressor (compressor.hpp)
// One translation unit per program must compile lzma_vendor.c.

#include "compressor.hpp"
#include <cstdint>
#include <unistd.h>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

extern "C" {
#include "../../third_party/lzma/LzmaEnc.h"
#include "../../third_party/lzma/LzmaDec.h"
}

namespace psum {

namespace serialization {

    namespace lzma_detail {

        struct vendor_alloc {
            static void* alloc(const ISzAlloc*, size_t size) { return std::malloc(size); }
            static void free_mem(const ISzAlloc*, void* addr) { std::free(addr); }
        };

        inline ISzAlloc& vendor_allocator() {
            static ISzAlloc instance = { &vendor_alloc::alloc, &vendor_alloc::free_mem };
            return instance;
        }

        // stdio-backed ISeqInStream / ISeqOutStream for streaming encode.
        // The stream object is the first member, so a pointer to it can be
        // cast back to the adapter.
        struct c_file_in_stream {
            ISeqInStream base;
            std::FILE* file;
            static SRes read_impl(const ISeqInStream* p, void* buf, size_t* size) {
                c_file_in_stream* self = (c_file_in_stream*)p;
                *size = std::fread(buf, 1, *size, self->file);
                return *size == 0 && std::ferror(self->file) ? SZ_ERROR_READ : SZ_OK;
            }
            explicit c_file_in_stream(std::FILE* f) : file(f) {
                base.Read = &read_impl;
            }
        };
        struct c_file_out_stream {
            ISeqOutStream base;
            std::FILE* file;
            static size_t write_impl(const ISeqOutStream* p, const void* buf, size_t size) {
                return std::fwrite(buf, 1, size, ((c_file_out_stream*)p)->file);
            }
            explicit c_file_out_stream(std::FILE* f) : file(f) {
                base.Write = &write_impl;
            }
        };

        // file-level API for lzma compression

        inline std::filesystem::path lzma_name(const std::filesystem::path& file_path) {
            return file_path.string() + ".lzma";
        }

        struct options_t {
            unsigned level = 4;
            size_t dict_size = 0;    // 0 keeps the built-in 4 MiB default
        };

        inline options_t parse_options(const std::string& options) {
            options_t o;
            size_t pos = 0;
            while (pos < options.size()) {
                size_t end = options.find_first_of(",;", pos);
                if (end == std::string::npos)
                    end = options.size();
                const std::string item = options.substr(pos, end - pos);
                pos = end + 1;
                const size_t eq = item.find('=');
                if (eq == std::string::npos)
                    throw std::runtime_error("lzma_compressor: bad option item: " + item);
                const std::string key = item.substr(0, eq);
                const std::string val = item.substr(eq + 1);
                size_t idx = 0;
                unsigned long long num = 0;
                try {
                    num = std::stoull(val, &idx);
                } catch (const std::exception&) {
                    throw std::runtime_error("lzma_compressor: bad option value: " + item);
                }
                unsigned long long mult = 1;
                if (idx < val.size()) {
                    const char suffix = (char)std::tolower((unsigned char)val[idx]);
                    if (idx + 1 != val.size())
                        throw std::runtime_error("lzma_compressor: bad option value: " + item);
                    if (suffix == 'k') mult = 1ull << 10;
                    else if (suffix == 'm') mult = 1ull << 20;
                    else if (suffix == 'g') mult = 1ull << 30;
                    else throw std::runtime_error("lzma_compressor: bad option value: " + item);
                }
                if (key == "level") {
                    if (num > 9 || mult != 1)
                        throw std::runtime_error("lzma_compressor: level must be 0..9");
                    o.level = (unsigned)num;
                } else if (key == "dict") {
                    if (num * mult >= (1ull << 32))
                        throw std::runtime_error("lzma_compressor: dict must be < 4 GiB");
                    o.dict_size = (size_t)(num * mult);
                } else {
                    throw std::runtime_error("lzma_compressor: unknown option key: " + key);
                }
            }
            return o;
        }

        // plain file -> .lzma compressed file (published atomically: compress to a
        // temp name, rename into place, so a killed run never leaves a
        // truncated compressed file behind)
        inline void compress_file(const std::filesystem::path& file_path,
                                const std::string& options = "") {
            namespace lz = lzma_detail;
            const options_t opt = parse_options(options);
            const std::string final_name = lzma_name(file_path).string();
            const std::string tmp_name = final_name + ".tmp" + std::to_string(::getpid());
            std::FILE* in = std::fopen(file_path.string().c_str(), "rb");
            if (!in)
                throw std::runtime_error("lzma_compressor: cannot open source file");
            std::FILE* out = std::fopen(tmp_name.c_str(), "wb");
            if (!out) {
                std::fclose(in);
                throw std::runtime_error("lzma_compressor: cannot open destination file");
            }
            try {
                const std::uint64_t src_size =
                    (std::uint64_t)std::filesystem::file_size(file_path);

                CLzmaEncProps props;
                LzmaEncProps_Init(&props);
                props.level = (int)opt.level;
                props.dictSize = (UInt32)(opt.dict_size ? opt.dict_size
                                                        : (size_t)4 * 1024 * 1024);
                props.writeEndMark = 0;

                CLzmaEncHandle enc = LzmaEnc_Create(&lz::vendor_allocator());
                if (!enc)
                    throw std::runtime_error("lzma_compressor: encoder create failed");
                SRes res = LzmaEnc_SetProps(enc, &props);
                if (res == SZ_OK)
                    LzmaEnc_SetDataSize(enc, src_size);
                unsigned char props_buf[LZMA_PROPS_SIZE];
                SizeT props_size = LZMA_PROPS_SIZE;
                if (res == SZ_OK)
                    res = LzmaEnc_WriteProperties(enc, props_buf, &props_size);
                if (res != SZ_OK) {
                    LzmaEnc_Destroy(enc, &lz::vendor_allocator(), &lz::vendor_allocator());
                    throw std::runtime_error("lzma_compressor: encoder props failed");
                }

                // .lzma header: props + uncompressed size (LE)
                unsigned char header[LZMA_PROPS_SIZE + 8];
                std::memcpy(header, props_buf, LZMA_PROPS_SIZE);
                for (int b = 0; b < 8; b++)
                    header[LZMA_PROPS_SIZE + b] =
                        (unsigned char)((src_size >> (8 * b)) & 0xFF);
                if (std::fwrite(header, 1, sizeof(header), out) != sizeof(header))
                    throw std::runtime_error("lzma_compressor: short header write");

                lz::c_file_in_stream in_stream(in);
                lz::c_file_out_stream out_stream(out);
                res = LzmaEnc_Encode(enc, &out_stream.base, &in_stream.base, NULL,
                                    &lz::vendor_allocator(), &lz::vendor_allocator());
                LzmaEnc_Destroy(enc, &lz::vendor_allocator(), &lz::vendor_allocator());
                if (res != SZ_OK || std::fflush(out) != 0)
                    throw std::runtime_error("lzma_compressor: encode failed");
            }
            catch (...) {
                std::fclose(in);
                std::fclose(out);
                std::error_code rm;
                std::filesystem::remove(tmp_name, rm);
                throw;
            }
            std::fclose(in);
            if (std::fclose(out) != 0) {
                std::error_code rm;
                std::filesystem::remove(tmp_name, rm);
                throw std::runtime_error("lzma_compressor: close failed");
            }
            std::error_code ec;
            std::filesystem::rename(tmp_name, final_name, ec);
            if (ec) {
                std::filesystem::remove(tmp_name, ec);
                throw std::runtime_error("lzma_compressor: publish failed");
            }
        }

        // .lzma compressed file -> plain file
        inline void decompress_file(const std::filesystem::path& file_path) {
            namespace lz = lzma_detail;
            std::filesystem::path src_path = lzma_name(file_path);
            std::FILE* in = std::fopen(src_path.string().c_str(), "rb");
            if (!in)
                throw std::runtime_error("lzma_compressor: cannot open source file");
            std::FILE* out = std::fopen(file_path.string().c_str(), "wb");
            if (!out) {
                std::fclose(in);
                throw std::runtime_error("lzma_compressor: cannot open destination file");
            }
            try {
                unsigned char header[LZMA_PROPS_SIZE + 8];
                if (std::fread(header, 1, sizeof(header), in) != sizeof(header))
                    throw std::runtime_error("lzma_compressor: short header read");
                std::uint64_t out_size = 0;   // little-endian size field
                for (int b = 0; b < 8; b++)
                    out_size |= (std::uint64_t)header[LZMA_PROPS_SIZE + b] << (8 * b);

                CLzmaDec state;
                LzmaDec_Construct(&state);
                SRes res = LzmaDec_Allocate(&state, header, LZMA_PROPS_SIZE,
                                            &lz::vendor_allocator());
                if (res != SZ_OK)
                    throw std::runtime_error("lzma_compressor: decoder props failed");
                LzmaDec_Init(&state);

                std::vector<unsigned char> in_buf(1 << 20);
                std::uint64_t written = 0;
                SizeT dic_dumped = 0;
                SizeT in_len = 0, in_pos = 0;
                while (written < out_size) {
                    // give the decoder a chance first: after a ring handback it
                    // can emit pending output from input it already consumed
                    // (the range decoder reads ahead), so only refill once no
                    // forward progress is possible
                    const SizeT dic_limit =
                        (SizeT)std::min<std::uint64_t>(
                            (std::uint64_t)state.dicBufSize,
                            (std::uint64_t)state.dicPos + (out_size - written));
                    ELzmaStatus status = LZMA_STATUS_NOT_SPECIFIED;
                    const SizeT dic_dumped_at_entry = dic_dumped;
                    SizeT consumed = in_len - in_pos;
                    res = LzmaDec_DecodeToDic(&state, dic_limit,
                                            in_buf.data() + in_pos, &consumed,
                                            LZMA_FINISH_ANY, &status);
                    in_pos += consumed;
                    // dump the newly produced dictionary window (wrap-aware)
                    SizeT d = state.dicPos;
                    if (d < dic_dumped) {            // wrapped
                        if (std::fwrite(state.dic + dic_dumped, 1,
                                        state.dicBufSize - dic_dumped, out)
                                != state.dicBufSize - dic_dumped
                            || std::fwrite(state.dic, 1, d, out) != d)
                            throw std::runtime_error("lzma_compressor: short write");
                        written += state.dicBufSize - dic_dumped + d;
                    } else if (d > dic_dumped) {
                        if (std::fwrite(state.dic + dic_dumped, 1, d - dic_dumped, out)
                            != (size_t)(d - dic_dumped))
                            throw std::runtime_error("lzma_compressor: short write");
                        written += d - dic_dumped;
                    }
                    const bool dic_advanced = (d != dic_dumped_at_entry);
                    dic_dumped = d;
                    if (state.dicPos == state.dicBufSize) {
                        // ring handback: dictionary window is full, reuse it
                        // from the start (decoder tracks its own history)
                        state.dicPos = 0;
                        dic_dumped = 0;
                    }
                    if (res != SZ_OK)
                        throw std::runtime_error("lzma_compressor: decode failed");
                    if (written >= out_size)
                        break;
                    if (consumed == 0 && !dic_advanced && in_pos == in_len) {
                        in_len = (SizeT)std::fread(in_buf.data(), 1, in_buf.size(), in);
                        in_pos = 0;
                        if (in_len == 0)
                            throw std::runtime_error("lzma_compressor: truncated stream");
                    }
                }
                if (written != out_size)
                    throw std::runtime_error("lzma_compressor: size mismatch");
                if (std::fflush(out) != 0)
                    throw std::runtime_error("lzma_compressor: flush failed");
                LzmaDec_Free(&state, &lz::vendor_allocator());
            }
            catch (...) {
                std::fclose(in);
                std::fclose(out);
                throw;
            }
            std::fclose(in);
            if (std::fclose(out) != 0)
                throw std::runtime_error("lzma_compressor: close failed");
        }

    }

    // adapter binding the free functions above to the abstract codec interface
    struct lzma_compressor final : psum::serialization::compressor {
        void compress(const std::filesystem::path& file_path,
                      const std::string& options = "") const override {
            lzma_detail::compress_file(file_path, options);
        }
        void decompress(const std::filesystem::path& file_path) const override {
            lzma_detail::decompress_file(file_path);
        }
        std::filesystem::path compressed_path(const std::filesystem::path& file_path) const override {
            return lzma_detail::lzma_name(file_path);
        }
        std::shared_ptr<const psum::serialization::compressor> clone() const override {
            return std::make_shared<lzma_compressor>(*this);
        }
    };


}

}

#endif
