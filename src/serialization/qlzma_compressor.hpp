#ifndef PSUM_SERIALIZATION_QLZMA_COMPRESSOR_HPP
#define PSUM_SERIALIZATION_QLZMA_COMPRESSOR_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <unistd.h>

#include "compressor.hpp"
#include "lzma_compressor.hpp"
#include "mas.hpp"

namespace psum {

namespace serialization {

    namespace qlzma_detail {
        // fixed quantization gears: mantissa bits kept per double;
        // relative error:
        //   40 -> ~1e-12 
        //   32 -> ~1e-10    24 -> ~1e-8    
        //   20 -> ~1e-7     16 -> ~1e-5
        enum class gear { q40, q32, q24, q20, q16 };

        // split an options string into the quantize gear ("q=40|32|24|20|16",
        // default q20) and the remainder forwarded to the LZMA stage
        inline std::pair<gear, std::string> split_options(const std::string& options) {
            gear g = gear::q20;
            std::string rest;
            size_t pos = 0;
            while (pos < options.size()) {
                size_t end = options.find_first_of(",;", pos);
                if (end == std::string::npos) end = options.size();
                std::string token = options.substr(pos, end - pos);
                pos = end + 1;
                size_t eq = token.find('=');
                if (eq != std::string::npos && token.substr(0, eq) == "q") {
                    const std::string val = token.substr(eq + 1);
                    if (val == "40") g = gear::q40;
                    else if (val == "32") g = gear::q32;
                    else if (val == "24") g = gear::q24;
                    else if (val == "20") g = gear::q20;
                    else if (val == "16") g = gear::q16;
                    else throw std::runtime_error(
                        "qlzma_compressor: q must be one of 40, 32, 24, 20, 16");
                } else if (!token.empty()) {
                    if (!rest.empty()) rest += ",";
                    rest += token;
                }
            }
            return {g, rest};
        }

        // in-place mantissa rounding of a double buffer
        inline void quantize(double* v, size_t n, gear g) {
            uint64_t half, keep;
            switch (g) {
            case gear::q40: half = 1ull << 11; break;
            case gear::q32: half = 1ull << 19; break;
            case gear::q24: half = 1ull << 27; break;
            case gear::q20: half = 1ull << 31; break;
            default:        half = 1ull << 35; break;
            }
            keep = ~((half << 1) - 1);
            for (size_t i = 0; i < n; ++i) {
                uint64_t u;
                std::memcpy(&u, &v[i], sizeof u);
                if (((u >> 52) & 0x7ffu) != 0x7ffu) {
                    u = (u + half) & keep;
                    std::memcpy(&v[i], &u, sizeof u);
                }
            }
        }

        // load `src`, quantize every double block holding at least
        // `min_doubles` elements, and save the copy as `dst`; all other
        // blocks are copied verbatim in original order, so scalars,
        // metadata and integer indices stay exact
        inline void quantize_mas(const std::filesystem::path& src, const std::filesystem::path& dst, gear g) {
            const size_t min_doubles = 1024;
            mas_file in(src, mas_file::autoMode);
            mas_file out(dst, mas_file::replaceMode);
            for (size_t i = 0; i < in.blockNum(); ++i) {
                mas_block blk = in.readData((int)i);
                if (auto* p = std::get_if<std::unique_ptr<double[]>>(&blk.content);
                    p != nullptr && blk.info.blockSize >= min_doubles) {
                    quantize(p->get(), blk.info.blockSize, g);
                    out.writeData(blk.info.blockName, p->get(), blk.info.blockShape);
                } else {
                    out.writeData(blk.info.blockName, blk);
                }
            }
        }
    }

    // LZMA compressor with lossy quantization pre-pass.
    // Compress() tries to parse the file as a mas archive; 
    // on success every sufficiently large double block is mantissa-quantized before LZMA.
    // Decompression will only restore the quantized archive. 
    // Files that do not parse as mas fall through to plain LZMA. 
    // Options: "q=20,level=6" — the q gear belongs to this stage (default 20 mantissa bits),
    // everything else is forwarded to the LZMA stage.
    struct qlzma_compressor final : compressor {
        void compress(const std::filesystem::path& file_path, const std::string& options = "") const override {
            auto [g, lzma_options] = qlzma_detail::split_options(options);
            if (!is_mas(file_path)) {
                lzma_.compress(file_path, lzma_options);
                return;
            }
            // quantized copy of the archive, LZMA on the copy, publish
            // under the final name, drop the intermediate
            std::filesystem::path tmp = file_path.string() + ".qtmp" + std::to_string(::getpid());
            std::error_code ec;
            try {
                qlzma_detail::quantize_mas(file_path, tmp, g);
            } catch (...) {
                // mas magic matched but the block structure did not parse: treat as a non-mas file
                std::filesystem::remove(tmp, ec);
                lzma_.compress(file_path, lzma_options);
                return;
            }
            try {
                lzma_.compress(tmp, lzma_options);
                std::filesystem::rename(lzma_.compressed_path(tmp), lzma_.compressed_path(file_path), ec);
                if (ec)
                    throw std::runtime_error("qlzma_compressor: failed to publish " + lzma_.compressed_path(file_path).string());
            } catch (...) {
                std::filesystem::remove(tmp, ec);
                std::filesystem::remove(lzma_.compressed_path(tmp), ec);
                throw;
            }
            std::filesystem::remove(tmp, ec);
        }

        void decompress(const std::filesystem::path& file_path) const override {
            lzma_.decompress(file_path);
        }

        std::filesystem::path compressed_path(const std::filesystem::path& file_path) const override {
            return lzma_.compressed_path(file_path);
        }

        std::shared_ptr<const compressor> clone() const override {
            return std::make_shared<qlzma_compressor>(*this);
        }

    private:
        // mas archives start with the mas_file magic string
        static bool is_mas(const std::filesystem::path& file_path) {
            std::FILE* f = std::fopen(file_path.string().c_str(), "rb");
            if (!f) return false;
            char head[6];
            size_t got = std::fread(head, 1, sizeof head, f);
            std::fclose(f);
            static const char magic[6] = {'#', 'M', 'A', 'S', '!', '#'};
            return got == sizeof head && std::string(head, got) == std::string(magic, sizeof magic);
        }

        lzma_compressor lzma_;
    };

}

}

#endif
