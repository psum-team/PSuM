#ifndef PSUM_SERIALIZATION_COMPRESSOR_HPP
#define PSUM_SERIALIZATION_COMPRESSOR_HPP

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

namespace psum {

namespace serialization {

    // runtime-polymorphic compressor
    struct compressor {
        virtual ~compressor() = default;

        // plain file -> compressed file (atomically published)
        // options: compressor-specific strings
        virtual void compress(const std::filesystem::path& file_path, const std::string& options = "") const = 0;

        // compressed file -> plain file
        virtual void decompress(const std::filesystem::path& file_path) const = 0;

        // compressed-file path belonging to a plain file (e.g. "a.mas"-> "a.mas.lzma")
        virtual std::filesystem::path compressed_path(const std::filesystem::path& file_path) const = 0;
        
        // used for generate another instance of derived class
        virtual std::shared_ptr<const compressor> clone() const = 0;
    };

    // handle for a compression job created by the compress entry points.
    //   * default: destroying the handle waits for completion
    //   * job.wait():   synchronize at a chosen point
    //   * job.detach(): fire-and-forget (true async)
    class compress_job {
    public:
        compress_job() = default;
        compress_job(std::thread t, std::shared_ptr<std::exception_ptr> err)
            : thread_(std::move(t)), err_(std::move(err)) {}
        compress_job(compress_job&&) = default;
        compress_job& operator=(compress_job&&) = default;
        ~compress_job() {
            finish();
        }
        void wait() {
            finish();
            if (err_ && *err_)
                std::rethrow_exception(*err_);
        }
        void detach() {
            if (thread_.joinable()) thread_.detach();
            err_.reset();   // detached: reporting is the worker's job
        }
    private:
        void finish() { if (thread_.joinable()) thread_.join(); }
        std::thread thread_;
        std::shared_ptr<std::exception_ptr> err_;
    };

    // rebuild the plain file from its compressed file via the compressor, return the path
    inline const std::filesystem::path& ensure_plain(const std::filesystem::path& file_path, const compressor& method) {
        method.decompress(file_path);
        return file_path;
    }

}

}

#endif
