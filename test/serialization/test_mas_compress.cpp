#include "../../src/serialization/lzma_compressor.hpp"
#include "../../src/field/device_field.hpp"
#include "../../src/timer.hpp"
#include "../../src/serialization/mas.hpp"
#include "../../src/serialization/object_manager.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <filesystem>

using namespace std;
using namespace psum::serialization;
using namespace psum::field;

int main() {
    const std::filesystem::path mas_path = "test_mc.mas";
    const std::filesystem::path lzma_path = lzma_compressor{}.compressed_path(mas_path);

    // 1. write a plain archive
    const size_t n = 524288;
    vector<double> phi(n), rho(n);
    for (size_t i = 0; i < n; i++) {
        phi[i] = sin(i * 0.01) * 100.0 + 150.0;
        rho[i] = cos(i * 0.003) * 1e-6;
    }
    {
        mas_file fp(mas_path.string(), mas_file::replaceMode);
        fp.writeData("phi", phi.data(), {n, 1});
        fp.writeData("rho", rho.data(), {n, 1});
    }

    // 2. mas_file::compress, job dropped -> synchronous completion
    {
        mas_file fp(mas_path.string(), mas_file::quickMode);
        fp.compress(lzma_compressor{});          // return value dropped: waits
    }
    if (!std::filesystem::exists(lzma_path)) {
        cout << "FAIL: compressed file not produced" << endl;
        return 1;
    }
    cout << "mas_file::compress produced compressed file" << endl;

    // 3. construct from the compressed archive; blocks survive the roundtrip
    {
        mas_file fp(mas_path, lzma_compressor{});
        auto blk = fp.readData("phi");
        const double* p = std::get<std::unique_ptr<double[]>>(blk.content).get();
        const bool same = blk.info.blockSize == n && std::memcmp(p, phi.data(), n * 8) == 0;
        cout << "mas_file(path, compressor) roundtrip: " << (same ? "identical" : "FAIL") << endl;
        if (!same) return 1;
    }

    // 4. object_manager::save_compressed with wait(): only the compressed
    //    file remains; reload through the compressor constructor
    {
        const std::filesystem::path om_path = "test_om.mas";
        vector<double> field(64);
        for (int i = 0; i < 64; i++) field[i] = i * 0.5;
        {
            object_manager mng("");
            mng.obj<std::vector<double>>("field") = field;
            auto job = mng.save_compressed(om_path, lzma_compressor{});
            job.wait();
        }
        const bool only_compressed = !std::filesystem::exists(om_path)
                                     && std::filesystem::exists(lzma_compressor{}.compressed_path(om_path));
        cout << "save_compressed leaves only compressed file: " << (only_compressed ? "yes" : "FAIL") << endl;
        if (!only_compressed)
            return 1;
        {
            object_manager mng(om_path, lzma_compressor{});
            auto& v = mng.obj<std::vector<double>>("field");
            const bool same = v == field;
            cout << "object_manager(path, compressor) reload: " << (same ? "identical" : "FAIL") << endl;
            if (!same) return 1;
        }
        std::filesystem::remove(lzma_compressor{}.compressed_path(om_path));
    }

    // 5. detach(): fire-and-forget completion in the background
    {
        const std::filesystem::path d_path = "test_d.mas";
        object_manager mng("");
        mng.obj<std::vector<double>>("field") = vector<double>(64, 7.5);
        mng.save_compressed(d_path, lzma_compressor{}).detach();
        const auto done = [&] {
            return std::filesystem::exists(lzma_compressor{}.compressed_path(d_path))
                   && !std::filesystem::exists(d_path);
        };
        for (int spin = 0; spin < 600 && !done(); spin++)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        cout << "detach background completion: " << (done() ? "yes" : "FAIL") << endl;
        if (!done()) return 1;
        std::filesystem::remove(lzma_compressor{}.compressed_path(d_path));
    }

    // 6. error path: compressed file missing -> constructor throws
    {
        bool threw = false;
        try { mas_file missing("no_such_archive.mas", lzma_compressor{}); }
        catch (const std::exception&) { threw = true; }
        cout << "missing compressed file: " << (threw ? "throws OK" : "FAIL") << endl;
        if (!threw) return 1;
    }

    std::filesystem::remove(mas_path);
    std::filesystem::remove(lzma_path);

    // 7. production-scale payload: 1M-cell grid fields as real device_field
    {
        sycl::queue q;
        const int n = 100;                             // n^3 cells
        const size_t Nc = (size_t)n * n * n;
        const simple_grid<3> g({0.0, 0.0, 0.0}, {0.3 * n, 0.4 * n, 1.223 * n},
                               {n, n, n});
        using df3 = device_field<3, var_loc::cellCentered, float, 1>;
        vector<float> phi(Nc), rho(Nc);
        for (size_t i = 0; i < Nc; i++) {
            phi[i] = (float)(sin(i * 1e-5) * 50.0 + 150.0);
            rho[i] = (float)(cos(i * 3e-6) * 1e-6);
        }
        const std::filesystem::path big = "test_scale.mas";
        psum::timer tm;
        tm.tic("write+compress");
        {
            object_manager mng("", q);
            mng.obj<df3>("phi") = df3(device_array<float>(q, phi), g);
            mng.obj<df3>("rho") = df3(device_array<float>(q, rho), g);
            auto job = mng.save_compressed(big, lzma_compressor{});
            const auto raw = std::filesystem::file_size(big);
            job.wait();
            const auto packed = std::filesystem::file_size(lzma_compressor{}.compressed_path(big));
            const double wc = tm.toc("write+compress");
            tm.tic("reload");
            // after the job only the compressed file remains; the reload
            // constructor rebuilds the plain file by design
            const bool only_compressed = !std::filesystem::exists(big);
            object_manager back(big, lzma_compressor{}, q);
            const auto phi_host = back.obj<df3>("phi").getContent().to_host();
            const auto rho_host = back.obj<df3>("rho").getContent().to_host();
            const double rl = tm.toc("reload");
            const bool same = only_compressed
                              && phi_host.size() == Nc && rho_host.size() == Nc
                              && std::memcmp(phi_host.data(), phi.data(), Nc * sizeof(float)) == 0
                              && std::memcmp(rho_host.data(), rho.data(), Nc * sizeof(float)) == 0;
            cout << "device_field grid (n=" << n << ", f32): raw "
                 << (raw >> 10) << " KiB -> " << (packed >> 10) << " KiB"
                 << " (ratio " << (double)packed / raw << ")" << endl;
            cout << "  write+compress " << wc << "s"
                 << "  reload " << rl << "s"
                 << "  content " << (same ? "identical" : "FAIL") << endl;
            if (!same) return 1;
        }
        std::filesystem::remove(lzma_compressor{}.compressed_path(big));
    }

    cout << "PASS: mas <-> compressed integration" << endl;
    return 0;
}
