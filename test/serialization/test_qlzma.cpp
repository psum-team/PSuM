#include "../../src/serialization/qlzma_compressor.hpp"
#include "../../src/serialization/mas.hpp"
#include "../../src/serialization/object_manager.hpp"
#include "../../src/field/device_field.hpp"
#include "../../src/timer.hpp"
#include "../../src/random/rander.hpp"
#include <psum/tag.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>

using namespace std;
using namespace psum::serialization;
using namespace psum::field;
using namespace psum::tag;
using namespace psum::tag::property;

using df3d = device_field<3, var_loc::cellCentered, double, 1>;

// production particle type, following the repo test convention
// (test_objmng_complex / test_objmng_queue)
using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector2d>,
    tag_bind<velocity, Eigen::RowVector2d>,
    tag_bind<weight, double>
>;

int main() {
    sycl::queue q;
    psum::timer tm;

    // 1. production-shape payload: 1M particles in two segments, 1M-cell grid field, all as device_field
    const size_t Np = 1000000;                       // particles total
    const size_t Nseg = Np / 2;
    const int ng = 100;                              // 100^3 cells
    const size_t Nc = (size_t)ng * ng * ng;
    const simple_grid<3> g({0.0, 0.0, 0.0},
                           {0.3 * ng, 0.4 * ng, 1.223 * ng}, {ng, ng, ng});

    psum::random::rander rng;      // deterministic LCG (repo rander)
    auto noise = [&]() { return 2.0 * rng() - 1.0; };   // uniform in [-1, 1)
    vector<Particle> seg1(Nseg), seg2(Nseg);
    for (size_t i = 0; i < Nseg; i++) {
        for (int half = 0; half < 2; half++) {
            Particle& p = half == 0 ? seg1[i] : seg2[i];
            const double x = double(i + half * Nseg);
            get<position>(p) = Eigen::RowVector2d(
                1e-6 * x + 0.5 * sin(x * 1e-4) + noise() * 0.2,
                2e-6 * x + 0.3 * cos(x * 3.7e-3) + noise() * 0.3);
            get<velocity>(p) = Eigen::RowVector2d(
                0.4 * sin(x * 8e-4) + noise() * 1000,
                0.2 * cos(x * 5e-4) + noise() * 2333);
            get<weight>(p) = half == 0 ? 1e7 : 1e9;
        }
    }
    vector<double> field(Nc);
    for (size_t i = 0; i < Nc; i++)
        field[i] = 0.5 * sin(i * 1e-5) * 50.0 + 150.0 + noise() * 1e-8;

    // production write path: plain mas via object_manager::save, then
    // mas_file::compress with options
    auto write_payload = [&](const filesystem::path& path) {
        object_manager mng("", q);
        mng.obj<device_array<Particle>>("particles1") = device_array<Particle>(q, seg1);
        mng.obj<device_array<Particle>>("particles2") = device_array<Particle>(q, seg2);
        mng.obj<df3d>("field") = df3d(device_array<double>(q, field), g);
        mng.save_compressed(path, qlzma_compressor{}).wait();   // defaults: q20 level=4 dict=4m
        mng.save(path);
    };

    // element-wise check of a double array against the original
    auto array_stats = [&](const char* what, const vector<double>& orig,
                           const vector<double>& got, int bits) {
        if (orig.size() != got.size()) {
            cout << "  " << what << ": size " << orig.size() << " vs "
                 << got.size() << " FAIL" << endl;
            return false;
        }
        const double bound = pow(2.0, -bits);
        double maxrel = 0;
        size_t zero_orig = 0, zero_bad = 0;
        for (size_t i = 0; i < orig.size(); i++) {
            if (orig[i] == 0.0) {
                ++zero_orig;
                if (got[i] != 0.0) ++zero_bad;
            } else {
                maxrel = max(maxrel, abs(got[i] - orig[i]) / abs(orig[i]));
            }
        }
        const bool ok = maxrel < bound && zero_bad == 0;
        cout << "  " << what << " (" << orig.size() << " doubles): max rel err "
             << maxrel << " (bound " << bound << "), zeros " << zero_orig
             << (zero_bad ? " CORRUPT" : " exact")
             << (ok ? "" : "  FAIL") << endl;
        return ok;
    };

    // direct particle-vector comparison: position / velocity relative
    // error per component, weight must survive exactly (the segment
    // constants are chosen to sit on the quantization grid)
    auto particle_stats = [&](const char* what, const vector<Particle>& orig,
                              const vector<Particle>& got, int bits) {
        if (orig.size() != got.size()) {
            cout << "  " << what << ": size " << orig.size() << " vs "
                 << got.size() << " FAIL" << endl;
            return false;
        }
        const double bound = pow(2.0, -bits);
        double maxrel = 0;
        size_t zeros = 0, zero_bad = 0;
        bool weight_exact = true;
        for (size_t i = 0; i < orig.size(); i++) {
            const Eigen::RowVector2d op = get<position>(orig[i]);
            const Eigen::RowVector2d gp = get<position>(got[i]);
            const Eigen::RowVector2d ov = get<velocity>(orig[i]);
            const Eigen::RowVector2d gv = get<velocity>(got[i]);
            for (int c = 0; c < 2; c++) {
                const double ps[2] = {op[c], ov[c]};
                const double gs[2] = {gp[c], gv[c]};
                for (int k = 0; k < 2; k++) {
                    if (ps[k] == 0.0) {
                        ++zeros;
                        if (gs[k] != 0.0) ++zero_bad;
                    } else {
                        maxrel = max(maxrel, abs(gs[k] - ps[k]) / abs(ps[k]));
                    }
                }
            }
            weight_exact = weight_exact && get<weight>(orig[i]) == get<weight>(got[i]);
        }
        const bool ok = maxrel < bound && zero_bad == 0 && weight_exact;
        cout << "  " << what << " (" << orig.size() << " particles): max rel err "
             << maxrel << " (bound " << bound << "), zeros " << zeros
             << (zero_bad ? " CORRUPT" : " exact") << ", weight exact "
             << (weight_exact ? "YES" : "FAIL")
             << (ok ? "" : "  FAIL") << endl;
        return ok;
    };

    // 2. single production run at the default gear with the Pareto
    //    dictionary: write, compress, drop the plain file, reload
    //    through the compressor constructor
    const filesystem::path big = "test_qlz_scale.mas";
    tm.tic("write+compress");
    write_payload(big);                       // save_compressed inside (default gear)
    const auto raw = filesystem::file_size(big);
    filesystem::remove(big);                  // plain re-saved inside: drop it,
                                              // force reload through the .lzma
    const double wc = tm.toc("write+compress");
    tm.tic("reload");
    object_manager back(big, qlzma_compressor{}, q);
    vector<Particle> p1 = back.obj<device_array<Particle>>("particles1").to_host();
    vector<Particle> p2 = back.obj<device_array<Particle>>("particles2").to_host();
    const auto rf = back.obj<df3d>("field").getContent().to_host();
    const double rl = tm.toc("reload");
    const auto packed = filesystem::file_size(qlzma_compressor{}.compressed_path(big));
    cout << "qlzma defaults (q20): raw " << raw << " B (" << (raw >> 20)
         << " MiB), ratio " << double(packed) / raw
         << ", write+compress " << wc << "s, reload " << rl << "s" << endl;
    bool ok = particle_stats("particles1", seg1, p1, 20);
    ok = particle_stats("particles2", seg2, p2, 20) && ok;
    ok = array_stats("field", field, rf, 20) && ok;
    if (!ok) return 1;
    filesystem::remove(qlzma_compressor{}.compressed_path(big));
    if (!ok) return 1;

    // 3. small and non-double blocks are never quantized
    {
        const filesystem::path s = "test_qlz_small.mas";
        vector<double> big_block(4096), small_block(16);
        vector<int32_t> idx(4096);
        for (size_t i = 0; i < 4096; i++) { big_block[i] = noise(); idx[i] = int32_t(i); }
        for (size_t i = 0; i < 16; i++) small_block[i] = 0.1 * double(i);
        {
            mas_file fp(s.string(), mas_file::replaceMode);
            fp.writeData("big", big_block.data(), {4096});
            fp.writeData("small", small_block.data(), {16});
            fp.writeData("idx", idx.data(), {4096});
        }
        qlzma_compressor qc;
        qc.compress(s, "level=1");
        qc.decompress(s);
        mas_file fp(s.string(), mas_file::autoMode);
        const auto sb = fp.readData("small");
        const auto ib = fp.readData("idx");
        const bool ok = memcmp(get<unique_ptr<double[]>>(sb.content).get(),
                               small_block.data(), 16 * 8) == 0
                        && memcmp(get<unique_ptr<int32_t[]>>(ib.content).get(),
                                  idx.data(), 4096 * 4) == 0;
        cout << "small/int blocks untouched: " << (ok ? "exact" : "FAIL") << endl;
        filesystem::remove(s);
        filesystem::remove(qc.compressed_path(s));
        if (!ok) return 1;
    }

    // 4. non-mas input falls through to plain lzma: byte-identical
    //    roundtrip and the same output the plain lzma compressor
    //    produces for the same options
    {
        const filesystem::path txt = "test_qlz_plain.txt";
        vector<char> text(200000);
        for (size_t i = 0; i < text.size(); i++)
            text[i] = char('a' + (i * 7 + i / 13) % 26);
        {   FILE* f = fopen(txt.string().c_str(), "wb");
            fwrite(text.data(), 1, text.size(), f);
            fclose(f); }
        qlzma_compressor qc;
        qc.compress(txt, "level=1");
        const auto qsz = filesystem::file_size(qc.compressed_path(txt));
        lzma_compressor pc;
        pc.compress(txt, "level=1");
        const auto psz = filesystem::file_size(pc.compressed_path(txt));
        qc.decompress(txt);
        vector<char> back(text.size());
        {   FILE* f = fopen(txt.string().c_str(), "rb");
            fread(back.data(), 1, back.size(), f);
            fclose(f); }
        const bool same = memcmp(text.data(), back.data(), text.size()) == 0;
        cout << "non-mas fallback: roundtrip " << (same ? "identical" : "FAIL")
             << ", ratio " << double(qsz) / text.size()
             << ", == plain lzma output " << (qsz == psz ? "YES" : "NO") << endl;
        filesystem::remove(txt);
        filesystem::remove(qc.compressed_path(txt));
        if (!same || qsz != psz) return 1;
    }

    // 5. invalid gear is rejected
    {
        bool threw = false;
        try { qlzma_compressor{}.compress("test_qlz_small.mas", "q=31"); }
        catch (const exception&) { threw = true; }
        cout << "q=31 rejected: " << (threw ? "YES" : "NO") << endl;
        if (!threw) return 1;
    }

    cout << "PASS: qlzma production-scale quantized compression" << endl;
    return 0;
}
