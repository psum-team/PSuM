// device_batch_field vs plain device_field behavior consistency
#include <psum/batch_pic.hpp>
#include <iostream>
#include <vector>

using namespace psum;
using namespace psum::field;
using namespace psum::batch_pic;


using NF = node_field2D<double, 1>;

static bool same_content(const NF& a, const NF& b) {
    auto ha = a.getContent().to_host();
    auto hb = b.getContent().to_host();
    return ha == hb;
}

int main() {
    sycl::queue q{sycl::default_selector_v};

    int C = 3;
    simple_grid<2> grid({ 0.0, 0.0 }, { 1.0, 1.0 }, { 80, 60 });
    std::vector<NF> ref;
    for (size_t c = 0; c < C; c++)
        ref.emplace_back(q, grid);
    batch_node_field2D<double, 1> batch(q, grid, C);

    bool ok = true;
    auto check = [&](bool cond, const std::string& what) {
        if (!cond) {
            ok = false;
            std::cout << "❌ " << what << std::endl;
        } else {
            std::cout << "✅ " << what << std::endl;
        }
    };

    // --- setZero consistency
    batch.setZero();
    for (auto& f : ref) f.setZero();
    {
        bool all = true;
        for (size_t c = 0; c < C; c++)
            all = all && same_content(batch.at(c), ref[c]);
        check(all, "setZero: batch == per-case plain fields");
    }

    // --- identical fill via for_each, content equality
    for (size_t c = 0; c < C; c++) {
        auto fill = [c, grid](size_t i, typename NF::Value& val, const Eigen::RowVector2d& p) {
            val = c * 1000.0 + cos(p.x() * 10) * sin(p.y() * 3.14159 * 2) * (1 + i / 1000.0);
        };
        ref[c].for_each([&](sycl::handler&) { return fill; });
        batch.at(c).for_each([&](sycl::handler&) { return fill; });
    }
    {
        bool all = true;
        for (size_t c = 0; c < C; c++)
            all = all && same_content(batch.at(c), ref[c]);
        check(all, "for_each fill: batch == per-case plain fields");
    }

    // --- kernel interp via batch acc table vs per-field acc, per case
    {
        bool all = true;
        for (size_t c = 0; c < C; c++) {
            NF via_batch(q, grid), via_plain(q, grid);
            via_batch.for_each([&](sycl::handler& h) {
                auto bacc = batch.get_access(h);
                return [=](size_t, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    val = batch_pic::interp(p, c, bacc);
                };
            });
            via_plain.for_each([&](sycl::handler& h) {
                auto acc = ref[c].get_access(h);
                return [=](size_t, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    val = field::interp(p, acc);
                };
            });
            all = all && same_content(via_batch, via_plain);
        }
        check(all, "kernel interp(pos, case_id, batch_acc) == field::interp(pos, plain acc)");
    }

    // --- kernel interp_diff consistency
    {
        bool all = true;
        for (size_t c = 0; c < C; c++) {
            NF via_batch(q, grid), via_plain(q, grid);
            // probe interior nodes only: linear_interp_diff at exact-boundary
            // positions reads corner+1 indices that leave the node array (an
            // upstream grid-helper property, identical for both paths)
            via_batch.for_each([&](sycl::handler& h) {
                auto bacc = batch.get_access(h);
                return [=](size_t, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    if (p.x() < 1.0 && p.y() < 1.0)
                        val = batch_pic::interp_diff(p, c, bacc)[0];
                    else
                        val = 0.0;
                };
            });
            via_plain.for_each([&](sycl::handler& h) {
                auto acc = ref[c].get_access(h);
                return [=](size_t, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    if (p.x() < 1.0 && p.y() < 1.0)
                        val = field::interp_diff(p, acc)[0];
                    else
                        val = 0.0;
                };
            });
            auto hb = via_batch.getContent().to_host();
            auto hs = via_plain.getContent().to_host();
            size_t diffs = 0;
            for (size_t i = 0; i < hb.size(); i++) {
                double d = std::abs(hb[i] - hs[i]);
                if (d > 0.0) diffs++;
            }
            all = all && diffs == 0;
        }
        check(all, "kernel interp_diff(pos, case_id, batch_acc) == field::interp_diff(pos, plain acc)");
    }

    // --- kernel add_back consistency: P tagged particles land on their case
    {
        const size_t P = 40;
        NF driver_b(q, grid), driver_s(q, grid);
        for (size_t c = 0; c < C; c++) {
            driver_b.for_each([&](sycl::handler& h) {
                auto bacc = batch.get_access(h);
                return [=](size_t i, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    double w = 0.5 + 0.01 * i;
                    batch_pic::add_back(p, w, c, bacc);
                };
            });
            driver_s.for_each([&](sycl::handler& h) {
                auto acc = ref[c].get_access(h);
                return [=](size_t i, typename NF::Value& val, const Eigen::RowVector2d& p) {
                    double w = 0.5 + 0.01 * i;
                    field::add_back(p, w, acc);
                };
            });
        }
        bool all = true;
        for (size_t c = 0; c < C; c++) all = all && same_content(batch.at(c), ref[c]);
        check(all, "kernel add_back(pos, w, case_id, batch_acc) == field::add_back(pos, w, plain acc)");
    }

    // --- setConstant consistency
    batch.setConstant(3.25);
    for (auto& f : ref) f.setConstant(3.25);
    {
        bool all = true;
        for (size_t c = 0; c < C; c++) all = all && same_content(batch.at(c), ref[c]);
        check(all, "setConstant: batch == per-case plain fields");
    }

    // --- copy consistency
    {
        std::vector<std::vector<double>> hb(C);
        for (size_t c = 0; c < C; c++)
            for (size_t i = 0; i < batch.at(c).size(); i++)
                hb[c].push_back(c * 7.0 + i * 0.25);
        batch.copy(hb);
        for (size_t c = 0; c < C; c++) ref[c].copy(hb[c]);
        bool all = true;
        for (size_t c = 0; c < C; c++) all = all && same_content(batch.at(c), ref[c]);
        check(all, "copy(host batches): batch == per-case plain fields");
    }

    // --- batch-level getContent vs per-field getContent
    {
        auto batch_host = batch.getContent().to_host();
        bool all = batch_host.size() == C;
        for (size_t c = 0; c < C && all; c++)
            all = batch_host[c] == ref[c].getContent().to_host();
        check(all, "getContent().to_host(): batch == per-case plain fields");
    }

    std::cout << (ok ? "ALL PASS" : "FAILED") << std::endl;
    return ok ? 0 : 1;
}
