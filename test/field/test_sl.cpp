#include <algorithm>
#include <psum/field.hpp>
#include <psum/particle_container.hpp>
#include <iostream>

using namespace psum;
using namespace psum::serialization;

static int failures = 0;

static void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "failed: " << what << std::endl;
        failures++;
    }
}

template <typename F>
static bool throws_runtime(F&& f) {
    try { f(); } catch (const std::runtime_error&) { return true; }
    return false;
}

int main() {

    sycl::queue q;

    // ---- host_field ----
    {
        field::grid1D gridA({0.0}, {1.0}, {678});
        field::host_cell_field1D<double> f(gridA);
        f.for_each([](size_t i, double& v) { v = 0.5 * i; });
        mas_file fp("test_field_sl.mas", mas_file::replaceMode);
        save(fp, "f1", f);

        // no-arg construction: grid is empty and behavior is restricted
        field::host_cell_field1D<double> f0;
        check(!f0.has_grid(), "host no-arg has_grid==false");
        check(throws_runtime([&] { (void)f0.size(); }), "host no-arg size() throws");
        check(throws_runtime([&] { (void)f0.getGrid(); }), "host no-arg getGrid() throws");
        check(throws_runtime([&] { f0.setZero(); }), "host no-arg setZero() throws");
        check(throws_runtime([&] { f0.setConstant(1.0); }), "host no-arg setConstant() throws");
        check(throws_runtime([&] { f0.copy(std::vector<double>{}); }), "host no-arg copy() throws");
        check(throws_runtime([&] { (void)f0.data(); }), "host no-arg data() throws");
        check(throws_runtime([&] { (void)f0.getContent(); }), "host no-arg getContent() throws");

        // load sets the grid from the file
        load(fp, "f1", f0);
        check(f0.has_grid(), "host load sets grid");
        check(f0.size() == gridA.contentSize(field::var_loc::cellCentered), "host load size matches");
        bool eq = true;
        for (size_t i = 0; i < f0.size(); i++) eq = eq && (f0(i) == 0.5 * i);
        check(eq, "host load values match");

        // load adopts the file's grid even if the target was built with another grid
        field::grid1D gridB({0.0}, {2.0}, {8});
        field::host_cell_field1D<double> f2(gridB);
        load(fp, "f1", f2);
        check(f2.getGrid().get_cell_num() == gridA.get_cell_num(), "host load adopts file grid");
        eq = true;
        for (size_t i = 0; i < f2.size(); i++) eq = eq && (f2(i) == 0.5 * i);
        check(eq, "host grid-adopt values match");
    }

    // ---- host_field: QS>1 ----
    {
        field::grid3D grid({0.0, 0.0, 0.0}, {1.0, 1.1, 1.05}, {41, 104, 49});
        field::host_cell_field3D<double, 3> f(grid);
        f.for_each([](size_t i, Eigen::RowVector<double, 3>& v) { v = Eigen::RowVector<double, 3>(0.25 * i, 0.5 * i, 1.0 * i); });
        mas_file fp("test_field_sl.mas", mas_file::replaceMode);
        save(fp, "f2", f);
        field::host_cell_field3D<double, 3> g;
        load(fp, "f2", g);
        bool eq = true;
        for (size_t i = 0; i < g.size(); i++) eq = eq && (g(i) == Eigen::RowVector<double, 3>(0.25 * i, 0.5 * i, 1.0 * i));
        check(eq, "host 3D QS=3 round trip");
    }

    // ---- device_field ----
    {
        field::grid2D gridA({0.0, 0.0}, {1.0, 100.0}, {456, 256});
        field::cell_field2D<double, 3> df(q, gridA);
        using DValue = Eigen::RowVector<double, 3>;
        df.for_each([](sycl::handler& h) {
            return [](size_t i, DValue& v) { v = DValue(0.25 * i, 0.5 * i, 1.0 * i); };
        });
        mas_file fp("test_field_sl.mas", mas_file::replaceMode);
        save(fp, "d1", df);

        // single-arg (queue) construction: grid is empty and behavior is restricted
        field::cell_field2D<double, 3> d0(q);
        check(!d0.has_grid(), "device single-arg has_grid==false");
        check(throws_runtime([&] { (void)d0.size(); }), "device single-arg size() throws");
        check(throws_runtime([&] { (void)d0.getGrid(); }), "device single-arg getGrid() throws");
        check(throws_runtime([&] { d0.setZero(); }), "device single-arg setZero() throws");
        check(throws_runtime([&] { d0.setConstant(DValue::Zero()); }), "device single-arg setConstant() throws");
        check(throws_runtime([&] { d0.copy(std::vector<DValue>{}); }), "device single-arg copy() throws");
        check(throws_runtime([&] { (void)d0.data(); }), "device single-arg data() throws");
        check(throws_runtime([&] { (void)d0.getContent(); }), "device single-arg getContent() throws");

        // load rebuilds content and sets the grid
        load(fp, "d1", d0);
        check(d0.has_grid(), "device load sets grid");
        check(d0.size() == gridA.contentSize(field::var_loc::cellCentered), "device load size matches");
        auto host_copy = d0.getContent().to_host();
        bool eq = true;
        for (size_t i = 0; i < host_copy.size(); i++) eq = eq && (host_copy[i] == DValue(0.25 * i, 0.5 * i, 1.0 * i));
        check(eq, "device load values match");

        // load adopts the file's grid even if the target was built with another grid
        field::grid2D gridB({0.0, 0.0}, {2.0, 2.0}, {2, 2});
        field::cell_field2D<double, 3> d2(q, gridB);
        load(fp, "d1", d2);
        check(d2.getGrid().get_cell_num() == gridA.get_cell_num(), "device load adopts file grid");
        host_copy = d2.getContent().to_host();
        eq = true;
        for (size_t i = 0; i < host_copy.size(); i++) eq = eq && (host_copy[i] == DValue(0.25 * i, 0.5 * i, 1.0 * i));
        check(eq, "device grid-adopt values match");
    }

    // ---- device_multi_patch_field ----
    {
        using MPF = field::multi_patch::cell_mp_field2D<double, 3>;
        using MValue = Eigen::RowVector<double, 3>;
        using MPG = field::multi_patch::multi_patch_grid<2>;
        MPG mpg(q, {0.0, 0.0}, {1.0, 1.2}, {2, 3}, {4, 4, 5, 5, 5, 6});
        MPF mf(std::move(mpg));
        mf.for_each([](sycl::handler& h) {
            return [](size_t i, MValue& v) { v = MValue(0.5 * i, 2.0 * i, 1.0); };
        });
        mas_file fp("test_field_sl.mas", mas_file::replaceMode);
        save(fp, "mp1", mf);

        // load rebuilds grid/content/positions into a single-queue empty field
        MPF mf0(q);
        load(fp, "mp1", mf0);
        check(mf0.has_grid(), "mp load sets grid");
        check(mf0.size() == mf.size(), "mp load size matches");
        auto mp_host = mf0.getContent().to_host();
        bool eq = mp_host.size() == mf.size();
        for (size_t i = 0; eq && i < mp_host.size(); i++)
            eq = eq && (mp_host[i] == MValue(0.5 * i, 2.0 * i, 1.0));
        check(eq, "mp load values match");

        // grid consistency across S/L
        check(mf0.getGrid().get_patch_num() == mf.getGrid().get_patch_num(), "mp grid patch_num matches");
        check(mf0.getGrid().get_lower_bounds() == mf.getGrid().get_lower_bounds(), "mp grid lower matches");
        check(mf0.getGrid().get_upper_bounds() == mf.getGrid().get_upper_bounds(), "mp grid upper matches");
        check(mf0.getGrid().get_resolution_exponents() == mf.getGrid().get_resolution_exponents(), "mp grid resolution matches");

        // positions consistency: collect on both sides via element-wise for_each
        using Pos2 = Eigen::RowVector<double, 2>;
        psum::particle_container::device_vector<Pos2> pos_src(q), pos_dst(q);
        mf.for_each([&](sycl::handler& h) {
            auto out = pos_src.get_access(h);
            return [=](size_t, MValue&, const Pos2& p) { out.push_back(p); };
        });
        mf0.for_each([&](sycl::handler& h) {
            auto out = pos_dst.get_access(h);
            return [=](size_t, MValue&, const Pos2& p) { out.push_back(p); };
        });
        auto ps = pos_src.to_host(), pd = pos_dst.to_host();
        std::sort(ps.begin(), ps.end(), [](const Pos2& a, const Pos2& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
        std::sort(pd.begin(), pd.end(), [](const Pos2& a, const Pos2& b) { return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end()); });
        bool pos_eq = (ps == pd);
        check(pos_eq, "mp positions rebuilt consistently");
    }

    if (failures == 0) {
        std::cout << "test_sl: all checks passed" << std::endl;
        return 0;
    }
    std::cout << "test_sl: " << failures << " check(s) failed" << std::endl;
    return 1;
}
