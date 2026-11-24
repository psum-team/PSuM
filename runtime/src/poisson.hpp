#ifndef PSUM_RUNTIME_POISSON_HPP
#define PSUM_RUNTIME_POISSON_HPP

// Poisson solvers behind the C interface
// solver_kind follows psum_runtime_api.h (a 4-value enum: 1d / 2d cartesian / 2d axisymmetric / 3d).

#include <psum/psum.hpp>
#include "field_solver/Poisson_solver_1d.hpp"
#include "field_solver/Poisson_solver_2d.hpp"
#include "field_solver/Poisson_solver_3d.hpp"
#include "field_solver/boundary_creator/Dirichlet.hpp"
#include "field_solver/boundary_creator/Neumann.hpp"
#include "field_solver/boundary_creator/Robin.hpp"

namespace psum_runtime {

namespace poisson {

    namespace fs = psum::field_solver;
    namespace fsb = psum::field_solver::boundary_creator;

    using grid1 = psum::field::simple_grid<1>;
    using grid2 = psum::field::simple_grid<2>;
    using grid3 = psum::field::simple_grid<3>;
    template <int Dim>
    using field1 = psum::field::device_field<Dim, psum::field::var_loc::nodeCentered, double, 1>;
    template <int Dim>
    using host_field1 = psum::field::host_field<Dim, psum::field::var_loc::nodeCentered, double, 1>;

    inline constexpr const char* edge_names[3][6] = {
        {"L", "R"},
        {"N", "S", "E", "W"},
        {"Xpos", "Xneg", "Ypos", "Yneg", "Zpos", "Zneg"}
    };
    // outward unit normals, same order as the edge names
    inline constexpr double edge_normals[3][6][3] = {
        {{-1, 0, 0}, {1, 0, 0}},
        {{0, 1, 0}, {0, -1, 0}, {1, 0, 0}, {-1, 0, 0}},
        {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}}
    };

    // one solver per dimension slot, initialized at most once.
    struct slots {
        std::optional<fs::Poisson_solver_1d> s1;
        std::optional<fs::Poisson_solver_2d> s2;
        std::optional<fs::Poisson_solver_3d> s3;
        // dim d solved through the built-in native path because the
        // requested backend is not registered in this build
        bool native_forced[4] = {};
    };

    inline std::string value_key(int dim, int edge) {
        return "poisson" + std::to_string(dim) + "d_bc_" + edge_names[dim - 1][edge];
    }

    template <typename Mng>
    double* value_ptr(Mng& mng, const std::string& key) {
        return &mng.template obj<double>(key);
    }

    // bc = value_ref: the boundary reads the managed double at every
    // evaluation, surviving the solver's copy of the boundary lists
    template <int Dim>
    std::function<double(const std::array<double, Dim>&)> value_ref(const double* p) {
        return [p](const std::array<double, Dim>&) { return *p; };
    }

    inline fs::mixed_boundary_1d robin_1(const grid1& g,
                                            psum::field::boundary_direction_1d dir,
                                            double c0, double nx) {
        return fsb::Robin_point(g, dir, [c0, nx](double x) {
            double r = x - c0;
            return std::make_pair(1.0, r * r / (r * nx));
        });
    }
    inline fs::mixed_boundary_2d robin_2(const grid2& g,
                                            psum::field::boundary_direction_2d dir,
                                            const double* c, const double* n) {
        return fsb::Robin_Line(g, dir, [c0 = c[0], c1 = c[1], nx = n[0], ny = n[1]](
                                    double x, double y) {
            double rx = x - c0, ry = y - c1;
            return std::make_pair(1.0, (rx * rx + ry * ry) / (rx * nx + ry * ny));
        });
    }
    inline fs::mixed_boundary_3d robin_3(const grid3& g,
                                            psum::field::boundary_direction_3d dir,
                                            const double* c, const double* n) {
        return fsb::Robin_plane(g, dir, [c0 = c[0], c1 = c[1], c2 = c[2],
                                            nx = n[0], ny = n[1], nz = n[2]](
                                    double x, double y, double z) {
            double rx = x - c0, ry = y - c1, rz = z - c2;
            return std::make_pair(1.0, (rx * rx + ry * ry + rz * rz) /
                                            (rx * nx + ry * ny + rz * nz));
        });
    }

    // set_related_boundaries replaces, so the list is assembled once
    // from the per-edge objects
    template <typename ListT, typename B>
    ListT list_from(const std::vector<B>& v) {
        auto make = [&]<size_t... I>(std::index_sequence<I...>) {
            return ListT{v[I]...};
        };
        switch (v.size()) {
        case 0: return make(std::make_index_sequence<0>{});
        case 1: return make(std::make_index_sequence<1>{});
        case 2: return make(std::make_index_sequence<2>{});
        case 3: return make(std::make_index_sequence<3>{});
        case 4: return make(std::make_index_sequence<4>{});
        case 5: return make(std::make_index_sequence<5>{});
        case 6: return make(std::make_index_sequence<6>{});
        default: return ListT{};
        }
    }

    template <typename Mng>
    int init(Mng& mng, slots& s, int solver_kind, const int* bc_kinds,
                const double* robin_center, const char* grid_name,
                const char* backend_name, const char* options,
                double epsilon = 8.854e-12) {
        const int dim = (solver_kind == 0) ? 1 : (solver_kind == 3) ? 3 : 2;
        const bool cylindrical = (solver_kind == 2);
        for (int b = 0; b < 2 * dim; b++)
            if (bc_kinds[b] < 0 || bc_kinds[b] > 3) return -1;
        bool has_robin = false;
        for (int b = 0; b < 2 * dim; b++) has_robin |= (bc_kinds[b] == 2);
        if (has_robin && !robin_center) return -1;
        std::string backend = backend_name ? backend_name : "";
        std::string opts = options ? options : "";
        if (!backend.empty() && backend != "native"
            && get_solver_funcs(backend.c_str()).init == nullptr) {
            // backend unavailable in this build (e.g. gpu-less): degrade
            // to the built-in native solver; solve_d then round-trips
            // through the host for that dimension
            std::cerr << "poisson: backend '" << backend
                      << "' not registered, degrading to native" << std::endl;
            backend = "native";
            s.native_forced[dim] = true;
        }
        if ((dim == 1 && s.s1) || (dim == 2 && s.s2) || (dim == 3 && s.s3))
            return -2;   // one init per slot, no re-initialization
        try {
        switch (dim) {
        case 1: {
            if (!mng.template contains<grid1>(grid_name)) return -1;
            auto& g = mng.template obj<grid1>(grid_name);
            std::vector<fs::mixed_boundary_1d> mixed;
            std::vector<fs::fixed_boundary_1d> fixed;
            for (int b = 0; b < 2; b++) {
                auto dir = psum::field::boundary_direction_1d(b);
                std::string key = value_key(1, b);
                if (bc_kinds[b] == 1) {
                    auto bc = fsb::Dirichlet_point(g, dir);
                    bc = value_ref<1>(value_ptr(mng, key));
                    fixed.push_back(bc);
                } else if (bc_kinds[b] == 2) {
                    auto bc = robin_1(g, dir, robin_center[0],
                                        edge_normals[0][b][0]);
                    bc = value_ref<1>(value_ptr(mng, key));
                    mixed.push_back(bc);
                } else if (bc_kinds[b] == 3) {
                    auto bc = fsb::Neumann_point(g, dir);
                    bc = value_ref<1>(value_ptr(mng, key));
                    mixed.push_back(bc);
                }
            }
            s.s1.emplace();
            s.s1->init(backend, g, list_from<fs::mixed_boundary_1d_list>(mixed),
                        list_from<fs::fixed_boundary_1d_list>(fixed), epsilon);
            return 0;
        }
        case 2: {
            if (!mng.template contains<grid2>(grid_name)) return -1;
            auto& g = mng.template obj<grid2>(grid_name);
            std::vector<fs::mixed_boundary_2d> mixed;
            std::vector<fs::fixed_boundary_2d> fixed;
            for (int b = 0; b < 4; b++) {
                if (bc_kinds[b] == 0) continue;
                auto dir = psum::field::boundary_direction_2d(b);
                std::string key = value_key(2, b);
                const double* n = edge_normals[1][b];
                const int kind = (cylindrical && b == 1) ? 3 : bc_kinds[b];
                if (kind == 1) {
                    auto bc = fsb::Dirichlet_line(g, dir);
                    bc = value_ref<2>(value_ptr(mng, key));
                    fixed.push_back(bc);
                } else if (kind == 2) {
                    auto bc = robin_2(g, dir, robin_center, n);
                    bc = value_ref<2>(value_ptr(mng, key));
                    mixed.push_back(bc);
                } else {
                    auto bc = fsb::Neumann_line(g, dir);
                    bc = value_ref<2>(value_ptr(mng, key));
                    mixed.push_back(bc);
                }
            }
            s.s2.emplace();
            if (!opts.empty()) s.s2->set_options(opts);
            s.s2->init(backend.empty() ? "native" : backend, g,
                        cylindrical ? fs::Poisson_solver_2d::Cylindrical
                                    : fs::Poisson_solver_2d::Cartesian,
                        list_from<fs::mixed_boundary_2d_list>(mixed),
                        list_from<fs::fixed_boundary_2d_list>(fixed), epsilon);
            return 0;
        }
        case 3: {
            if (!mng.template contains<grid3>(grid_name)) return -1;
            auto& g = mng.template obj<grid3>(grid_name);
            std::vector<fs::mixed_boundary_3d> mixed;
            std::vector<fs::fixed_boundary_3d> fixed;
            for (int b = 0; b < 6; b++) {
                if (bc_kinds[b] == 0) continue;
                auto dir = psum::field::boundary_direction_3d(b);
                std::string key = value_key(3, b);
                const double* n = edge_normals[2][b];
                if (bc_kinds[b] == 1) {
                    auto bc = fsb::Dirichlet_plane(g, dir);
                    bc = value_ref<3>(value_ptr(mng, key));
                    fixed.push_back(bc);
                } else if (bc_kinds[b] == 2) {
                    auto bc = robin_3(g, dir, robin_center, n);
                    bc = value_ref<3>(value_ptr(mng, key));
                    mixed.push_back(bc);
                } else {
                    auto bc = fsb::Neumann_plane(g, dir);
                    bc = value_ref<3>(value_ptr(mng, key));
                    mixed.push_back(bc);
                }
            }
            s.s3.emplace();
            if (!opts.empty()) s.s3->set_options(opts);
            s.s3->init(backend.empty() ? "native" : backend, g,
                        fs::Poisson_solver_3d::Cartesian,
                        list_from<fs::mixed_boundary_3d_list>(mixed),
                        list_from<fs::fixed_boundary_3d_list>(fixed), epsilon);
            return 0;
        }
        }
        } catch (const std::runtime_error&) {
            return -4;   // backend unavailable in this build
        }
        return -1;
    }

    template <typename Mng>
    int bc_value(Mng& mng, slots& s, int solver_kind, int boundary_index, double value) {
        const int dim = (solver_kind == 0) ? 1 : (solver_kind == 3) ? 3 : 2;
        if (boundary_index < 0 || boundary_index >= 2 * dim) return -1;
        std::string key = value_key(dim, boundary_index);
        if (!mng.template contains<double>(key)) return -2;   // periodic or unset edge
        mng.template obj<double>(key) = value;
        return 0;
    }

    // device flavor: device field data pointers straight into the solver
    template <typename Mng>
    int solve_d(Mng& mng, slots& s, int solver_kind,
                const char* rho_name, const char* phi_name) {
        auto solve_dim = [&]<int Dim>() -> int {
            if (!mng.template contains<field1<Dim>>(rho_name) ||
                !mng.template contains<field1<Dim>>(phi_name))
                return -1;
            auto& rho = mng.template obj<field1<Dim>>(rho_name);
            auto& phi = mng.template obj<field1<Dim>>(phi_name);
            if (s.native_forced[Dim]) {
                // degraded slot: native solves on the host — round-trip
                auto hr = rho.getContent().to_host();
                std::vector<double> hp(hr.size());
                if constexpr (Dim == 1) { if (!s.s1) return -1; s.s1->solve(hp.data(), hr.data()); }
                else if constexpr (Dim == 2) { if (!s.s2) return -1; s.s2->solve(hp.data(), hr.data()); }
                else { if (!s.s3) return -1; s.s3->solve(hp.data(), hr.data()); }
                phi.copy(hp);
                return 0;
            }
            if constexpr (Dim == 1) {
                if (!s.s1) return -1;
                s.s1->solve(phi.data(), rho.data());
            } else if constexpr (Dim == 2) {
                if (!s.s2) return -1;
                s.s2->solve(phi.data(), rho.data());
            } else {
                if (!s.s3) return -1;
                s.s3->solve(phi.data(), rho.data());
            }
            return 0;
        };
        if (solver_kind == 0) return solve_dim.template operator()<1>();
        if (solver_kind == 1 || solver_kind == 2) return solve_dim.template operator()<2>();
        if (solver_kind == 3) return solve_dim.template operator()<3>();
        return -1;
    }

    // host flavor: host field data pointers (requires a host backend)
    template <typename Mng>
    int solve_h(Mng& mng, slots& s, int solver_kind,
                const char* rho_name, const char* phi_name) {
        auto solve_dim = [&]<int Dim>() -> int {
            if (!mng.template contains<host_field1<Dim>>(rho_name) ||
                !mng.template contains<host_field1<Dim>>(phi_name))
                return -1;
            auto& rho = mng.template obj<host_field1<Dim>>(rho_name);
            auto& phi = mng.template obj<host_field1<Dim>>(phi_name);
            if constexpr (Dim == 1) {
                if (!s.s1) return -1;
                s.s1->solve(phi.data(), rho.data());
            } else if constexpr (Dim == 2) {
                if (!s.s2) return -1;
                s.s2->solve(phi.data(), rho.data());
            } else {
                if (!s.s3) return -1;
                s.s3->solve(phi.data(), rho.data());
            }
            return 0;
        };
        if (solver_kind == 0) return solve_dim.template operator()<1>();
        if (solver_kind == 1 || solver_kind == 2) return solve_dim.template operator()<2>();
        if (solver_kind == 3) return solve_dim.template operator()<3>();
        return -1;
    }

} // namespace poisson

} // namespace psum_runtime

#endif
