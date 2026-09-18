#ifndef PSUM_PARTICLE_BOUNDARY_TRIANGLE_MESH_TRIGGER_HPP
#define PSUM_PARTICLE_BOUNDARY_TRIANGLE_MESH_TRIGGER_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include "geometry.hpp"
#include "device_partitioned_array.hpp"
#include "../field/device_array.hpp"

namespace psum {

namespace particle_boundary {

    template<typename T>
    concept like_integral = std::integral<T> || std::is_enum_v<T>;

    template <like_integral MeshId>
    struct triangle_mesh_with_id {
        std::vector<geometry::triangle_data> triangles_;
        MeshId id_;
    };

    template <like_integral MeshId>
    struct triangle_with_id {
        geometry::triangle_data data_;
        MeshId id_;
    };

    template <like_integral MeshId>
    class triangle_mesh_trigger_acc {
        using triangle_data = geometry::triangle_data;
    public:
        triangle_mesh_trigger_acc(
            int I, int J, int K,
            double x0, double y0, double z0,
            double x1, double y1, double z1,
            double r_dx, double r_dy, double r_dz,
            bool restrict_impl,
            bool udf_enabled,
            const device_partitioned_array<triangle_with_id<MeshId>>::acc_type& nodes_acc,
            const field::device_array_acc<float>& udf_acc
        ) : I_(I), J_(J), K_(K),
            x0_(x0), y0_(y0), z0_(z0),
            x1_(x1), y1_(y1), z1_(z1),
            r_dx_(r_dx), r_dy_(r_dy), r_dz_(r_dz),
            restrict_impl_(restrict_impl),
            udf_enabled_(udf_enabled),
            nodes_acc_(nodes_acc),
            udf_acc_(udf_acc) {}

        bool detect(double p1x, double p1y, double p1z,
                   double p2x, double p2y, double p2z,
                   double& k, double& nx, double& ny, double& nz, MeshId& id) const {

            using namespace geometry;

            double nx1 = (p1x - x0_) * r_dx_;
            double ny1 = (p1y - y0_) * r_dy_;
            double nz1 = (p1z - z0_) * r_dz_;
            double nx2 = (p2x - x0_) * r_dx_;
            double ny2 = (p2y - y0_) * r_dy_;
            double nz2 = (p2z - z0_) * r_dz_;

            if (restrict_impl_) {

                if (udf_enabled_) {
                    int ci = sycl::floor(nx1);
                    int cj = sycl::floor(ny1);
                    int ck = sycl::floor(nz1);

                    if (ci >= 0 && ci < I_ && cj >= 0 && cj < J_ && ck >= 0 && ck < K_) {
                        int cell_idx = ci * J_ * K_ + cj * K_ + ck;
                        float udf_val = udf_acc_[cell_idx];

                        if (udf_val > 1.0f) {
                            double dx = nx2 - nx1;
                            double dy = ny2 - ny1;
                            double dz = nz2 - nz1;
                            double sqared_length = dx * dx + dy * dy + dz * dz;

                            if (sqared_length < udf_val * udf_val) {
                                return false;
                            } else {
                                // skip at length of udf_val
                                double ray_length = sycl::sqrt(sqared_length);
                                nx1 += udf_val * dx / ray_length;
                                ny1 += udf_val * dy / ray_length;
                                nz1 += udf_val * dz / ray_length;
                            }
                        }
                    }
                }

                trace_to_idxs_state state;

                int first_idx = trace_to_idxs_begin(I_, J_, K_, nx1, ny1, nz1, nx2, ny2, nz2, state);

                if (first_idx >= 0) {
                    for (int idx = first_idx; trace_to_idxs_valid(state); idx = trace_to_idxs_next(state))
                    {
                        if (detect_in_cell_(idx, p1x, p1y, p1z, p2x, p2y, p2z, k, nx, ny, nz, id))
                            return true;
                    }
                }
                return false;
            } else {
                auto [in_1, in_2, idx_1, idx_2] = trace_to_idxs_fast(I_, J_, K_, nx1, ny1, nz1, nx2, ny2, nz2);
                if (in_1 && detect_in_cell_(idx_1, p1x, p1y, p1z, p2x, p2y, p2z, k, nx, ny, nz, id))
                    return true;
                if (in_2 && idx_1 != idx_2)
                    return detect_in_cell_(idx_2, p1x, p1y, p1z, p2x, p2y, p2z, k, nx, ny, nz, id);
                return false;
            }
        }

    private:
        int I_, J_, K_;
        double x0_, y0_, z0_;
        double x1_, y1_, z1_;
        double r_dx_, r_dy_, r_dz_;
        bool restrict_impl_;
        bool udf_enabled_;

        device_partitioned_array<triangle_with_id<MeshId>>::acc_type nodes_acc_;
        field::device_array_acc<float> udf_acc_;

        inline bool detect_in_cell_(int cell_idx,
                                   double p1x, double p1y, double p1z,
                                   double p2x, double p2y, double p2z,
                                   double& k, double& nx, double& ny, double& nz, MeshId& id) const {
            int n_trigs = nodes_acc_.partition_size(cell_idx);
            if (n_trigs == 0) {
                return false;
            }

            double min_lambda = 1.0;
            int min_idx = -1;
            bool found = false;

            const triangle_with_id<MeshId>* trigs = &nodes_acc_(cell_idx, 0);

            for (int i = 0; i < n_trigs; ++i) {
                const triangle_data& trig = trigs[i].data_;
                double lambda;
                if (geometry::line_seg_tri_intersect_test(
                    trig.p1x, trig.p1y, trig.p1z,
                    trig.p2x, trig.p2y, trig.p2z,
                    trig.p3x, trig.p3y, trig.p3z,
                    p1x, p1y, p1z, p2x, p2y, p2z,
                    lambda
                )) {
                    if (lambda < min_lambda) {
                        min_lambda = lambda;
                        min_idx = i;
                    }
                }
            }

            if (min_idx >= 0) {
                k = min_lambda;
                auto trig = trigs[min_idx];
                nx = trig.data_.nx;
                ny = trig.data_.ny;
                nz = trig.data_.nz;
                id = trig.id_;
                found = true;
            }

            return found;
        }
    };

    template <like_integral MeshId>
    class triangle_mesh_trigger {
    private:
        int I_, J_, K_;
        double x0_, y0_, z0_;
        double x1_, y1_, z1_;
        double span_x_, span_y_, span_z_;
        double r_span_x_, r_span_y_, r_span_z_;
        double r_dx_, r_dy_, r_dz_;
        bool inited_;
        bool restrict_impl_;
        bool udf_enabled_;
        mutable sycl::queue q_;

        using triangle_data = geometry::triangle_data;
        // it works like a vector<vector<triangle_data>> but on device
        device_partitioned_array<triangle_with_id<MeshId>> nodes_;
        // Unsigned Distance Field for acceleration
        field::device_array<float> udf_;
        // Occupancy map for UDF computation
        std::vector<bool> has_triangles_;

        inline void compute_normal_(triangle_data& trig) const {
            double e1x = trig.p2x - trig.p1x;
            double e1y = trig.p2y - trig.p1y;
            double e1z = trig.p2z - trig.p1z;

            double e2x = trig.p3x - trig.p1x;
            double e2y = trig.p3y - trig.p1y;
            double e2z = trig.p3z - trig.p1z;

            double nx = e1y * e2z - e1z * e2y;
            double ny = e1z * e2x - e1x * e2z;
            double nz = e1x * e2y - e1y * e2x;

            double norm = sycl::sqrt(nx * nx + ny * ny + nz * nz);
            if (norm < geometry::finite_small) {
                throw std::runtime_error("Error: triangle has zero area");
            }

            trig.nx = nx / norm;
            trig.ny = ny / norm;
            trig.nz = nz / norm;
        }

    public:
        using acc_type = triangle_mesh_trigger_acc<MeshId>;
        triangle_mesh_trigger(const sycl::queue& q)
            : I_(0), J_(0), K_(0),
              x0_(0), y0_(0), z0_(0),
              x1_(0), y1_(0), z1_(0),
              span_x_(0), span_y_(0), span_z_(0),
              r_span_x_(0), r_span_y_(0), r_span_z_(0),
              r_dx_(0), r_dy_(0), r_dz_(0),
              restrict_impl_(false),
              udf_enabled_(false),
              udf_(q),
              nodes_(q), q_(q),
              inited_(false) {}

        void set(const std::vector<triangle_with_id<MeshId>>& triangles,
                double x0, double y0, double z0,
                double x1, double y1, double z1,
                int I, int J, int K) {
            if (inited_) {
                throw std::runtime_error("Error: triangle_mesh_trigger::set() called multiple times");
            }

            I_ = I;
            J_ = J;
            K_ = K;
            x0_ = x0;
            y0_ = y0;
            z0_ = z0;
            x1_ = x1;
            y1_ = y1;
            z1_ = z1;

            span_x_ = x1_ - x0_;
            span_y_ = y1_ - y0_;
            span_z_ = z1_ - z0_;

            if (span_x_ < geometry::finite_small ||
                span_y_ < geometry::finite_small ||
                span_z_ < geometry::finite_small) {
                throw std::runtime_error("Error: invalid span dimensions");
            }

            r_span_x_ = 1.0 / span_x_;
            r_span_y_ = 1.0 / span_y_;
            r_span_z_ = 1.0 / span_z_;

            r_dx_ = static_cast<double>(I_) / span_x_;
            r_dy_ = static_cast<double>(J_) / span_y_;
            r_dz_ = static_cast<double>(K_) / span_z_;

            std::vector<std::vector<triangle_with_id<MeshId>>> nodes_host(I_ * J_ * K_);
            
            for (size_t t_id = 0; t_id < triangles.size(); ++t_id) {
                triangle_data trig = triangles[t_id].data_;

                try {
                    compute_normal_(trig);
                } catch (const std::runtime_error& e) {
                    std::cerr << "Error: " << e.what() << std::endl;
                    continue;
                }
                
                double minx = sycl::fmin(trig.p1x, sycl::fmin(trig.p2x, trig.p3x));
                double maxx = sycl::fmax(trig.p1x, sycl::fmax(trig.p2x, trig.p3x));
                double miny = sycl::fmin(trig.p1y, sycl::fmin(trig.p2y, trig.p3y));
                double maxy = sycl::fmax(trig.p1y, sycl::fmax(trig.p2y, trig.p3y));
                double minz = sycl::fmin(trig.p1z, sycl::fmin(trig.p2z, trig.p3z));
                double maxz = sycl::fmax(trig.p1z, sycl::fmax(trig.p2z, trig.p3z));
                
                int I0 = static_cast<int>(sycl::floor((minx - x0_) * r_dx_));
                int J0 = static_cast<int>(sycl::floor((miny - y0_) * r_dy_));
                int K0 = static_cast<int>(sycl::floor((minz - z0_) * r_dz_));
                int I1 = static_cast<int>(sycl::ceil((maxx - x0_) * r_dx_));
                int J1 = static_cast<int>(sycl::ceil((maxy - y0_) * r_dy_));
                int K1 = static_cast<int>(sycl::ceil((maxz - z0_) * r_dz_));

                I0 = I0 < 0 ? 0 : (I0 >= I_ ? I_ - 1 : I0);
                I1 = I1 < 0 ? 0 : (I1 >= I_ ? I_ - 1 : I1);
                J0 = J0 < 0 ? 0 : (J0 >= J_ ? J_ - 1 : J0);
                J1 = J1 < 0 ? 0 : (J1 >= J_ ? J_ - 1 : J1);
                K0 = K0 < 0 ? 0 : (K0 >= K_ ? K_ - 1 : K0);
                K1 = K1 < 0 ? 0 : (K1 >= K_ ? K_ - 1 : K1);

                for (int i = I0; i <= I1; ++i) {
                    for (int j = J0; j <= J1; ++j) {
                        for (int k = K0; k <= K1; ++k) {
                            int cell_idx = i * J_ * K_ + j * K_ + k;

                            double dx = span_x_ / static_cast<double>(I_);
                            double dy = span_y_ / static_cast<double>(J_);
                            double dz = span_z_ / static_cast<double>(K_);

                            double bx0 = x0_ + i * dx;
                            double by0 = y0_ + j * dy;
                            double bz0 = z0_ + k * dz;
                            double bx1 = bx0 + dx;
                            double by1 = by0 + dy;
                            double bz1 = bz0 + dz;

                            if (geometry::box_tri_overlap_test(
                                trig.p1x, trig.p1y, trig.p1z,
                                trig.p2x, trig.p2y, trig.p2z,
                                trig.p3x, trig.p3y, trig.p3z,
                                bx0, by0, bz0, bx1, by1, bz1
                            )) {
                                // NOT triangles[t_id] !!
                                // because trig is not triangles[t_id].data_
                                // its normal might be modified by compute_normal_()
                                nodes_host[cell_idx].push_back({trig, triangles[t_id].id_});
                            }
                        }
                    }
                }
            }

            inited_ = true;
            nodes_ = device_partitioned_array<triangle_with_id<MeshId>>(q_, nodes_host);

            has_triangles_.resize(I_ * J_ * K_);
            for (int ci = 0; ci < I_; ++ci) {
                for (int cj = 0; cj < J_; ++cj) {
                    for (int ck = 0; ck < K_; ++ck) {
                        int cell_idx = ci * J_ * K_ + cj * K_ + ck;
                        has_triangles_[cell_idx] = !nodes_host[cell_idx].empty();
                    }
                }
            }

            std::vector<float> udf_host(I_ * J_ * K_, 0.0f);
            udf_ = field::device_array<float>(q_, udf_host);
        }

        acc_type get_access(sycl::handler& h) const {
            if (!inited_) {
                throw std::runtime_error("Error: triangle_mesh_trigger::get_access() called before set()");
            }
            return acc_type(
                I_, J_, K_,
                x0_, y0_, z0_,
                x1_, y1_, z1_,
                r_dx_, r_dy_, r_dz_,
                restrict_impl_,
                udf_enabled_,
                nodes_.get_access(h),
                udf_.get_access(h)
            );
        }

        int get_I() const { return I_; }
        int get_J() const { return J_; }
        int get_K() const { return K_; }
        double get_x0() const { return x0_; }
        double get_y0() const { return y0_; }
        double get_z0() const { return z0_; }
        double get_x1() const { return x1_; }
        double get_y1() const { return y1_; }
        double get_z1() const { return z1_; }
        double span_x() const { return span_x_; }
        double span_y() const { return span_y_; }
        double span_z() const { return span_z_; }

        void set_fast() {restrict_impl_ = false;}
        void set_restrict() {restrict_impl_ = true;}
        void set_acceleration_UDF(bool enabled) {
            udf_enabled_ = enabled;
            if (enabled) {
                if (!inited_) {
                    throw std::runtime_error("Error: triangle_mesh_trigger::set_acceleration_UDF() called before set()");
                }
                std::vector<float> udf_host;
                geometry::compute_udf(I_, J_, K_, has_triangles_, udf_host);
                udf_ = field::device_array<float>(q_, udf_host);
            }
        }

        void set(const std::vector<triangle_mesh_with_id<MeshId>>& meshes, int I, int J, int K) {
            std::vector<triangle_with_id<MeshId>> triangles;
            for (const auto& mesh : meshes) {
                for (const auto& trig : mesh.triangles_) {
                    triangles.push_back({trig, mesh.id_});
                }
            }

            if (triangles.empty()) {
                throw std::runtime_error("Error: triangle_mesh_trigger::set() called with empty triangles vector");
            }
            
            double x0 = triangles[0].data_.p1x;
            double x1 = triangles[0].data_.p1x;
            double y0 = triangles[0].data_.p1y;
            double y1 = triangles[0].data_.p1y;
            double z0 = triangles[0].data_.p1z;
            double z1 = triangles[0].data_.p1z;
            
            for (const auto& trig : triangles) {
                x0 = std::min({x0, trig.data_.p1x, trig.data_.p2x, trig.data_.p3x});
                x1 = std::max({x1, trig.data_.p1x, trig.data_.p2x, trig.data_.p3x});
                y0 = std::min({y0, trig.data_.p1y, trig.data_.p2y, trig.data_.p3y});
                y1 = std::max({y1, trig.data_.p1y, trig.data_.p2y, trig.data_.p3y});
                z0 = std::min({z0, trig.data_.p1z, trig.data_.p2z, trig.data_.p3z});
                z1 = std::max({z1, trig.data_.p1z, trig.data_.p2z, trig.data_.p3z});
            }
            double smaller_x0 = x0 - 0.1003 * (x1 - x0);
            double smaller_y0 = y0 - 0.1027 * (y1 - y0);
            double smaller_z0 = z0 - 0.1047 * (z1 - z0);
            double larger_x1 = x1 + 0.1057 * (x1 - x0);
            double larger_y1 = y1 + 0.1059 * (y1 - y0);
            double larger_z1 = z1 + 0.1069 * (z1 - z0);
            set(triangles, smaller_x0, smaller_y0, smaller_z0, larger_x1, larger_y1, larger_z1, I, J, K);
        }
    };

}

}

#endif

