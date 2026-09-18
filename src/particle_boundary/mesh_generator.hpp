#ifndef PSUM_PARTICLE_BOUNDARY_MESH_GENERATOR_HPP
#define PSUM_PARTICLE_BOUNDARY_MESH_GENERATOR_HPP

#include <vector>
#include <cmath>
#include "geometry.hpp"

namespace psum {

namespace particle_boundary {

namespace mesh_generator {

    using geometry::triangle_data;

    inline std::vector<triangle_data> axis_aligned_cube(double x0, double y0, double z0,
                                                         double x1, double y1, double z1) {
        std::vector<triangle_data> triangles;

        double vertices[8][3] = {
            {x0, y0, z0}, {x1, y0, z0}, {x0, y1, z0}, {x1, y1, z0},
            {x0, y0, z1}, {x1, y0, z1}, {x0, y1, z1}, {x1, y1, z1}
        };

        int faces[6][4] = {
            {0, 1, 3, 2},
            {4, 5, 7, 6},
            {0, 1, 5, 4},
            {2, 3, 7, 6},
            {0, 2, 6, 4},
            {1, 3, 7, 5}
        };

        for (int f = 0; f < 6; ++f) {
            int v0 = faces[f][0];
            int v1 = faces[f][1];
            int v2 = faces[f][2];
            int v3 = faces[f][3];

            triangle_data trig1;
            trig1.p1x = vertices[v0][0]; trig1.p1y = vertices[v0][1]; trig1.p1z = vertices[v0][2];
            trig1.p2x = vertices[v1][0]; trig1.p2y = vertices[v1][1]; trig1.p2z = vertices[v1][2];
            trig1.p3x = vertices[v2][0]; trig1.p3y = vertices[v2][1]; trig1.p3z = vertices[v2][2];

            triangle_data trig2;
            trig2.p1x = vertices[v0][0]; trig2.p1y = vertices[v0][1]; trig2.p1z = vertices[v0][2];
            trig2.p2x = vertices[v2][0]; trig2.p2y = vertices[v2][1]; trig2.p2z = vertices[v2][2];
            trig2.p3x = vertices[v3][0]; trig2.p3y = vertices[v3][1]; trig2.p3z = vertices[v3][2];

            double e1x = trig1.p2x - trig1.p1x;
            double e1y = trig1.p2y - trig1.p1y;
            double e1z = trig1.p2z - trig1.p1z;
            double e2x = trig1.p3x - trig1.p1x;
            double e2y = trig1.p3y - trig1.p1y;
            double e2z = trig1.p3z - trig1.p1z;

            double nx1 = e1y * e2z - e1z * e2y;
            double ny1 = e1z * e2x - e1x * e2z;
            double nz1 = e1x * e2y - e1y * e2x;
            double norm1 = std::sqrt(nx1 * nx1 + ny1 * ny1 + nz1 * nz1);

            trig1.nx = nx1 / norm1;
            trig1.ny = ny1 / norm1;
            trig1.nz = nz1 / norm1;

            double e3x = trig2.p2x - trig2.p1x;
            double e3y = trig2.p2y - trig2.p1y;
            double e3z = trig2.p2z - trig2.p1z;
            double e4x = trig2.p3x - trig2.p1x;
            double e4y = trig2.p3y - trig2.p1y;
            double e4z = trig2.p3z - trig2.p1z;

            double nx2 = e3y * e4z - e3z * e4y;
            double ny2 = e3z * e4x - e3x * e4z;
            double nz2 = e3x * e4y - e3y * e4x;
            double norm2 = std::sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);

            trig2.nx = nx2 / norm2;
            trig2.ny = ny2 / norm2;
            trig2.nz = nz2 / norm2;

            triangles.push_back(trig1);
            triangles.push_back(trig2);
        }

        return triangles;
    }

    inline std::vector<triangle_data> extrude_xy_line_to_width(const std::vector<std::pair<double, double>>& xy_points,
                                                                  double z_width) {
        std::vector<triangle_data> triangles;

        if (xy_points.size() < 2) {
            return triangles;
        }

        double z0 = -z_width / 2.0;
        double z1 = z_width / 2.0;

        for (size_t i = 0; i < xy_points.size() - 1; ++i) {
            double x0 = xy_points[i].first;
            double y0 = xy_points[i].second;
            double x1 = xy_points[i + 1].first;
            double y1 = xy_points[i + 1].second;

            triangle_data trig1, trig2, trig3, trig4;

            trig1.p1x = x0; trig1.p1y = y0; trig1.p1z = z0;
            trig1.p2x = x1; trig1.p2y = y1; trig1.p2z = z0;
            trig1.p3x = x1; trig1.p3y = y1; trig1.p3z = z1;
            trig1.nx = 0.0; trig1.ny = 0.0; trig1.nz = 1.0;

            trig2.p1x = x0; trig2.p1y = y0; trig2.p1z = z0;
            trig2.p2x = x1; trig2.p2y = y1; trig2.p2z = z1;
            trig2.p3x = x0; trig2.p3y = y0; trig2.p3z = z1;
            trig2.nx = 0.0; trig2.ny = 0.0; trig2.nz = 1.0;

            double dx = x1 - x0;
            double dy = y1 - y0;
            double len = std::sqrt(dx * dx + dy * dy);
            double nx = -dy / len;
            double ny = dx / len;

            trig3.p1x = x0; trig3.p1y = y0; trig3.p1z = z0;
            trig3.p2x = x1; trig3.p2y = y1; trig3.p2z = z0;
            trig3.p3x = x0; trig3.p3y = y0; trig3.p3z = z1;
            trig3.nx = nx; trig3.ny = ny; trig3.nz = 0.0;

            trig4.p1x = x1; trig4.p1y = y1; trig4.p1z = z0;
            trig4.p2x = x1; trig4.p2y = y1; trig4.p2z = z1;
            trig4.p3x = x0; trig4.p3y = y0; trig4.p3z = z1;
            trig4.nx = nx; trig4.ny = ny; trig4.nz = 0.0;

            triangles.push_back(trig1);
            triangles.push_back(trig2);
            triangles.push_back(trig3);
            triangles.push_back(trig4);
        }

        return triangles;
    }

    inline std::vector<triangle_data> axis_aligned_plane(double x0, double y0, double z0,
                                                         double x1, double y1, double z1) {
        std::vector<triangle_data> triangles;

        bool fix_x = (x0 == x1);
        bool fix_y = (y0 == y1);
        bool fix_z = (z0 == z1);

        if (!(fix_x || fix_y || fix_z)) {
            return triangles;
        }

        double vertices[4][3];
        if (fix_x) {
            double x = x0;
            vertices[0][0] = x; vertices[0][1] = y0; vertices[0][2] = z0;
            vertices[1][0] = x; vertices[1][1] = y1; vertices[1][2] = z0;
            vertices[2][0] = x; vertices[2][1] = y1; vertices[2][2] = z1;
            vertices[3][0] = x; vertices[3][1] = y0; vertices[3][2] = z1;
        } else if (fix_y) {
            double y = y0;
            vertices[0][0] = x0; vertices[0][1] = y; vertices[0][2] = z0;
            vertices[1][0] = x1; vertices[1][1] = y; vertices[1][2] = z0;
            vertices[2][0] = x1; vertices[2][1] = y; vertices[2][2] = z1;
            vertices[3][0] = x0; vertices[3][1] = y; vertices[3][2] = z1;
        } else {
            double z = z0;
            vertices[0][0] = x0; vertices[0][1] = y0; vertices[0][2] = z;
            vertices[1][0] = x1; vertices[1][1] = y0; vertices[1][2] = z;
            vertices[2][0] = x1; vertices[2][1] = y1; vertices[2][2] = z;
            vertices[3][0] = x0; vertices[3][1] = y1; vertices[3][2] = z;
        }

        triangle_data trig1;
        trig1.p1x = vertices[0][0]; trig1.p1y = vertices[0][1]; trig1.p1z = vertices[0][2];
        trig1.p2x = vertices[1][0]; trig1.p2y = vertices[1][1]; trig1.p2z = vertices[1][2];
        trig1.p3x = vertices[2][0]; trig1.p3y = vertices[2][1]; trig1.p3z = vertices[2][2];

        triangle_data trig2;
        trig2.p1x = vertices[0][0]; trig2.p1y = vertices[0][1]; trig2.p1z = vertices[0][2];
        trig2.p2x = vertices[2][0]; trig2.p2y = vertices[2][1]; trig2.p2z = vertices[2][2];
        trig2.p3x = vertices[3][0]; trig2.p3y = vertices[3][1]; trig2.p3z = vertices[3][2];

        double e1x = trig1.p2x - trig1.p1x;
        double e1y = trig1.p2y - trig1.p1y;
        double e1z = trig1.p2z - trig1.p1z;
        double e2x = trig1.p3x - trig1.p1x;
        double e2y = trig1.p3y - trig1.p1y;
        double e2z = trig1.p3z - trig1.p1z;

        double nx1 = e1y * e2z - e1z * e2y;
        double ny1 = e1z * e2x - e1x * e2z;
        double nz1 = e1x * e2y - e1y * e2x;
        double norm1 = std::sqrt(nx1 * nx1 + ny1 * ny1 + nz1 * nz1);

        trig1.nx = nx1 / norm1;
        trig1.ny = ny1 / norm1;
        trig1.nz = nz1 / norm1;

        double e3x = trig2.p2x - trig2.p1x;
        double e3y = trig2.p2y - trig2.p1y;
        double e3z = trig2.p2z - trig2.p1z;
        double e4x = trig2.p3x - trig2.p1x;
        double e4y = trig2.p3y - trig2.p1y;
        double e4z = trig2.p3z - trig2.p1z;

        double nx2 = e3y * e4z - e3z * e4y;
        double ny2 = e3z * e4x - e3x * e4z;
        double nz2 = e3x * e4y - e3y * e4x;
        double norm2 = std::sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);

        trig2.nx = nx2 / norm2;
        trig2.ny = ny2 / norm2;
        trig2.nz = nz2 / norm2;

        triangles.push_back(trig1);
        triangles.push_back(trig2);

        return triangles;
    }

    inline std::vector<triangle_data> revolve_zr_curve(const std::vector<std::pair<double, double>>& zr_points,
                                                        int azimuthal_segments) {
        std::vector<triangle_data> triangles;

        if (zr_points.size() < 2) {
            return triangles;
        }

        double dtheta = 2.0 * M_PI / azimuthal_segments;

        for (int i = 0; i < azimuthal_segments; ++i) {
            double theta0 = i * dtheta;
            double theta1 = (i + 1) * dtheta;
            double cos0 = std::cos(theta0);
            double sin0 = std::sin(theta0);
            double cos1 = std::cos(theta1);
            double sin1 = std::sin(theta1);

            for (size_t j = 0; j < zr_points.size() - 1; ++j) {
                double z0 = zr_points[j].first;
                double r0 = zr_points[j].second;
                double z1 = zr_points[j + 1].first;
                double r1 = zr_points[j + 1].second;

                double x00 = r0 * cos0;
                double y00 = r0 * sin0;
                double x01 = r0 * cos1;
                double y01 = r0 * sin1;
                double x10 = r1 * cos0;
                double y10 = r1 * sin0;
                double x11 = r1 * cos1;
                double y11 = r1 * sin1;

                triangle_data trig1, trig2;

                trig1.p1x = x00; trig1.p1y = y00; trig1.p1z = z0;
                trig1.p2x = x10; trig1.p2y = y10; trig1.p2z = z1;
                trig1.p3x = x11; trig1.p3y = y11; trig1.p3z = z1;

                trig2.p1x = x00; trig2.p1y = y00; trig2.p1z = z0;
                trig2.p2x = x11; trig2.p2y = y11; trig2.p2z = z1;
                trig2.p3x = x01; trig2.p3y = y01; trig2.p3z = z0;

                double e1x = trig1.p2x - trig1.p1x;
                double e1y = trig1.p2y - trig1.p1y;
                double e1z = trig1.p2z - trig1.p1z;
                double e2x = trig1.p3x - trig1.p1x;
                double e2y = trig1.p3y - trig1.p1y;
                double e2z = trig1.p3z - trig1.p1z;

                double nx1 = e1y * e2z - e1z * e2y;
                double ny1 = e1z * e2x - e1x * e2z;
                double nz1 = e1x * e2y - e1y * e2x;
                double norm1 = std::sqrt(nx1 * nx1 + ny1 * ny1 + nz1 * nz1);

                trig1.nx = nx1 / norm1;
                trig1.ny = ny1 / norm1;
                trig1.nz = nz1 / norm1;

                double e3x = trig2.p2x - trig2.p1x;
                double e3y = trig2.p2y - trig2.p1y;
                double e3z = trig2.p2z - trig2.p1z;
                double e4x = trig2.p3x - trig2.p1x;
                double e4y = trig2.p3y - trig2.p1y;
                double e4z = trig2.p3z - trig2.p1z;

                double nx2 = e3y * e4z - e3z * e4y;
                double ny2 = e3z * e4x - e3x * e4z;
                double nz2 = e3x * e4y - e3y * e4x;
                double norm2 = std::sqrt(nx2 * nx2 + ny2 * ny2 + nz2 * nz2);

                trig2.nx = nx2 / norm2;
                trig2.ny = ny2 / norm2;
                trig2.nz = nz2 / norm2;

                triangles.push_back(trig1);
                triangles.push_back(trig2);
            }
        }

        return triangles;
    }

    inline void rescale_triangles(std::vector<triangle_data>& triangles, double scale) {
        double center_x = 0.0;
        double center_y = 0.0;
        double center_z = 0.0;
        for (auto& t : triangles) {
            center_x += (t.p1x + t.p2x + t.p3x) / 3.0;
            center_y += (t.p1y + t.p2y + t.p3y) / 3.0;
            center_z += (t.p1z + t.p2z + t.p3z) / 3.0;
        }
        center_x /= triangles.size();
        center_y /= triangles.size();
        center_z /= triangles.size();
        for (auto& t : triangles) {
            t.p1x = (t.p1x - center_x) * scale + center_x;
            t.p1y = (t.p1y - center_y) * scale + center_y;
            t.p1z = (t.p1z - center_z) * scale + center_z;
            t.p2x = (t.p2x - center_x) * scale + center_x;
            t.p2y = (t.p2y - center_y) * scale + center_y;
            t.p2z = (t.p2z - center_z) * scale + center_z;
            t.p3x = (t.p3x - center_x) * scale + center_x;
            t.p3y = (t.p3y - center_y) * scale + center_y;
            t.p3z = (t.p3z - center_z) * scale + center_z;
        }
    }
}

}

}

#endif
