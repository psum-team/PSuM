#include <iostream>
#include <vector>
#include <cmath>
#include "../../src/random/rander.hpp"
#include "../../src/particle_boundary/geometry.hpp"

using namespace psum::random;
using namespace psum::particle_boundary::geometry;

void test_line_seg_tri_intersect() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: line_seg_tri_intersect_test standalone verification" << std::endl;
    std::cout << "(Verify lambda by direct geometric calculation)" << std::endl;

    rander seed;
    auto R = [&seed](){return seed() * 2.0 - 1.0;};

    int total_tests = 10000;
    int passed_tests = 0;
    int failed_tests = 0;

    auto verify_lambda = [](double x1, double y1, double z1,
                            double x2, double y2, double z2,
                            double x3, double y3, double z3,
                            double p1x, double p1y, double p1z,
                            double p2x, double p2y, double p2z,
                            double lambda) -> bool {
        if(lambda <= 0.0 || lambda > 1.0) {
            return false;
        }

        double qx = p1x + lambda * (p2x - p1x);
        double qy = p1y + lambda * (p2y - p1y);
        double qz = p1z + lambda * (p2z - p1z);

        double v0x = x3 - x1, v0y = y3 - y1, v0z = z3 - z1;
        double v1x = x2 - x1, v1y = y2 - y1, v1z = z2 - z1;
        double v2x = qx - x1, v2y = qy - y1, v2z = qz - z1;

        double d00 = v0x * v0x + v0y * v0y + v0z * v0z;
        double d01 = v0x * v1x + v0y * v1y + v0z * v1z;
        double d11 = v1x * v1x + v1y * v1y + v1z * v1z;
        double d20 = v2x * v0x + v2y * v0y + v2z * v0z;
        double d21 = v2x * v1x + v2y * v1y + v2z * v1z;

        double denom = d00 * d11 - d01 * d01;
        if(abs(denom) < psum::particle_boundary::geometry::finite_small) {
            return false;
        }

        double v = (d11 * d20 - d01 * d21) / denom;
        double w = (d00 * d21 - d01 * d20) / denom;

        return (v >= 0.0 && w >= 0.0 && (v + w) <= 1.0);
    };

/*
 * Verification principle:
 *
 * The verify_lambda function verifies that the computed intersection point is indeed on the triangle:
 *
 * 1. Compute the intersection point Q from the line segment using lambda:
 *    Q = p1 + lambda * (p2 - p1), where lambda in (0, 1]
 *
 * 2. Use barycentric coordinates to check if Q lies inside the triangle (A, B, C):
 *    - Define vectors from vertex A: v0 = C-A, v1 = B-A, v2 = Q-A
 *    - Compute dot products: d00 = v0·v0, d01 = v0·v1, d11 = v1·v1
 *    - Compute: d20 = v2·v0, d21 = v2·v1
 *    - Compute barycentric coordinates (v, w):
 *      v = (d11*d20 - d01*d21) / (d00*d11 - d01^2)
 *      w = (d00*d21 - d01*d20) / (d00*d11 - d01^2)
 *      u = 1 - v - w
 *
 * 3. Point Q is inside the triangle if and only if:
 *    v >= 0 && w >= 0 && (v + w) <= 1
 *
 * This provides a direct geometric verification independent of the Möller-Trumbore algorithm.
 */

    for (int i = 0; i < total_tests; i++) {
        double x1 = R(), y1 = R(), z1 = R();
        double x2 = R(), y2 = R(), z2 = R();
        double x3 = R(), y3 = R(), z3 = R();
        double p1x = R(), p1y = R(), p1z = R();
        double p2x = R(), p2y = R(), p2z = R();

        double lambda;

        bool intersected = psum::particle_boundary::geometry::line_seg_tri_intersect_test(
            x1, y1, z1, x2, y2, z2, x3, y3, z3,
            p1x, p1y, p1z, p2x, p2y, p2z, lambda);

        if(intersected) {
            bool verified = verify_lambda(x1, y1, z1, x2, y2, z2, x3, y3, z3,
                                       p1x, p1y, p1z, p2x, p2y, p2z, lambda);
            if(verified) {
                passed_tests++;
            } else {
                failed_tests++;
                std::cout << "FAILED verification at test " << i << ", lambda = " << lambda << std::endl;
            }
        }
    }

    std::cout << "Tests with intersection: " << passed_tests + failed_tests << std::endl;
    std::cout << "Passed: " << passed_tests << std::endl;
    std::cout << "Failed: " << failed_tests << std::endl;

    if(failed_tests == 0) {
        std::cout << "✅ All verification tests passed!" << std::endl;
    } else {
        std::cout << "❌ Some verification tests failed!" << std::endl;
    }
}

int main() {
    test_line_seg_tri_intersect();
    return 0;
}
