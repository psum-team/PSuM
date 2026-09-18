#include <iostream>
#include <vector>
#include <cmath>
#include "../../src/random/rander.hpp"
#include "../../src/particle_boundary/geometry.hpp"

using namespace psum::random;

bool randTest1(rander& R, int tryRound, double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3,
              double bx0, double by0, double bz0, double bx1, double by1, double bz1) {
    bool has_overlap = false;

    for (int i = 0; i < tryRound; i++) {
        double k1 = R();
        double k2 = R();
        double k3 = R();
        double k_sum = k1 + k2 + k3;
        k1 /= k_sum;
        k2 /= k_sum;
        k3 /= k_sum;
        double px = k1 * x1 + k2 * x2 + k3 * x3;
        double py = k1 * y1 + k2 * y2 + k3 * y3;
        double pz = k1 * z1 + k2 * z2 + k3 * z3;
        has_overlap = has_overlap || ((px > bx0 && px < bx1) && (py > by0 && py < by1) && (pz > bz0 && pz < bz1));
        if(has_overlap)
            return has_overlap;
    }
    return has_overlap;
}

void test_box_tri_overlap_random() {
    std::cout << "==========================================" << std::endl;
    std::cout << "check: box_tri_overlap_test" << std::endl;
    std::cout << " - fail count should be ZERO \n - positive divergence count should decrease with increasing random round" << std::endl;

    rander seed;
    auto R = [&seed]()
    { return 5 * seed() - 2; };

    for (int k = 1000; k <= 40000; k *= 1.5) {
        int pass_counter = 0;
        int same_counter = 0;
        int fail_counter = 0;
        int pos_dvrg_counter = 0;
        int neg_dvrg_counter = 0;

        for (int i = 0; i < 10000; i++) {
            std::vector<double> point = {R(), R(), R(), R(), R(), R(), R(), R(), R()};

            bool algo_result = psum::particle_boundary::geometry::box_tri_overlap_test(
                point[0], point[1], point[2], point[3], point[4], point[5], point[6], point[7], point[8],
                0, 0, 0, 1, 1, 1
            );

            if (algo_result)
                pass_counter++;
            else
                fail_counter++;

            bool rand_check = randTest1(seed, k, point[0], point[1], point[2], point[3], point[4], point[5],
                                       point[6], point[7], point[8], 0, 0, 0, 1, 1, 1);
            if (rand_check && (!algo_result))
                neg_dvrg_counter++;
            if ((!rand_check) && algo_result)
                pos_dvrg_counter++;
            if(algo_result == rand_check)
                same_counter++;
        }
        std::cout << "same_rate=" << same_counter * 100.0 / (pass_counter + fail_counter) << "% (allRound = 10,000);\n"
                  << "\t pos divergence count:" << pos_dvrg_counter
                  << "\t neg divergence count:" << neg_dvrg_counter
                  << "\t empty case:" << fail_counter
                  << "\t k=" << k << std::endl;
    }
}

int main() {
    test_box_tri_overlap_random();
    return 0;
}
