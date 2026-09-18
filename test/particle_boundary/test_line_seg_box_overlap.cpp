#include <iostream>
#include <vector>
#include <cmath>
#include "../../src/random/rander.hpp"
#include "../../src/particle_boundary/geometry.hpp"

using namespace psum::random;

bool rand_sampling_in_box(int tryRound, double box_x0, double box_y0, double box_z0,
                          double box_x1, double box_y1, double box_z1,
                          double x1, double y1, double z1, double x2, double y2, double z2) {
    std::vector<double> rands;
    double sum = 0;
    rander R;

    for (int i = 0; i < tryRound + 1; i++) {
        double newR = R();
        sum += newR;
        rands.push_back(newR);
    }
    double t = 0;
    for (int i = 0; i < tryRound; i++) {
        t += rands[i] / sum;
        double x_t = x1 + t * (x2 - x1);
        double y_t = y1 + t * (y2 - y1);
        double z_t = z1 + t * (z2 - z1);
        if(x_t >= box_x0 && x_t < box_x1 && y_t >= box_y0 && y_t < box_y1 && z_t >= box_z0 && z_t < box_z1){
            return true;
        }
    }
    return false;
}

void test_line_seg_box_overlap_random() {
    std::cout << "==========================================" << std::endl;
    std::cout << "test: line_seg_box_overlap_test" << std::endl;
    std::cout << " - fail count should be ZERO \n - true-subset count should decrease with increasing random round" << std::endl;

    rander seed;
    auto R = [&seed]() { return seed() * 5 - 2; };

    for (int k = 40; k <= 2000; k*=1.5){
        int same_counter = 0;
        int Tsubset_counter = 0;
        int fail_counter = 0;

        for (int i = 0; i < 10000; i++) {
            double x1 = R();
            double y1 = R();
            double z1 = R();
            double x2 = R();
            double y2 = R();
            double z2 = R();

            bool algo_result = psum::particle_boundary::geometry::line_seg_box_overlap_test(
                0, 0, 0, 1, 1, 1,
                x1, y1, z1, x2, y2, z2
            );

            bool rand_result = rand_sampling_in_box(k, 0, 0, 0, 1, 1, 1,
                                                x1, y1, z1, x2, y2, z2);

            if(algo_result == rand_result){
                same_counter++;
            }
            else{
                if(rand_result && !algo_result){
                    fail_counter++;
                }
                else{
                    Tsubset_counter++;
                }
            }
        }
        std::cout << "same_rate=" << same_counter * 100.0 / (same_counter + Tsubset_counter + fail_counter) << "% (allRound = 10,000);\n"
                  << "\t true subset count:" << Tsubset_counter << "\t failure count:" << fail_counter
                  << "\t k=" << k << std::endl;
    }
}

int main() {
    test_line_seg_box_overlap_random();
    return 0;
}
