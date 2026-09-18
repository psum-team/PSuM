#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <psum/particle_boundary.hpp>
#include "../../src/random/rander.hpp"

using namespace std;
using namespace psum::random;
using namespace psum::particle_boundary::geometry;

void randTest2(rander& R, int tryRound, int I, int J, int K, double x1, double y1, double z1, double x2, double y2, double z2, std::vector<int>& buffer_id) {
    buffer_id.resize(0);
    std::vector<double> rands;
    double sum = 0;
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
        if(x_t<0||x_t>=I||y_t<0||y_t>=J||z_t<0||z_t>=K){}
        else{
            int newId = floor(x_t) * J * K + floor(y_t) * K + floor(z_t);
            if(buffer_id.size()!=0)
            {
                if(buffer_id.back()!=newId) buffer_id.push_back(newId);
            }
            else buffer_id.push_back(newId);
        }
    }
}

/*
 * Verification principle:
 * 
 * The randTest2 function verifies the correctness of trace_to_idxs by using Monte Carlo sampling:
 * 
 * 1. Generate random numbers rands[0...tryRound] and normalize them to get parameters t in [0,1]
 * 2. For each t, compute the point on the line segment: P(t) = p1 + t * (p2 - p1)
 * 3. For each point P(t) inside the grid, compute the cell index: idx = floor(x) * J*K + floor(y) * K + floor(z)
 * 4. Collect unique cell indices (filtering duplicates)
 * 
 * The verification criteria:
 * - "same": The algorithm's output exactly matches the sampled cell indices
 * - "true-subset": The sampled indices are a subset of the algorithm's output (acceptable, as sampling may miss some cells)
 * - "failure": The sampled indices contain cells not found by the algorithm (indicates algorithm error)
 *
 * As tryRound increases (more sampling points), the true-subset count should decrease and same count should increase.
 */

void test_trace_to_idxs_consistency() {
    cout << "==========================================" << endl;
    cout << "check: trace_to_idxs" << endl;
    cout << " - fail count should be ZERO \n - true-subset count should decrease with increasing random round" << endl;
    rander seed;
    auto R = [&seed]()
    { return 8 * seed() - 2; };
    for (int k = 40; k <= 2000; k*=1.5){
        int same_counter = 0;
        int same_counter_empty = 0;
        int Tsubset_counter = 0;
        int fail_counter = 0;

        for (int i = 0; i < 10000; i++) {
            vector<int> ans_trace_to_idxs;
            vector<int> ans_randtest;

            vector<double> point = {R(), R(), R(), R(), R(), R()};

            trace_to_idxs_state state;
            int first_idx = trace_to_idxs_begin(4, 3, 2, point[0], point[1], point[2], point[3], point[4], point[5], state);
            for (int idx = first_idx; trace_to_idxs_valid(state); idx = trace_to_idxs_next(state)) {
                ans_trace_to_idxs.push_back(idx);
            }

            randTest2(seed, k, 4, 3, 2, point[0], point[1], point[2], point[3], point[4], point[5], ans_randtest);

            if(ans_trace_to_idxs==ans_randtest){
                same_counter++;
                if(ans_trace_to_idxs.size()==0) same_counter_empty++;
            }
            else{
                bool is_Tsubset = true;
                for(int v: ans_randtest)
                {
                    if(find(ans_trace_to_idxs.begin(),ans_trace_to_idxs.end(),v)==ans_trace_to_idxs.end())
                    {
                        is_Tsubset = false;
                        break;
                    }
                }
                if(is_Tsubset) Tsubset_counter++;
                else fail_counter++;
            }
        }
        cout << "same_rate=" << same_counter * 100.0 / (same_counter + Tsubset_counter + fail_counter) << "% (allRound = 10,000);\n"
                  << "\t true subset count:" << Tsubset_counter << "\t failure count:" << fail_counter <<"\t empty same:" <<same_counter_empty
                  << "\t k=" << k << endl;
    }
}

int main() {
    test_trace_to_idxs_consistency();
    return 0;
}
