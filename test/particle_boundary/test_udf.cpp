#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <psum/field.hpp>
#include <psum/particle_boundary.hpp>
#include "../../src/random/rander.hpp"
#include "../../src/timer.hpp"

using namespace std;
using namespace psum::particle_boundary::geometry;
using namespace psum::random;

float compute_precise_lower_bound(int I, int J, int K, const std::vector<bool>& has_triangles,
                                  int ci, int cj, int ck) {
    if (has_triangles[ci * J * K + cj * K + ck]) return 0.0f;

    float min_dist = std::numeric_limits<float>::max();
    for (int ti = 0; ti < I; ++ti) {
        for (int tj = 0; tj < J; ++tj) {
            for (int tk = 0; tk < K; ++tk) {
                int idx = ti * J * K + tj * K + tk;
                if (has_triangles[idx]) {
                    float dx = abs(ci - ti);
                    float dy = abs(cj - tj);
                    float dz = abs(ck - tk);
                    float lb_i = std::max(dx - 1.0f, 0.0f);
                    float lb_j = std::max(dy - 1.0f, 0.0f);
                    float lb_k = std::max(dz - 1.0f, 0.0f);
                    float dist = std::sqrt(lb_i*lb_i + lb_j*lb_j + lb_k*lb_k);
                    min_dist = std::min(min_dist, dist);
                }
            }
        }
    }
    return min_dist;
}

void test_basic_grid_cases() {
    cout << "=== Test: Basic grid cases ===" << endl;

    int I = 10, J = 10, K = 10;
    std::vector<float> udf;
    
    // full empty
    std::vector<bool> empty(I*J*K, false);
    compute_udf(I,J,K,empty,udf);
    bool empty_ok = std::all_of(udf.begin(), udf.end(), [](float v){ return v >= 0.0f; });
    cout << (empty_ok ? "✅ Empty grid OK" : "❌ Empty grid failed") << endl;

    // full occupied
    std::vector<bool> full(I*J*K, true);
    compute_udf(I,J,K,full,udf);
    bool full_ok = std::all_of(udf.begin(), udf.end(), [](float v){ return fabs(v) < 1e-6f; });
    cout << (full_ok ? "✅ Full grid OK" : "❌ Full grid failed") << endl;

    // single triangle
    std::vector<bool> single(I*J*K,false);
    int cx=5, cy=5, cz=5;
    single[cx*J*K + cy*K + cz] = true;
    compute_udf(I,J,K,single,udf);
    bool center_zero = fabs(udf[cx*J*K + cy*K + cz]) < 1e-6f;
    cout << (center_zero ? "✅ Single triangle cell OK" : "❌ Single triangle cell failed") << endl;
}

void test_random_conservative() {
    cout << "=== Test: Random conservative check ===" << endl;

    int I=20, J=20, K=20;
    std::vector<bool> has_triangles(I*J*K,false);
    std::vector<float> udf;
    rander rng(12345);

    // randomly add 100 triangles
    for(int n=0;n<100;++n){
        int ci = static_cast<int>(rng()*I);
        int cj = static_cast<int>(rng()*J);
        int ck = static_cast<int>(rng()*K);
        has_triangles[ci*J*K + cj*K + ck] = true;
    }

    compute_udf(I,J,K,has_triangles,udf);

    bool all_conservative = true;
    float max_error = 0.0f;

    // randomly sample 1000 cells and check lower bound
    for(int n=0;n<1000;++n){
        int ci = static_cast<int>(rng()*I);
        int cj = static_cast<int>(rng()*J);
        int ck = static_cast<int>(rng()*K);
        int idx = ci*J*K + cj*K + ck;
        if(!has_triangles[idx]){
            float lb = compute_precise_lower_bound(I,J,K,has_triangles,ci,cj,ck);
            if(udf[idx] - lb > 1e-6f){
                cout << "❌ UDF exceeds lower bound at ("<<ci<<","<<cj<<","<<ck<<")"
                     << " udf="<<udf[idx]<<" lb="<<lb<<endl;
                all_conservative = false;
            }
            max_error = std::max(max_error, lb - udf[idx]);
        }
    }
    cout << (all_conservative ? "✅ All sampled UDF conservative" : "❌ Some UDF too large") 
         << ", max conservative error: "<< max_error << endl;
}

void visualize_udf() {
    int I=100, J=200, K=1;
    std::vector<bool> has_triangles(I*J*K,false);
    rander rng(12345);
    //random 10 triangle
    for(int n=0;n<10;++n){
        int ci = static_cast<int>(rng() * I);
        int cj = static_cast<int>(rng() * J);
        int ck = static_cast<int>(rng() * K);
        has_triangles[ci*J*K + cj*K + ck] = true;
    }
    std::vector<int> udf_k = {8, 16, 24, 32, 40};
    for(int k : udf_k) {
        std::vector<float> udf;
        compute_udf(I,J,K,has_triangles,udf, k);
        
        using namespace psum::field;
        sycl::queue q;
        using grid = simple_grid<2>;
        grid g({0.0, 0.0}, {double(I), double(J)}, {I, J});
        using df = device_field<2, var_loc::cellCentered, float>;
        df udf_field(device_array<float>(q, udf), g);
        udf_field.plot("udf" + std::to_string(k) + ".plt", "udf");
    }
}

int main() {
    test_basic_grid_cases();
    test_random_conservative();
    visualize_udf();
    return 0;
}