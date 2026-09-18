#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;

host_node_field2D<double> read_node_field_plt(const string& filename) {
    ifstream in(filename);

    string line;
    string zone;
    getline(in, line);
    getline(in, zone);

    vector<double> values;
    double xmin = 1e300;
    double xmax = -1e300;
    double ymin = 1e300;
    double ymax = -1e300;
    double x = 0.0;
    double y = 0.0;
    double v = 0.0;
    while (getline(in, line)) {
        istringstream row(line);
        if (!(row >> x >> y >> v)) {
            continue;
        }
        xmin = fmin(xmin, x);
        xmax = fmax(xmax, x);
        ymin = fmin(ymin, y);
        ymax = fmax(ymax, y);
        values.push_back(v);
    }

    int ni = stoi(zone.substr(zone.find('=', zone.find('i')) + 1));
    int nj = stoi(zone.substr(zone.find('=', zone.find('j')) + 1));

    grid2D grid({xmin, ymin}, {xmax, ymax}, {nj - 1, ni - 1});
    return host_node_field2D<double>(std::move(values), grid);
}

int main() {
    const string bitmap_file = "../../docs/psum_tour/psum_bitmap.plt";
    auto mask = read_node_field_plt(bitmap_file);
    mask.plot("output/mask.plt", "mask", 0.0);
    cout << "Neon Light stage 1: wrote output/mask.plt" << endl;
    return 0;
}
