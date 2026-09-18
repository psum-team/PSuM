#include <psum/serialization.hpp>
#include <psum/tag.hpp>
#include <iostream>

using namespace std;
using namespace psum::tag;
using namespace psum::tag::property;
using namespace psum::serialization;

using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<weight, double>
>;

auto check_same = [](Particle& p1, Particle& p2) {
    try {
        assert(get<position>(p1) == get<position>(p2));
        assert(get<velocity>(p1) == get<velocity>(p2));
        assert(get<weight>(p1) == get<weight>(p2));
        return true;
    } catch (const std::exception& e) {
        return false;
    }
};

int main() {

	mas_file fp("tagsl.mas", mas_file::replaceMode);
    // single particle
    {
        Particle p;
        get<velocity>(p) << Eigen::RowVector3d::Random();
        get<position>(p) << Eigen::RowVector3d::Random();
        get<weight>(p) = Eigen::Vector<double, 1>::Random()(0);
	    save(fp, "P", p);
        
	    Particle p2;
	    load(fp, "P", p2);
        bool same = check_same(p, p2);

        if (same) cout << "single S/L test : pass." << endl;
        else cout << "single S/L test : fail." << endl;
    }

    // multiple particles
    {
        std::vector<Particle> particles(2000);
        for (int i = 0; i < particles.size(); i++) {
            get<velocity>(particles[i]) << Eigen::RowVector3d::Random();
            get<position>(particles[i]) << Eigen::RowVector3d::Random();
            get<weight>(particles[i]) = Eigen::Vector<double, 1>::Random()(0);
        }
        save(fp, "Ps", particles);

        std::vector<Particle> particles2;
        load(fp, "Ps", particles2);

        if (particles.size() != particles2.size()) {
            cout << "multiple S/L test : fail." << endl;
        } else {
            bool same = true;
            for (int i = 0; i < particles.size(); i++) {
                same &= check_same(particles[i], particles2[i]);
            }
            if (same) cout << "multiple S/L test : pass." << endl;
            else cout << "multiple S/L test : fail." << endl;
        }
    }

	return 0;
}