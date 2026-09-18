English | [中文](telescope.md)

## Telescope

This document demonstrates a 2D geometric ray tracing example.

The example simulates a simplified Maksutov telescope: parallel light enters from the left, passes through a corrector (refraction), primary mirror (reflection), and secondary mirror (reflection), then converges onto the focal plane.
The geometric parameters are largely based on the Gregory–Maksutov Telescope example from COMSOL.

![2D Geometric Optics Simulation](density_step.gif)

Modules, features, and typical usage covered:
- Particle definition and particle containers
- Particle boundary setup

### Preparation

Create `telescope.cpp` under `example/telescope/`, then modify this file step by step following each stage.
`example/telescope/stage_1.cpp` through `stage_5.cpp` are reference solutions at the end of each stage, for comparison with your own code; they are not meant to be run as standalone programs.

It is recommended to enter the example directory from the project root and load the build environment:

```bash
cd example/telescope
source ../../env_load.sh
```

There may be leftover `.plt`, `.png`, or `.gif` files in `output/`.
You can manually clear `output/` first.

### Stage 1: Creating Photons and Basic Motion Loop

In `telescope.cpp`, first include the necessary headers and set up the namespace:

```cpp
#include <cmath>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;
using namespace Eigen;
```

Add the photon type definition:

```cpp
using Photon = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,
    tag_bind<property::velocity, Eigen::RowVector2d>
>;

using PhotonGroup = particle_group<Photon, pos_x_nan_is_invalid>;
```

Photons need position and velocity (direction); `pos_x_nan_is_invalid` means that once a particle is marked invalid, `particle_group` iteration will automatically skip it.

Add the main function:

```cpp
int main() {
    sycl::queue q{sycl::default_selector_v};
    cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    grid2D grid({-0.11, -0.18}, {0.7, 0.18}, {540, 240});
    node_field2D<double> density(q, grid);
    PhotonGroup photons(q);
    return 0;
}
```

Add the photon emission function:

```cpp
vector<Photon> make_photons(int count, int num_way, double ds, double width, random::rander& R) {
    vector<Photon> photons(count);
    for (int i = 0; i < count; ++i) {
        double y = -0.095 + 0.190 * (int(R()*num_way) + 0.5) / num_way;
        y += width * (R() - 0.5);
        RowVector2d dir(1.0, 0.0);

        get<property::velocity>(photons[i]) = dir;
        get<property::position>(photons[i]) = RowVector2d(-0.095, y) + dir * (ds * R());
    }
    return photons;
}
```

Here `num_way` controls the number of light rays, `width` controls the random offset of each ray, and `ds` controls the random perturbation of the emission position.

In the main function, create a simple motion loop and plot output:

```cpp
    // ...
    double ds = 0.002;
    int steps = 500, pps = 1000, interval = 30;
    random::rander R;
    for (int step = 0; step < steps; ++step) {
        photons.insert(make_photons(pps, 20, ds, grid.span<1>() * 0.01, R));
        photons.for_each([&](sycl::handler& h) {
            return [=](Photon &p) {
                get<property::position>(p) += get<property::velocity>(p) * ds;
            };
        });
        if (step % interval == 0) {
            density.setZero();
            photons.for_each([&](sycl::handler& h) {
                auto da = density.get_access(h);
                return [=](Photon& p) {
                    if (da.getGrid().inGrid(get<property::position>(p)))
                        add_back(get<property::position>(p), 1.0, da);
                    else PhotonGroup::validator::make_invalid(p);
                };
            });
            density.plot("output/density_step_" + to_string(step) + ".plt", "density", step);
            cout << "step = " << step << ", alive = " << photons.size() << endl;
        }
    }
    cout << "Ray tracing 2D: wrote photon density snapshots." << endl;
```

Here `ds` is the optical path step size, acting like a time step. Every `interval` steps, a 2D density snapshot is output for observing the motion trajectory.

Create the makefile:

```makefile
-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

telescope: telescope.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o telescope telescope.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./telescope
	python3 ../pltview.py density_step output
```

(These commands already exist in example/telescope/makefile, just uncomment them.)

For each subsequent stage, compile and run the same way:

```bash
make telescope
```

After a successful run, you should see `output/density_step.gif`, showing photons propagating from left to right and disappearing at the boundary.

Key takeaways for this stage:
- `Photon` defines which attributes a photon has.
- `make_photons` generates photons on the host side and inserts them into the particle container.
- `for_each` iterates over all photons and updates their positions.
- Photons leaving the grid are marked invalid by `make_invalid`.

The code state at the end of this stage can be found in `example/telescope/stage_1.cpp`.

### Stage 2: Adding the Boundary System

Continue modifying `telescope.cpp` from Stage 1:
- Add the `OpticalMaterialSet` class.
- Add the `boundary_router`.
- Create the router in `main` and call it within the motion loop.

First, define the optical material set class:

```cpp
class OpticalMaterialSet {
public:
    enum class material_type { absorb };

    class acc_type {
    public:
        void action(material_type mat, Photon& p, double hit_x, double hit_y, double, double nx, double ny, double) const {
            if (mat == material_type::absorb) {
                PhotonGroup::validator::make_invalid(p);
            }
        }
    };

    acc_type get_access(sycl::handler&) const { return acc_type(); }
};

using Router = boundary_router<OpticalMaterialSet, Photon>;
using material_type = OpticalMaterialSet::material_type;
```

`OpticalMaterialSet` defines the behavior of optical materials. Currently there is only the `absorb` type, which absorbs photons.
`acc_type` is the access type used on the device side; the `action` function handles photon-boundary interactions.

In main, create a router and set a rectangular boundary (using the grid boundary):

```cpp
Router router(q);
auto plane_1 = mesh_generator::extrude_xy_line_to_width({{0.0, 0.0}, {0.2, 0.2}}, 0.2);
router.set({{plane_1, material_type::absorb}}, 160, 80, 20);
```

Even though this is a 2D computational domain, the geometry input in the boundary module must be 3D. Therefore, `mesh_generator::extrude_xy_line_to_width` is used to extrude a 2D line segment into a 3D surface.
In this stage we only add one line segment: from (0, 0) to (0.2, 0.2). Any photon hitting this segment will be absorbed.

Modify the motion loop to call the router after moving:

```cpp
photons.for_each([&](sycl::handler& h) {
    auto router_acc = router.get_access(h);
    return [=](Photon &p) {
        RowVector2d old_p = get<property::position>(p);
        get<property::position>(p) += get<property::velocity>(p) * ds;
        RowVector2d new_p = get<property::position>(p);
        router_acc.deal(old_p.x(), old_p.y(), 0.0, new_p.x(), new_p.y(), 0.0, p);
    };
});
```

`router_acc.deal` receives the old and new positions, checks whether a boundary was crossed, and invokes the corresponding material behavior.

After a successful run, you should see `output/density_step.gif`, with photons being absorbed when they reach the defined line segment.

Key takeaways for this stage:
- `OpticalMaterialSet` defines material behavior (currently only absorption).
- `boundary_router` manages the mapping between boundaries and materials.
- `router_acc.deal` handles photon-boundary interactions.

The code state at the end of this stage can be found in `example/telescope/stage_2.cpp`.

### Stage 3: Adding Curved Boundaries

Continue modifying `telescope.cpp` from Stage 2:
- Add the `arc_surface` function.
- Use `mesh_generator::extrude_xy_line_to_width` to create curved surfaces.
- Set up a curved absorption boundary.

Add the curved surface generation function:

```cpp
vector<pair<double, double>> arc_surface(RowVector2d top, RowVector2d curv_vector, double half_angle, bool norm_out = true) {
    vector<pair<double, double>> pts;
    int seg = 80;
    RowVector2d center = top - curv_vector;
    double radius = curv_vector.norm();
    double base_angle = atan2(curv_vector.y(), curv_vector.x());
    for (int i = 0; i <= seg; ++i) {
        double t = -1.0 + 2.0 * i / seg;
        if (norm_out == false) t = -1.0 + 2.0 * (seg - i) / seg;
        double angle = base_angle + t * half_angle;
        pts.push_back({center.x() + radius * cos(angle), center.y() + radius * sin(angle)});
    }
    return pts;
}
```

This is a simple geometry function that generates a circular arc with `top` as the arc apex and `top - curv_vector` as the center. `norm_out` controls the direction of the normal.

Remove the original plane_1, and create a simple curved absorption boundary in main:

```cpp
auto barrier = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.3, 0.0}, {0.3, 0.0}, 0.3), 0.2);
router.set({{barrier, material_type::absorb}}, 160, 80, 20);
```

This creates a curved barrier.

After a successful run, you should see `output/density_step.gif`, with photons being blocked by the curved barrier.

Key takeaways for this stage:
- `arc_surface` generates a point sequence for curved surfaces.
- `mesh_generator::extrude_xy_line_to_width` extrudes a line segment into a mesh with width.

The code state at the end of this stage can be found in `example/telescope/stage_3.cpp`.

### Stage 4: Adding Reflection

Continue modifying `telescope.cpp` from Stage 3:
- Add `specular` type to `OpticalMaterialSet`.
- Implement specular reflection logic.
- Set up the secondary and primary mirrors.

Add `specular` to the `material_type` enum:

```cpp
enum class material_type { absorb, specular };
```

Add specular reflection logic in the `action` function:

```cpp
void action(material_type mat, Photon& p, double hit_x, double hit_y, double, double nx, double ny, double) const {
    RowVector2d hit(hit_x, hit_y);
    RowVector2d n(nx, ny);
    n.normalize();

    RowVector2d dir = get<property::velocity>(p).normalized();
    double speed = get<property::velocity>(p).norm();
    if (mat == material_type::absorb) {
        PhotonGroup::validator::make_invalid(p);
    } else if (mat == material_type::specular) {
        n = dir.dot(n) > 0.0 ? -n : n;
        RowVector2d out = dir - 2.0 * dir.dot(n) * n;
        out.normalize();
        get<property::velocity>(p) = out * speed;
        get<property::position>(p) = hit + n * 1e-6;
    }
}
```

Specular reflection formula: `out = dir - 2 * (dir · n) * n`, i.e., the incident direction minus twice the normal component. This formula works in both 2D and 3D.

Remove the original barrier, and create curved surfaces for the secondary and primary mirrors:

```cpp
auto secondary = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.0301, 0.0}, {0.2861193, 0.0}, 0.08), 0.2);
auto primary_up = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.4808846, 0.0693117}, {1.1094470, 0.0693117}, 0.035), 0.2);
auto primary_down = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.4808846, -0.0693117}, {1.1094470, -0.0693117}, 0.035), 0.2);

router.set({{secondary, material_type::specular},
            {primary_up, material_type::specular},
            {primary_down, material_type::specular}}, 160, 80, 20);
```

Additionally, since the optical path is longer, increase the iteration steps from `steps = 500` to 1200.

Here `primary_up` and `primary_down` are used to split the arc into two parts; in a real optical system, a hole would be cut here.

After a successful run, you should see `output/density_step.gif`, with photons first hitting the primary mirror, then the secondary mirror, and being effectively focused.

Key takeaways for this stage:
- Specular reflection formula: `out = dir - 2 * (dir · n) * n`.
- Both the secondary and primary mirrors are concave mirrors, simulated using curved surfaces.

The code state at the end of this stage can be found in `example/telescope/stage_4.cpp`.

### Stage 5: Adding Refraction

Continue modifying `telescope.cpp` from Stage 4:
- Add `refract` type to `OpticalMaterialSet`.
- Add glass refractive index parameter.
- Set up the corrector and central obstruction.

Add `refract` to the `material_type` enum:

```cpp
enum class material_type { absorb, specular, refract };
```

Add glass refractive index parameter to `OpticalMaterialSet`:
```cpp
class OpticalMaterialSet {
    // ... existing code ...
    void set_glass_ior(double v) { ior_ = v; }
private:
    double ior_ = 1.5;
};
```

And adjust the `acc_type` constructor to accept the glass refractive index parameter:
```cpp
class acc_type {
public:
    acc_type(double glass_ior) : glass_ior_(glass_ior) {}
    // ... existing code ...
private:
    double glass_ior_;
};
// ... existing code ...
acc_type get_access(sycl::handler&) const { return acc_type(ior_); }
```

Then add refraction logic in the `action` function:

```cpp
 else if (mat == material_type::refract) {
    double inc_ior = 1.0 / speed;  // ior is reciprocal to speed when c=1
    bool enter = dir.dot(n) < 0.0;
    RowVector2d norm = enter ? n : -n;
    double t_ior = enter ? glass_ior_ : 1.0;
    double eta = inc_ior / t_ior, ci = -dir.dot(norm);
    double sin2 = eta * eta * (1.0 - ci * ci);
    if (sin2 > 1.0) {
        // Total internal reflection (when leaving glass)
        RowVector2d out = dir - 2.0 * dir.dot(norm) * norm;
        out.normalize();
        get<property::velocity>(p) = out * speed;
        get<property::position>(p) = hit + out * 1e-6;
    } else {
        double ct = sycl::sqrt(1.0 - sin2);
        RowVector2d td = eta * dir + (eta * ci - ct) * norm;
        td.normalize();
        get<property::velocity>(p) = td / t_ior;
        get<property::position>(p) = hit + td * 1e-6;
    }
}
```

Snell's law: `n1 * sin(θ1) = n2 * sin(θ2)`, where `n1` and `n2` are the refractive indices of the two media.
When `sin2 > 1`, total internal reflection occurs.

Add new geometry shapes and bind them to the refraction material.
The Maksutov telescope has a meniscus lens ("corrector") in front of the primary mirror.

Create the corrector and central obstruction:

```cpp
auto central_obstruction = mesh_generator::extrude_xy_line_to_width(
    arc_surface({-0.001, 0.0}, {0.2686151, 0.0}, 0.13, false), 0.2);
auto corrector_1 = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.0, 0.0}, {0.2686151, 0.0}, 0.38, false), 0.2);
auto corrector_2 = mesh_generator::extrude_xy_line_to_width(
    arc_surface({0.03, 0.0}, {0.2861193, 0.0}, 0.37, true), 0.2);

router.set({{central_obstruction, material_type::absorb},
            {corrector_1, material_type::refract},
            {corrector_2, material_type::refract},
            {secondary, material_type::specular},
            {primary_up, material_type::specular},
            {primary_down, material_type::specular}}, 160, 80, 20);
router.get_material_set().set_glass_ior(1.45);
```

After a successful run, the final output includes:

```text
output/density_step.gif
```

You should see the complete telescope optical path: light passes through the corrector (refraction), is reflected by the primary mirror, then reflected by the secondary mirror, and converges.
Compared to the previous stage, adding the corrector significantly reduces aberration, visible in the image as a brighter "focal point".

Key takeaways for this stage:
- Snell's law: `n1 * sin(θ1) = n2 * sin(θ2)`.
- Total internal reflection occurs when light travels from a higher refractive index medium to a lower one and the incident angle exceeds the critical angle.
- The corrector corrects spherical aberration; the central obstruction blocks center rays.

The code state at the end of this stage can be found in `example/telescope/stage_5.cpp`.

### Output and Common Errors

`pltview.py` has two common usage patterns:

```bash
python3 ../pltview.py output/density_step_0.plt
python3 ../pltview.py density_step output
```

The first is for a single `.plt` file, typically generating a corresponding `.png`.
The second is for a set of files: it searches for `.plt` files starting with `density_step_` in `output/` and composes an animation.

Common errors can be investigated as follows:

- `make: acpp: No such file or directory`: Usually means `source ../../env_load.sh` was not executed.
- `No rule to make target 'telescope'`: The current directory is wrong, or the `telescope` target has not been added to the makefile yet.
- `No rule to make target 'telescope.cpp'`: The `telescope` target already exists, but `telescope.cpp` has not been created in the current directory yet.
- No `output/...` files generated: Confirm that the makefile contains `mkdir -p output` and that the program is run under `example/telescope/`.
- `pltview.py` does not print `wrote ...`: The plotting script did not complete successfully; check the Python error messages in the terminal above.
- Linker cannot find backend `.o` files: Build the field solver backend first, or check whether the backends enabled in `config.mk.local` match your local environment.
- Seeing `AdaptiveCpp Warning`: This is usually a runtime JIT compilation notice, not necessarily a program error; as long as the program continues to output and generate files, it is fine.
