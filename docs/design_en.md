English | [中文](design.md)

## PSuM Framework Design

### Design Goals and Audience

This document is intended for framework maintainers and contributors. It explains PSuM's internal organization, data structures, and implementation mechanisms. If you are a user wanting to learn how to write PSuM programs, please read the [Programming Guide](programming_en.md).

Framework design always serves the user — never forget the beginner's mind!

### Project File Layout

PSuM's source organization follows these basic principles: **separate user entry points from implementation details, favor header-only as the primary pattern, and keep test directories parallel to source modules**.

The most important directories in the repository are:

- `include/psum/`: Public entry points. User programs typically include headers from here, e.g. `#include <psum/psum.hpp>`. These files aggregate modules and should not carry heavy implementation logic.
- `src/`: Main framework implementation. Subdirectories generally correspond to modules and namespaces, e.g. `src/field/` for the field module, `src/particle_container/` for the particle container module.
- `test/`: Module tests. It largely mirrors the `src/` structure, e.g. `test/field/` tests `src/field/`, `test/field_solver/` tests `src/field_solver/`.
- `example/`: Tutorial examples. Programs here should be more readable than tests, demonstrating minimal API usage.
- `application/`: Real-world cases and benchmarks. These may include more complete physics models, configuration files, and output workflows.
- `pypsum/`: Python-side support, currently mainly serving serialization-related Python/C++ interop. It should not be understood as a complete Python version of PSuM.
- `docs/`: Project documentation.
- `docker_install/`: Docker installation path, an alternative to local dependency installation.

#### Public Entry Points and Implementation Directories

`include/psum/psum.hpp` is the most commonly used main entry point. It aggregates tag, serialization, particle_container, field, particle_boundary,
field_solver, particle_collision, random, timer, utils_sycl and other modules, and defines `psum::prelude`:

```cpp
#include <psum/psum.hpp>
using namespace psum::prelude;
```

Module-level entry points include:

- `include/psum/tag.hpp`
- `include/psum/serialization.hpp`
- `include/psum/particle_container.hpp`
- `include/psum/field.hpp`
- `include/psum/field_solver.hpp`
- `include/psum/particle_boundary.hpp`
- `include/psum/particle_collision.hpp`

At present, PSuM is a framework library that provides functionality in source form rather than as binaries: consumers compile the full source tree into their own build, where the `include/psum/` public entry points and the `src/` implementations together form the compiled source set.
Many implementations in the current project exist in header-only form, so public entry points ultimately include implementation headers from `src/`.
From a design perspective, however, `include/psum/` remains the user-facing boundary; application code should prefer depending on public entry points rather than treating `src/` directory structure as a stable API.

When adding a new module, it is recommended to first organize the implementation in `src/<module>/`, then decide whether to expose it via `include/psum/<module>.hpp`.
This prevents internal file structure from becoming an external commitment prematurely.

#### Test Directory and Module Correspondence

The `test/` directory serves both verification and documentation of behavioral contracts. It is not a tutorial directory.

Common correspondences:

- `test/field/`: Validates `simple_grid`, `host_field`, `device_field`, interpolation, and deposition.
- `test/field_solver/`: Validates Poisson solver, boundary conditions, ghost handling, and backend solving.
- `test/particle_boundary/`: Validates STL loading, geometry intersection, spatial partitioning, boundary router, and material behaviors.
- `test/particle_collision/`: Validates MCC/DPMCC, pairing engine, and collision-related utilities.
- `test/pypsum/`: Validates Python/C++ serialization interop.

The batch verification entry point is `test/runCheck.sh`, which compiles and runs all executable targets with a short timeout (smoke tests).

If you add a new `src/<module>/`, you should also consider adding or extending `test/<module>/`.
Example programs should go in `example/` or `application/` — do not maintain test code as user tutorials.
Additionally, test code generally does not use `#include <psum/psum.hpp>`, as it introduces dependencies beyond the scope of what is being tested.


### Major Modules in PSuM

```
┌──────────────────────────────────────────────────────────────┐
│   PIC algorithm modules                                      │
│  ┌───────────────┐ ┌────────────────────┐ ┌────────────────┐ │
│  │    particle   │ │      particle      │ │     field      │ │
│  │    boundary   │ │      collision     │ │     solver     │ │
│  └───────────────┘ └────────────────────┘ └────────────────┘ │
│     ┌───────────────────┐         ┌───────────────────┐      │
│     │ particle_container│         │       field       │      │
├─────┤ (particle_group,  │─────────┤  (device_field,   ├──────┤
│     │  device_vector)   │         │   device_array)   │      │
│     └───────────────────┘         └───────────────────┘      │
│  data structure layer                                        │
├──────────────────────────────────────────────────────────────┤
│  meta-programming layer                                      │
│               ┌──────────┐ ┌───────────────┐                 │
│               │ tag      │ │ serialization │                 │
│               └──────────┘ └───────────────┘                 │
└──────────────────────────────────────────────────────────────┘

```
- **Base meta-programming layer** — The `tag` system and `serialization` system do not directly participate in PIC computation, but provide compile-time information and persistence capabilities to upper layers.
Almost all modules depend on tag.
- **Container layer** — `device_vector` / `device_array` / `particle_group` manage data on the device side.
They do not implement PIC algorithms themselves; they are high-performance data structures.
- **PIC module layer** — Five core modules: "3P2F". `particle_container` and `field` belong to both the container layer and the PIC module layer (they provide both data structures and algorithm interfaces);
`particle_boundary` handles particle-geometry boundary interactions; `particle_collision` handles inter-particle collisions; `field_solver` solves the Poisson equation.
Beyond these, there are also some smaller modules and utilities:
- **Utility layer** — `random` (random number generation and physical distributions), `timer` (performance timing), `ui` (keyboard input), `utils_sycl` (SYCL utility functions). These do not depend on upper-layer modules.

File organization follows the convention **directory = namespace, filename = sub-namespace or main type**.
For example, `src/field/device_field.hpp` corresponds to `psum::field::device_field`, and `src/tag/foundation.hpp` corresponds to `psum::tag::foundation` (which contains multiple utility types).

### Tag System Implementation

PSuM is called a meta-programming framework precisely because of the tag system.

**Type hierarchy:**
```
abstract_tag                 ← Base class for tags
    ├── position             ← Built-in property tag
    ├── velocity             ← Built-in property tag
    ├──   ...                ← Built-in property tags
    └── User-defined tags

concrete_tag                 ← Base class for bound instances
    └── tag_bind<Tag, Type>  ← Tag + type pair
```

**Core concepts:**
- `a_tag_concept<T>` — Requires `T` to inherit `abstract_tag` and have `static std::string tag_name`
- `c_tag_concept<T>` — Requires `T` to inherit `concrete_tag`

**How tagged_struct works:**

`tagged_struct<tag_bind<T1,V1>, tag_bind<T2,V2>, ...>` expands to a class that inherits `std::tuple<V1, V2, ...>`. The workflow for `get<Tag>(obj)` is:

1. `tag_bind` exposes both `front` (tag type) and `back` (data type) aliases
2. `dual_tuple` uses `front`/`back` to separate these pairs into two tuples: `tuple_of_tag` (tag set) and `tuple_of_type` (type set)
3. `index_in_tuple<Tag, tuple_of_tag>` finds the tag's index position at compile time via recursive templates
4. `get<Tag>(obj)` obtains that index → calls `std::get<index>(tuple_base)` → returns a reference to the corresponding field

This entire process is completed at compile time. The generated code is equivalent to `obj.field_N` with zero runtime overhead.

**Tag validation:** `try_check<Tag, Type>()` is called within `tag_bind`'s `static_assert`.
It is triggered at compile time when the user writes `tagged_struct<tag_bind<Tag, Type>, ...>`.
If the tag carries a `check<Type>()` and the check fails, compilation fails here.
This ensures that unreasonable type bindings are caught at compile time.

### Particle Container Implementation

#### device_vector Memory Model

`device_vector<T>` uses SYCL USM to allocate memory on the device:

- **Construction**: Calls `sycl::malloc_device` to allocate raw device memory; calls `sycl::free` on destruction
- **Capacity management**: Maintains `data_` (raw pointer), `size_` (current element count), `capacity_` (total capacity), `overflow_` (overflow flag)
- **Thread-safe push_back**: `device_vector_acc`'s `push_back` uses `sycl::atomic_ref<size_t>` to atomically increment `size_`.
After obtaining a unique write position, it copies the data directly. Multiple threads can push simultaneously without conflict, but capacity cannot grow automatically — when capacity is exceeded, the `overflow_` flag is set.
Host-side operations (`size()`, `data()`, `to_host()`, `copy()`, etc.) all check this flag first and throw an exception, ensuring overflow errors are raised *as early as possible*.

**Double-lambda mechanism:**

```cpp
vec.for_each([&](handler& h) {
    auto acc = get_access(h);  // Package pointer+size as lightweight value
    return [=](T& elem) {
        elem = ...;
    };
});
```

- The outer lambda is converted to a SYCL host-side `h.parallel_for`, used to submit work
- The inner lambda must capture by value with `[=]`, because the device side cannot access host-side variables by reference
- `acc` (containing raw pointer + size) must be **trivially copyable** so it can be safely transferred to the device

#### particle_group Dual-Array Design

`particle_group` maintains two `device_vector`s:

- `data_` — Stores all particles (valid + invalid)
- `invalid_indexes_` — Stores indexes of invalid particles

**size() = data_.size() - invalid_indexes_.size()** — Number of valid particles

`device_vector` is the only container that supports device-side `push_back` (via `device_vector_acc` atomic operations). `particle_group` can only batch-add particles on the host side via `insert()`.

**Three insert strategies:**
1. No invalid slots (`invalid_indexes_.size() == 0`): `data_.resize()` to append → `memcpy` to the end
2. Enough invalid slots (`insert_size <= invalid_indexes_.size()`): Pop the corresponding number of indexes from the stack, copy new particles to those positions
3. Some invalid slots but not enough: Fill invalid slots first, then append the remainder

**Lazy deletion:** When `validator::make_invalid(p)` is called inside `for_each`, the current particle's index is pushed to `invalid_indexes_`.
After traversal, the positions of invalid particles can be reused by subsequent `insert`, but physical memory is not changed. The validator also guarantees that subsequent `for_each` calls will not access this particle again.

**compress algorithm:** Scans from back to front, moving valid particles from the end into earlier invalid positions, then truncates the array.

**shuffle algorithm:** Uses a Feistel network approach for rearrangement, achieving high performance.

### Field Implementation

#### simple_grid

`simple_grid<Dim>` describes a regular Cartesian grid. Its core internal data:
- `lower_bounds` / `upper_bounds` — Start and end coordinates, `std::array<double, Dim>`
- `cell_num` — Number of cells per dimension, `std::array<int, Dim>`

It also stores `delta` (cell edge length = 1/cell_num), `delta_r` (reciprocal), `length` (total length).

**Flat indexing** maps multi-dimensional grid indexes to a one-dimensional linear array:
- Total nodes = ∏(cell_num[i] + 1)
- Total cells = ∏(cell_num[i])
- `n2i(n)` / `c2i(c)` convert node/cell indexes to linear indexes (for array storage)
- `i2n(i)` / `i2c(i)` perform the reverse mapping

#### device_field and Accessors

`device_field<Dim, Location, Scalar, QSize>` internally holds:
- `device_array<Value>` — Device-side linear storage of field values
- `simple_grid<Dim>` — Grid

The field itself is **not directly accessible** — you must use an accessor:

```cpp
decltype(auto) get_access(sycl::handler& h) {
    return acc_type(device_content.get_access(h), grid);
}
```

The accessor `device_field_acc` wraps `device_array_acc` + `simple_grid` and is a trivially copyable value type. Kernel functions read and write field values via `acc(idx)`, and access the grid via `acc.getGrid()`.

The benefit of this design: the field object (`device_field`) can remain on the host side, while the accessor is copied to the device for interpolation and deposition — avoiding the need for the device to hold complex field objects.

#### Interpolation and Deposition

Users can use multiple interpolation/deposition-related functions in kernel code, all accepting a position and a field accessor:

- **`interp(pos, field_acc)`** — Linearly interpolates field values at the particle position (node fields only)
- **`interp_diff(pos, field_acc)`** — Interpolates field gradients, returning `Dim` sets of values (node fields only), one for each component's gradient
- **`add_back(pos, weight, field_acc)`** — Deposits particle weights onto the grid (atomic add, node fields only)
- **`interp_nearest(pos, field_acc)`** / **`add_back_nearest(pos, weight, field_acc)`** — Nearest-neighbor variants (available for both node and cell fields)

The `interp`, `interp_diff`, `add_back` functions are internally unified through `interp_tools::generate_interp` / `generate_add_back` etc., with template parameters specifying the concrete 1D interpolation kernel.

**Interpolation core `interp_tensor`:** A tensor product framework that extends 1D interpolation functions to N dimensions.

The 1D linear interpolation function `linear_interp_1D(x)` returns `{1-x, x}` (two coefficients, for left and right nodes).
`interp_tensor` independently calls the 1D function for each dimension and takes the Cartesian product of the results, producing `2^Dim` support points and their coefficients.

For example, in 2D, the support points are 4 corner nodes with coefficients `(1-x)(1-y), (1-x)y, x(1-y), xy`.

**Gather (`generate_interp`):**
```
For each support point i:
    res += field_acc(index[i]) * coeff[i]
```

**Scatter (`generate_add_back`):**
```
For each support point i:
    atomic_add(field_acc(index[i]), value * coeff[i])
```

The atomic add is wrapped in `_atomic_add_`, which internally uses `sycl::atomic_ref`'s `fetch_add`.

**Gradient interpolation `generate_interp_diff`:** For each dimension, returns a separate set of coefficients (tensor product of the 1D interpolation kernel derivative with the normal interpolation kernel for that dimension), producing `Dim` sets of results.

### Field Solver Implementation

The field solver's responsibility is to organize the grid, boundary conditions, and source terms into a linear system of equations, leaving "how to solve the linear system" to the backend.
PSuM primarily solves the Poisson equation common in electrostatic PIC. The interface is split by dimension into `Poisson_solver_1d`, `Poisson_solver_2d`, `Poisson_solver_3d`.

The solver has two internal layers:
- **Frontend**: Composed of C++ templates and Eigen sparse matrices, responsible for discretization, boundary condition assembly, right-hand side correction, and interfacing with PSuM's `field` data structures.
- **Backend**: Encapsulated by `solver_backend`, responsible for concrete linear equation solving. The default backend is `native`, which uses Eigen; CUDA or other solver implementations can also be plugged in through a unified function table.

#### Discretization and Matrix Assembly

`Poisson_solver_2d` and `Poisson_solver_3d` use finite difference discretization on regular grids.
Taking the 2D Cartesian case as an example, the unknown quantity at grid nodes is the electric potential `phi`. The discrete equation for interior nodes is approximately:

```text
(phi_E - 2 phi_C + phi_W) / dx^2
+ (phi_N - 2 phi_C + phi_S) / dy^2
= -rho / epsilon
// Here epsilon non-uniformity is ignored. In Poisson_solver_2d, epsilon can have a distribution.
```

During initialization, a sparse linear system is formed with `Eigen::SparseMatrix<double>` as the coefficient matrix.
This way the solver only needs to handle a standard sparse matrix, with different backends performing the solve.

The 2D solver also supports both `Cartesian` and `Cylindrical` coordinate forms.
In cylindrical coordinates, the radial term carries geometric coefficients, so matrix assembly adjusts neighbor node coefficients based on node position; externally, however, the user still simply solves `phi` on a 2D grid.

#### Fixed and Mixed Boundaries

PSuM splits boundary conditions into two categories:

- **Fixed boundaries**: e.g. Dirichlet conditions, directly specifying values at certain nodes. In implementation, the corresponding matrix row is replaced with `phi_i = value`, and the boundary value is written into the right-hand side.
- **Mixed boundaries**: e.g. Robin conditions, which formally involve both function values and normal derivatives. These cannot simply replace matrix rows; ghost nodes are introduced to express virtual values outside the boundary.

Fixed boundaries take priority over mixed boundaries.

#### ghost_element_manager

The core tool for mixed boundaries is `ghost_element_manager`.
It organizes the main system variables `x` and ghost variables `x_ghost` into a block linear system:

```text
[A, B] [x      ] = [b             ]
[C, D] [x_ghost]   [boundary_value]
```

Where:
- `A` represents the main grid node equations.
- `B` represents the contribution of ghost nodes to the main equations.
- `C, D` come from the boundary equations, expressing the relationship between ghost nodes and main nodes.

`ghost_element_manager::get_elimination_system()` eliminates the ghost variables, yielding a system containing only main variables:

```text
A' x = b + M * boundary_value
A' = A - B * D^-1 * C
M  = -B * D^-1
```

This is the origin of `A_` and `M_` in `Poisson_solver_*`.
`A_` is handed to the solver backend; `M_` participates in right-hand side correction before each `solve()`, since Robin boundary values may change over time.

#### solver_backend Plugin System

`solver_backend` is PSuM's unified interface class for multiple solver backends.
It internally stores two objects:

- `psum_field_solver_handle handle`: The backend solver instance.
- `func_table funcs`: A set of C-style function pointers, including `init`, `set_matrix`, `solve`, `set_options`, `set_source_replace`, `set_source_addback`, `free`.

When constructing `solver_backend(kind)`, the frontend obtains the function table via `get_solver_funcs(kind)` and calls `init()` to create the backend instance.
On destruction, it calls `free(handle)` to release resources.
This allows backends to be implemented with different technologies, as long as they conform to the same C ABI interface.

The solving flow is:

```text
1. Construct the right-hand side from input source terms
2. The native backend calls Eigen to solve; other backends call backend.solve(source, phi)
```

For the complete interface specification, resource management conventions, and implementation details of the backend plugin system, please refer to the [Field Solver Backend Design Document](detailed_design/field_solver_backend_en.md).

### Particle Boundary System Implementation

The particle boundary module handles "whether a particle crosses a geometric surface when moving from its old position to its new position, and what should happen after the crossing."
It consists of two independent parts:

- **trigger**: Detects whether a trajectory intersects with triangular facets, returning the nearest intersection point, normal, and material type.
- **material_set**: Modifies the particle based on material type, e.g. absorption, specular reflection, diffuse reflection, or velocity resampling based on wall temperature.

#### boundary_router

`boundary_router<MaterialSet, ParticleType>` is the user-facing entry point.
It internally holds:

- `triangle_mesh_trigger<material_type>`: Geometry trigger.
- `MaterialSet`: Material behavior collection.

In kernel code, the user obtains a `boundary_router_acc` via `router.get_access(h)`, then calls `router_acc.deal(old_pos, new_pos, p)` to process particle boundary interactions.

The `deal()` execution flow is:

```text
1. trigger_acc.detect(...) checks for intersection between the line segment and mesh surface
2. If no intersection, return false
3. If intersection, compute hit point hit = p1 + k * (p2 - p1) based on parameter k
4. Pass material type, hit point, and normal to material_set_acc.action(...)
5. material_set modifies the particle and returns true
```

Note that the old and new positions are passed in, rather than a single particle position.
This prevents missed detections when a particle crosses a thin wall in one step.

#### Spatial Partitioning and Acceleration

Checking every particle trajectory against every triangle directly would be very slow.
`triangle_mesh_trigger` therefore places the geometry into a regular spatial partition.
When the user calls `boundary_router::set(meshes, I, J, K)`, `I/J/K` specify the number of 3D partitions.

Spatial acceleration relies on two main steps:

1. **Triangle pre-assignment**: Triangles are placed into potentially intersecting structured grid spatial partitions. This is done outside the `deal` function, during trigger object configuration.
2. **Line segment traversal through partitions**: The particle trajectory from old to new position forms a line segment. `trace_to_idxs_begin` / `trace_to_idxs_next` sequentially traverse the partitions this line segment passes through.

Line segment traversal is similar to the DDA grid traversal algorithm.
It maintains the current partition, direction of motion, and the parametric distance `t` to the next grid face, crossing the nearest grid face each time and returning the new partition index.
This way, a particle trajectory only needs to check the few partitions it actually passes through.

The actual line segment-triangle intersection test is performed by `line_seg_tri_intersect_test`, using the Moller-Trumbore algorithm.

If a trajectory hits multiple triangles, the trigger selects the nearest intersection point. Additionally, some performance tuning options are available, such as enabling an unsigned distance field.

#### Material Behavior Dispatch

Material behaviors are dispatched via the `material_type` enum in `basic_material_set::acc_type::action()`.

Writing material sets is straightforward. PSuM provides a simple `basic_material_set` with four built-in material behaviors:

- `absorb`: Calls `Validator<ParticleType>::make_invalid(p)`, marking the particle as invalid.
- `specular`: Specular reflection, velocity updated to `v' = v - 2 (v dot n) n`.
- `diffuse`: Samples a new reflection direction from a cosine distribution, keeping speed approximately unchanged.
- `maxwell_reflect`: Samples a new velocity from a half-space Maxwell distribution based on wall temperature and particle mass.


### Collision System Implementation

PSuM's collision module is divided into two categories:

- **MCC**: Monte Carlo Collision, handling collisions between particles and background species, e.g. electron-neutral gas collisions.
- **DPMCC**: Deferred Pairing MCC, handling particle-particle collisions that require explicit pairing, e.g. molecular pair collisions in DSMC.

Both are designed around the principle of "users provide collision models, and the framework handles parallel traversal, context access, and new particle buffering."

#### mcc_model and Contexts

`mcc_model<SubModels...>` itself is primarily a type container.
Each `mcc_submodel_for_incident<IncidentTag, CollisionSet>` represents an incident particle species and its possible collision set.

`mcc_species_context<ParticleContainer, Field>` binds a particle container and a density field:

- `group_ref_getter` returns the real particle group.
- `field_acc_getter` generates a density field accessor when submitting kernel functions.
- `buffer` temporarily stores new particles produced by collisions.

Multiple species contexts are combined via `mcc_module_context<tag_bind<Tag, Context>...>`.
It follows PSuM's tag system, enabling kernel code to access a specific species' density field and new particle buffer via `ctx_acc.get<SpeciesTag>()`.

At runtime, `execute_mcc_model<Model>(ctx, dt)` expands submodels and calls `execute_single_collision` for each incident species.
The framework provides template metaprogramming tools for iterating over provided type tuples, allowing declarative model definitions.

The flow of a single MCC execution is:

```text
1. Execute for_each on the incident species' particle_group
2. In the kernel, interpolate background density and compute collision probability
3. Randomly select a specific collision channel
4. Modify the current particle, or push_back new particles to a species buffer
5. After all collisions finish, execute_all_clean_buffers batch-inserts buffer contents back into particle_group
```

The buffer's existence is important.
You cannot directly insert new particles into `particle_group` on the device side.
`device_vector` supports device-side `push_back` and can serve as a temporary buffer.

#### Deferred Pairing and pairing_engine

Some collisions cannot be handled by examining just one particle and a background field — a second real particle must be found as the collision partner.
DPMCC handles this class of problems through "deferred pairing."

A deferred pairing collision type inherits from `base_of_deferred_pairing_collision`.
`standard_dpmccm_context` filters these types from all collision channels in the `mcc_model` and prepares a buffer for each deferred collision type.

The execution flow is:

```text
1. During the normal MCC phase, if an incident particle triggers a deferred collision, write {position, particle pointer, null pointer} to the DPMCC buffer
2. pairing_engine maps these requests to partitions based on the spatial grid
3. Traverse background particles, claim requests in each partition, fill in the second particle pointer
4. clean_buffer traverses the buffer again, calling Col::pair_collide(p1, p2, ctx_acc) for successfully paired particles
```

`pairing_engine` internally maintains particle counts `n_counts_` and ticket counts `ticket_counters_` per grid cell.
`oversample_factor` controls the probability of background particles participating in pairing, preventing complete failure to pair when request counts are low, and avoiding excessive scanning of all particles when request counts are high.

This module also provides many metaprogramming tools.

### Utility: Random Numbers

The random number module provides reproducible pseudo-random sequences for device-side computation. In PSuM, better reproducibility is achieved by assigning a random seed attribute to each particle.

#### rander and Device-Side Constraints

`rander` is a device-side random number generator using the LCG (Linear Congruential Generator) algorithm:

```cpp
x = x * a + c;
```

Since SYCL device code cannot use standard library engines like `std::mt19937` (they depend on dynamic memory allocation and system calls), PSuM implements a lightweight `rander_core` template. It depends only on basic integer arithmetic, satisfying device-side execution constraints.

#### Seed Writeback Mechanism

Particle random seeds are commonly represented by the `property::random_seed` attribute. The typical pattern for using random numbers is:

```cpp
particles.for_each([&](sycl::handler& h) {
    return [=](Particle& p) {
        auto R = random::view_as_rander(get<property::random_seed>(p));
        // Use R() to generate random numbers
        // R's destructor automatically writes the updated seed back to p
    };
});
```

`view_as_rander` generates a `rander_wrapper`, which calls `sync_seed()` in its destructor to write the engine's internal state back to the particle's `random_seed` field.
This ensures:
- Each run produces the same sequence from the same initial seed (reproducibility)
- Different particles have independent random sequences (because initial seeds differ)
- Seed state is persistently stored with the particle

#### Physical Distribution Sampling

`rand_functions.hpp` provides commonly used physical distribution sampling functions:

- `RandV_Maxwell(r, T, m)`: Maxwell velocity distribution
- `RandV_spherical(r)`: Uniform spherical direction
- `RandV_Cosine(r, axis)`: Cosine distribution reflection direction
- `RandV_halfMaxw(r, axis, T, m)`: Half-space Maxwell distribution

These functions accept a `rander&` reference and are implemented in separate 1D/2D/3D namespaces.

### Serialization System

The serialization module is responsible for saving PSuM state to files and restoring it in subsequent runs.
It handles two categories of tasks:

- Saving basic arrays, Eigen objects, STL containers, tagged_structs, and device containers to MAS files.
- Managing simulation state and configuration parameters via `object_manager` and `json_loader`.

#### MAS Binary Format

MAS stands for Multi-Array-Set.
A MAS file consists of multiple data blocks appended sequentially. Each data block contains:

```text
magic string: #MAS!#
key length / type length / shape length
key string
type string
shape array
data bytes
```

`mas_file::readHead()` scans all blocks from the beginning to the end of the file, reading only metadata and recording each block's position in the file.
Actual data is read later by seeking to the block position.
This avoids loading all large arrays into memory when opening a file.

MAS does not allow duplicate data blocks with the same name by default.
If data needs to be replaced, `replaceData()` calls `smashData()` first.
`smashData()` does not move file contents — instead, it changes the first character of the old block's key to `!`, causing subsequent scans to ignore it.
This design makes replacement operations approximately O(1), at the cost of potentially leaving masked old data blocks in the file.

#### Generic save/load

The serialization interface centers on overloaded `save(fp, key, obj)` and `load(fp, key, obj)` functions.
Support for different types is distributed across several files:

- `foundation.hpp`: Basic types and type-erased pointers.
- `container_sl.hpp`: `std::vector`, `device_vector`, and other containers.
- `tag_sl.hpp`: Tag-based serialization of `tagged_struct`.
- `object_manager.hpp`: Object-level management and batch saving.

`tagged_struct` serialization particularly depends on the tag system.
Fields are written by tag name during saving and matched by tag name during reading, rather than relying on field order within the tuple.
Data corresponding to different fields can be found quite easily in the mas_file.

When serializing `device_vector` and `particle_group`, device-side data is copied back to the host side before being written to file.
The documentation and interface both emphasize that device container data across different hardware should not be assumed to be unconditionally interchangeable, as device-related information is also recorded in the file.

#### object_manager

`object_manager` is designed for "checkpoint restart" and "unified simulation state management".
It internally maintains:

- `objects_`: `std::unordered_map<std::string, std::any>`, storing registered objects.
- `savers_`: The save function for each object.
- `file_`: An optional existing MAS file for restoring initial state.

Users obtain object references via `mng.obj<T>(name)`.
If the object is already in memory, the existing object is returned; if not, a default-constructed object is created and an attempt is made to read data with the same name and type from the associated MAS file.
When saving, `object_manager::save()` iterates through all registered savers and writes objects to a new MAS file.

To avoid overwriting the checkpoint file being read, `object_manager` checks the save path.
If the current manager is associated with an existing file, the save target must be a different valid path.
This reduces the risk of "accidentally overwriting old state while reading it."

#### json_loader

`json_loader` inherits from `object_manager` and converts JSON configuration into a queryable parameter table.
`key_condition` provides prefix-based querying, allowing users to write access logic like `loader["solver"].obj<double>("dt")`.

`json_loader` only handles configuration reading, not arbitrary JSON object modeling.
