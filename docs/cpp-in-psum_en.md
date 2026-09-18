English | [中文](cpp-in-psum.md)

## C++ in PSuM

PSuM is written in C++20 and makes use of modern C++ features such as lambdas, templates, move semantics, and concepts. If you are not yet familiar with these features, this chapter will help you quickly grasp the fundamentals you need to know when working with PSuM.

---

### Lambda Expressions

Lambda expressions are C++ syntax for creating anonymous function objects and play a central role in PSuM.

**Basic syntax:**

```cpp
// Simplest lambda: [capture](parameters){ body }
auto f = [](int x) { return x * 2; };
int result = f(3);  // 6
```

**The capture list** determines which external variables the lambda can use:

```cpp
int a = 1, b = 2;
// [=] Capture all external variables by value. Captured variables and external variables are separate entities; modifying external variables does not affect variables inside the lambda
auto f1 = [=]() { return a + b; };
// [&] Capture all external variables by reference. Captured variables and external variables are the same entity; external modifications are visible inside the lambda and vice versa
auto f2 = [&]() { a = 10; };
```

**Generic lambdas** (since C++14) allow automatic deduction of parameter types. PSuM makes extensive use of this feature:

```cpp
auto f = [](auto& elem) { elem *= 2.0; };  // Works with int, double, or custom types
```

#### The "Double Lambda" Pattern in PSuM

PSuM's core traversal pattern is the **double lambda**. This pattern is dictated by the constraints of SYCL heterogeneous computing:

```cpp
// First layer (host side): [&] capture by reference, receives sycl::handler
device_vec.for_each([&](sycl::handler& h) {
    // Obtain accessors to device resources here
    auto acc = some_field.get_access(h);
    // Second layer (device side): [=] capture by value, executes in parallel on GPU
    return [=](Particle& p) {
        // p is an element in the container
        get<velocity>(p) = Eigen::RowVector3d::Zero();
    };
});
```

**Why split into two layers?**
- **The first layer** runs on the CPU and handles preparation: obtaining device data accessors via `handler`. Using `[&]` is appropriate because we are still in the host context and can safely reference external variables.
- **The second layer** runs on the GPU (or parallel CPU threads) and is the actual kernel function. It must use `[=]` value capture because the device cannot access host memory.
Accessors returned by the first layer are lightweight value objects (containing only pointers and sizes) that can be safely copied to the device.
In PSuM, any resource that resides on the device and needs to be accessible within a kernel provides an accessor.

*What is a kernel function: Simply put, a kernel function is a function that runs on the device. The primary way to launch kernels in PSuM is through the `for_each` method of various containers.*

**Supplementary: SYCL Kernel Submission**

Without PSuM's abstractions, writing a parallel for loop directly in SYCL looks like this:

```cpp
// USM style (using raw pointers)
sycl::queue q{sycl::default_selector_v};
std::vector<double> host_data(1000);
double* device_data = sycl::malloc_device<double>(1000, q);
q.memcpy(device_data, host_data.data(), 1000 * sizeof(double));
q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> idx) {
        device_data[idx] *= 2.0;
    });
}).wait();
```
Or the SYCL-native buffer/accessor style (which PSuM's `for_each` double lambda mimics):

```cpp
sycl::queue q{sycl::default_selector_v};
sycl::buffer<double> buf{sycl::range<1>(1000)};

// Initialize data: write via host_accessor
{
    sycl::host_accessor acc(buf, sycl::write_only);
    for (size_t i = 0; i < 1000; ++i) acc[i] = 1.0;
}

// Submit to queue + parallel_for launches the kernel
q.submit([&](sycl::handler& h) {
    sycl::accessor acc(buf, h, sycl::read_write);
    h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> idx) {
        acc[idx] *= 2.0;
    });
}).wait();

// Explicitly read results
std::vector<double> result(1000);
{
    sycl::host_accessor acc(buf, sycl::read_only);
    std::copy(&acc[0], &acc[1000], result.begin());
}
```

As you can see, PSuM's design is similar to SYCL's buffer/accessor style but simpler to use. The underlying implementation is based on SYCL's USM-related functionality.

PSuM's device containers encapsulate these low-level `malloc_device`, `memcpy`, and `parallel_for` calls as well as buffer/accessor creation and management, allowing you to focus on algorithm logic rather than memory management.

---

### Templates

Templates are C++'s core generic programming mechanism. PSuM uses templates to implement type-agnostic data structures and algorithms.

**Function templates:**

```cpp
// A function that can handle any type T
template<typename T>
T add(T a, T b) {
    return a + b;
}

add(1, 2);      // T = int
add(1.5, 2.3);  // T = double
```

**Class templates:**

```cpp
// device_vector is a class template where the element type is specified by T
template<typename T>
class device_vector {
    T* data_;
public:
    void for_each(auto func) { /* ... */ }
};

device_vector<Particle> particles(q, host_data);  // T = Particle
device_vector<double> scores(q, 10);              // T = double
```

**Non-type template parameters:**

In addition to types, template parameters can also be compile-time constants (integers, enums, etc.). PSuM's field system makes extensive use of this mechanism:

```cpp
// Dimension is a non-type template parameter of type int
template<int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
class device_field {
    static constexpr int Dimension = _Dimension;  // Can be used as a constant inside the class
    // ...
};

device_field<2, var_loc::cellCentered, double> field_2d;  // 2D
device_field<3, var_loc::nodeCentered, float>  field_3d;  // 3D, float scalar
```

The benefit of making dimension a template parameter rather than a runtime parameter is that the compiler can unroll loops and optimize memory layouts at compile time, generating more efficient executable code.

**Variadic templates and PSuM's particle definition:**

`tagged_struct` is the most intuitive application of variadic templates in PSuM. It can accept any number of `tag_bind<Tag, Type>` pairs:

```cpp
using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<mass, double>,
    tag_bind<charge, double>
    // You can continue adding any number of tag_binds
>;
```

The compiler expands `tagged_struct<A, B, C, ...>` into a class that inherits from `std::tuple`, with each `tag_bind` stored in order.
Access is done via `get<tag>(p)`, which looks up the index corresponding to the tag at compile time and generates a direct memory access.
Since `tagged_struct` types contain sufficient compile-time type information, serialization functions can also be generated directly.

---

### Concepts

Concepts, introduced in C++20, are a mechanism for constraining template parameters.
Since templates themselves do not restrict type parameters, users may sometimes provide unreasonable types.
This can lead to extremely hard-to-read compiler errors, or even silent errors.
Concepts provide explicit type constraints that produce more direct error messages when inappropriate types are supplied.

**Basic syntax:**

```cpp
// Define a concept: type T must support the + operator
template<typename T>
concept addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;  // T+T result must be T
};

// Use the concept to constrain the template parameter (replaces typename)
template<addable T>
T sum(T a, T b) { return a + b; }

sum(1, 2);      // OK: int is addable
sum("a", "b");  // Compile error, message will clearly state that "a" does not satisfy the addable concept
```

**Concept examples in PSuM:**

1. **`array_like<Dim>`** — The type must be a vector of the specified dimension (`std::array`, `Eigen::Vector`, or `Eigen::RowVector`):

```cpp
template<typename T>
concept array_like = 
    std::is_same_v<std::remove_cvref_t<T>, std::array<double, Dim>> ||
    std::is_same_v<std::remove_cvref_t<T>, Eigen::Vector<double, Dim>> ||
    std::is_same_v<std::remove_cvref_t<T>, Eigen::RowVector<double, Dim>>;
```

This constraint ensures that the vector type provided by the user meets the requirements, preventing dimension mismatches or errors from types that don't support `[]` element access.

2. **`is_validator`** — A validator must provide two static methods: `make_invalid(Particle&)` and `is_valid(const Particle&)`:

```cpp
template<typename Validator, typename Particle>
concept is_validator = requires(const Validator&, Particle p) {  
    { Validator::make_invalid(p) } -> std::same_as<void>;  
    { Validator::is_valid(std::declval<const Particle&>()) } -> std::convertible_to<bool>;
};
```

This concept is used as a constraint in `particle_group`'s template declaration (`requires is_validator<...>`).
If a user provides a validator that does not conform to the interface, the compiler error will precisely point to "does not satisfy is_validator" — rather than producing an incomprehensible template instantiation error deep inside `for_each`.

3. **`has_floating_x`** — The type must have a `.x()` method that returns a floating-point value:

```cpp
template<typename V>
concept has_floating_x = requires(V v) {
    { v.x() } -> std::convertible_to<double>;
};
```

PSuM uses this concept to constrain the bound types of the built-in tags `position` and `velocity`.
Any `T` placed in `tag_bind<position, T>` must satisfy `has_floating_x`, otherwise compilation fails.

4. **`a_tag_concept`** — A tag type must inherit from `abstract_tag` and have a name field:

```cpp
template<typename Type>
concept a_tag_concept =
    std::is_base_of_v<abstract_tag, Type> && 
    has_tag_name<Type>;
```

This is the most fundamental constraint of the tag system — any custom tag must satisfy both conditions before it can be used in `tag_bind`.

Concepts make the "contract" of a template interface understandable without reading the implementation code. If a provided type does not satisfy the requirements, the compiler will produce an error at the call site (rather than at some line deep inside the template).

---

### Device Containers and Ownership

In PSuM, any type that internally holds GPU device-side data has copying disabled. Writing code like the following will fail to compile:

```cpp
device_vector<Particle> vec2 = vec1;    // Error: copy is deleted
particle_group<P, V> pg2 = pg1;         // Error: copy is deleted
```

**Motivation:** The internal data of these types (particle arrays, field values, matrices, etc.) resides in device memory.
Allowing default copies could easily lead to expensive memory transfers or unsafe pointer management.

Types like `simple_grid` that only have CPU-side data are not subject to this restriction and can be copied normally.

**What to do in this situation?** It depends on your actual intent:

If you want an *independent copy*, explicitly reconstruct from host-side data:

```cpp
// device_vector: reconstruct via host data or relay through to_host/copy
device_vector<Particle> vec2(q, host_data);
auto host = vec1.to_host();
vec2.copy(host);
```

If you want to *transfer ownership*, both `device_vector` and `device_field` support moves, allowing you to hand off pointer ownership (the original object is emptied):

```cpp
device_vector<Particle> take_over = std::move(vec1);  // vec1 is no longer usable!
```

However, it is more recommended to use rvalue references (`T&&`) in function signatures for this purpose:

```cpp
void consume(device_vector<Particle>&& vec) {  // Explicitly requires: vec is an rvalue that can be taken over
    // ...
}
```
