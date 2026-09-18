[English](cpp-in-psum_en.md) | 中文

## PSuM 中的 C++

PSuM 使用 C++20 编写，其中用到了 lambda、模板、移动语义和 concept 等现代 C++特性。如果你对这些特性还不太熟悉，本章将帮助你快速掌握在 PSuM 中需要了解的基础知识。

---

### lambda 表达式

lambda 表达式是 C++中创建匿名函数对象的语法，在 PSuM 中承担核心角色。

**基础语法：**

```cpp
// 最简lambda: [捕获](参数){ 函数体 }
auto f = [](int x) { return x * 2; };
int result = f(3);  // 6
```

**捕获列表**决定了 lambda 可以使用哪些外部变量：

```cpp
int a = 1, b = 2;
// [=] 按值捕获所有外部变量. 被捕获变量和外部变量不是同一份实体, 修改外部变量不会影响lambda内的变量
auto f1 = [=]() { return a + b; };
// [&] 按引用捕获所有外部变量. 被捕获变量和外部变量是同一份实体, 外部的修改对lambda内可见, 反之亦然
auto f2 = [&]() { a = 10; };
```

**泛型 lambda**（C++14 起）允许参数类型自动推导，PSuM 大量使用这一特性：

```cpp
auto f = [](auto& elem) { elem *= 2.0; };  // 可用于 int, double, 或自定义类型
```

#### PSuM 中的"双层 lambda"

PSuM 的核心遍历模式是**双层 lambda**。这种模式由 SYCL 异构计算的约束决定：

```cpp
// 第一层（主机端）：[&]引用捕获，接收sycl::handler
device_vec.for_each([&](sycl::handler& h) {
    // 在此获取设备资源的访问器(accessor)
    auto acc = some_field.get_access(h);
    // 第二层（设备端）：[=]值捕获，在GPU上并行执行
    return [=](Particle& p) {
        // p 是容器中的一个元素
        get<velocity>(p) = Eigen::RowVector3d::Zero();
    };
});
```

**为什么要分成两层？**
- **第一层**运行在 CPU 上，负责准备工作：通过 `handler` 获取设备数据访问器。使用 `[&]` 是因为仍然在主机上下文，可以安全地引用外部变量。
- **第二层**运行在 GPU（或 CPU 的并行线程）上，是实际的核函数。必须使用 `[=]` 值捕获，因为设备端无法访问主机端内存。
第一层返回的访问器是轻量级值对象（只包含指针和尺寸），可以被安全地拷贝到设备端。
在 PSuM 中, 资源存在于设备端且能在核函数内访问的, 都会提供访问器。

*什么是核函数: 简而言之, 核函数就是运行在设备端的函数. PSuM 中最主要的核函数发起是使用各种容器的`for_each`方法。*

**补充说明：SYCL 的核函数提交**

如果不用 PSuM 封装，直接使用 SYCL 写一个并行 for 循环是这样的：

```cpp
// USM 风格 (直接使用指针)
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
或 SYCL 原生的 buffer/accessor 风格（PSuM 的`for_each`双层 lambda 模仿此模式）：

```cpp
sycl::queue q{sycl::default_selector_v};
sycl::buffer<double> buf{sycl::range<1>(1000)};

// 初始化数据：通过 host_accessor 写入
{
    sycl::host_accessor acc(buf, sycl::write_only);
    for (size_t i = 0; i < 1000; ++i) acc[i] = 1.0;
}

// submit提交到队列 + parallel_for发起核函数
q.submit([&](sycl::handler& h) {
    sycl::accessor acc(buf, h, sycl::read_write);
    h.parallel_for(sycl::range<1>(1000), [=](sycl::id<1> idx) {
        acc[idx] *= 2.0;
    });
}).wait();

// 显式读取结果
std::vector<double> result(1000);
{
    sycl::host_accessor acc(buf, sycl::read_only);
    std::copy(&acc[0], &acc[1000], result.begin());
}
```

可以看出, PSuM 中的设计与 SYCL 中的 buffer/accessor 风格类似但用起来更简单一些. 底层实现上则基于 SYCL 中 USM 相关功能.

PSuM 的设备端容器封装了这些底层的`malloc_device`、`memcpy`和`parallel_for`调用以及 buffer/accessor 的创建和管理，使用时可以专注于算法逻辑而非内存管理。

---

### 模板(template)

模板是 C++的核心泛型编程机制。PSuM 用模板来实现类型无关的数据结构和算法。

**函数模板：**

```cpp
// 一个可以处理任意类型T的函数
template<typename T>
T add(T a, T b) {
    return a + b;
}

add(1, 2);      // T = int
add(1.5, 2.3);  // T = double
```

**类模板：**

```cpp
// device_vector是一个类模板，其存储的元素类型由T指定
template<typename T>
class device_vector {
    T* data_;
public:
    void for_each(auto func) { /* ... */ }
};

device_vector<Particle> particles(q, host_data);  // T = Particle
device_vector<double> scores(q, 10);              // T = double
```

**非类型模板参数：**

除了类型，模板参数还可以是编译期常量（整数、枚举等）。PSuM 的场量系统大量使用这一机制：

```cpp
// Dimension 是一个int类型的非类型模板参数
template<int _Dimension, var_loc _location, typename _Scalar, int _QuantitySize = 1>
class device_field {
    static constexpr int Dimension = _Dimension;  // 可以在类内部作为常量使用
    // ...
};

device_field<2, var_loc::cellCentered, double> field_2d;  // 2维
device_field<3, var_loc::nodeCentered, float>  field_3d;  // 3维，float标量
```

将维度作为模板参数而非运行时参数的好处是：编译器可以在编译期展开循环、优化内存布局，从而生成更高效的可执行代码。

**变参模板与 PSuM 的粒子定义：**

`tagged_struct`是 PSuM 中最直观的变参模板应用, 它可以接受任意数量的`tag_bind<标签, 类型>`对：

```cpp
using Particle = tagged_struct<
    tag_bind<position, Eigen::RowVector3d>,
    tag_bind<velocity, Eigen::RowVector3d>,
    tag_bind<mass, double>,
    tag_bind<charge, double>
    // 可以继续添加任意多个 tag_bind
>;
```

编译器会将`tagged_struct<A, B, C, ...>`展开为继承了`std::tuple`的类，每个`tag_bind`按顺序存储。
访问时通过`get<tag>(p)`在编译期查找标签对应的索引，生成直接的内存访问。
由于`tagged_struct`类型中包含了足够多的编译期类型信息, 还能直接生成序列化函数.

---

### concept 概念

C++20 引入的 concept（概念）是一种约束模板参数的机制。
由于模板本身不限制类型参数, 用户有时会填入不合理的类型.
有时会导致可读性极差的编译错误, 有时甚至会导致静默错误.
concept 就是一种显式的类型约束, 能在填入不恰当的类型时给出更直接的错误信息。

**基础语法：**

```cpp
// 定义一个概念：类型T必须支持 + 操作
template<typename T>
concept addable = requires(T a, T b) {
    { a + b } -> std::same_as<T>;  // T+T 的结果必须是T
};

// 使用概念约束模板参数（替代typename）
template<addable T>
T sum(T a, T b) { return a + b; }

sum(1, 2);      // OK: int 可加
sum("a", "b");  // 编译错误，错误信息会明确说 "a" 不满足 addable 概念
```

**PSuM 中的概念实例：**

1. **`array_like<Dim>`** — 类型必须是某种指定维度的向量（`std::array`、`Eigen::Vector` 或 `Eigen::RowVector`）：

```cpp
template<typename T>
concept array_like = 
    std::is_same_v<std::remove_cvref_t<T>, std::array<double, Dim>> ||
    std::is_same_v<std::remove_cvref_t<T>, Eigen::Vector<double, Dim>> ||
    std::is_same_v<std::remove_cvref_t<T>, Eigen::RowVector<double, Dim>>;
```

这种约束可以确保用户传入的向量类型符合要求, 不会出现维度不匹配或不能用`[]`获取元素的错误。

2. **`is_validator`** — 验证器必须提供两个静态方法：`make_invalid(Particle&)` 和 `is_valid(const Particle&)`：

```cpp
template<typename Validator, typename Particle>
concept is_validator = requires(const Validator&, Particle p) {  
    { Validator::make_invalid(p) } -> std::same_as<void>;  
    { Validator::is_valid(std::declval<const Particle&>()) } -> std::convertible_to<bool>;
};
```

这个 concept 在`particle_group`的模板声明中被用作约束（`requires is_validator<...>`）。
如果用户提供了一个不符合接口的验证器，编译错误会精确地指向"不满足 is_validator"——而非在`for_each`深处产生难以理解的模板实例化错误。

3. **`has_floating_x`** — 类型必须有返回值为浮点的 `.x()` 方法：

```cpp
template<typename V>
concept has_floating_x = requires(V v) {
    { v.x() } -> std::convertible_to<double>;
};
```

PSuM 用这个 concept 来约束内置标签 `position` 和 `velocity` 的绑定类型。
任何放在`tag_bind<position, T>`中的 `T` 必须满足 `has_floating_x`，否则编译失败。

4. **`a_tag_concept`** — 标签类型必须继承`abstract_tag`并有名称字段：

```cpp
template<typename Type>
concept a_tag_concept =
    std::is_base_of_v<abstract_tag, Type> && 
    has_tag_name<Type>;
```

这是标签系统最基础的约束——任何自定义标签必须满足这两个条件，才能在 `tag_bind` 中使用。

concept 使得模板接口的"契约"在不阅读实现代码的情况下也能被理解。如果传入的类型不满足要求，编译器会在调用处（而非模板深处的某行）给出错误提示。

---

### 设备端容器与所有权

PSuM 中，凡是内部持有 GPU 设备端数据的类型都禁用了拷贝。写下面这样的代码会直接编译失败：

```cpp
device_vector<Particle> vec2 = vec1;    // 错误：拷贝已删除
particle_group<P, V> pg2 = pg1;         // 错误：拷贝已删除
```

**动机：** 这些类型的内部数据（粒子数组、场值、矩阵等）位于设备显存中。
如果允许默认拷贝，容易导致昂贵的内存移动或者不够安全的指针管理。

`simple_grid` 这类只有 CPU 端数据的类型不受此限制，可以正常拷贝。

**遇到这种情况怎么办？** 取决于你的真实意图：

想获得一份*独立的副本* 则从主机端数据显式重建：

```cpp
// device_vector: 通过主机数据或 to_host/copy 中转
device_vector<Particle> vec2(q, host_data);
auto host = vec1.to_host();
vec2.copy(host);
```

*想转移所有权*, 对于`device_vector` 和 `device_field` , 都支持移动，可以把指针所有权交出去（原对象被清空）：

```cpp
device_vector<Particle> take_over = std::move(vec1);  // vec1 不再可用!
```

不过更推荐在函数签名中使用右值引用（`T&&`）进行限定：

```cpp
void consume(device_vector<Particle>&& vec) {  // 明确要求: vec是一个可将被接管的右值
    // ...
}
```