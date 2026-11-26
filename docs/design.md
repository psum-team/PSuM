[English](design_en.md) | 中文

## PSuM 框架设计

### 设计目标与读者

本文面向框架维护者和贡献者，解释 PSuM 的内部组织、数据结构和实现机制。如果你是使用者，想学习如何写 PSuM 程序，请阅读 [编程指南](programming.md)。

框架设计总是为使用者服务，never forget the beginner's mind!

### 项目文件布局

PSuM 的源码组织遵循如下基本原则：**用户入口和实现细节分开，header-only 为主要模式，测试目录和源码模块尽量平行**。

仓库中最重要的目录如下：

- `include/psum/`：对外公共入口。用户程序通常包含这里的头文件，例如 `#include <psum/psum.hpp>`。这些文件负责聚合模块，不应承载大量复杂实现。
- `src/`：框架主体实现。子目录基本对应模块和命名空间，例如 `src/field/` 对应场量模块，`src/particle_container/` 对应粒子容器模块。
- `test/`：模块测试。它和 `src/` 大体保持平行结构，例如 `test/field/` 测试 `src/field/`，`test/field_solver/` 测试 `src/field_solver/`。
- `example/`：教学示例。这里的程序应该比测试更适合阅读，用来展示最小 API 用法。
- `application/`：实际算例或 benchmark。这里可以包含更完整的物理模型、配置文件和输出流程。
- `pypsum/`：Python 侧支持，目前主要服务于序列化相关的 Python/C++ 互操作，不应理解为完整 Python 版 PSuM。
- `docs/`：项目文档。
- `docker_install/`：Docker 安装路径，与本地依赖安装互为替代。

#### 公共入口与实现目录

`include/psum/psum.hpp` 是最常用的主入口。它聚合 tag、serialization、particle_container、field、particle_boundary、
field_solver、particle_collision、random、timer、utils_sycl 等模块，并定义 `psum::prelude`：

```cpp
#include <psum/psum.hpp>
using namespace psum::prelude;
```

模块级入口包括：

- `include/psum/tag.hpp`
- `include/psum/serialization.hpp`
- `include/psum/particle_container.hpp`
- `include/psum/field.hpp`
- `include/psum/field_solver.hpp`
- `include/psum/particle_boundary.hpp`
- `include/psum/particle_collision.hpp`

PSuM 目前是一个框架库，以源码形式提供功能而非以二进制提供：使用者将完整源码树纳入自己的编译过程，`include/psum/` 公共入口与 `src/` 实现共同构成被编译的源码集。
当前项目中许多实现以 header-only 形式存在，因此公共入口最终会包含 `src/` 下的实现头文件。
但从设计上看，`include/psum/` 仍然是用户侧边界；应用代码应优先依赖公共入口，而不是直接把 `src/` 的目录结构当作稳定 API。

新增模块时，建议先在 `src/<module>/` 中组织实现，再决定是否通过 `include/psum/<module>.hpp` 暴露给用户。
这样可以避免内部文件结构过早变成外部承诺。

#### 测试目录与模块对应

`test/` 目录既用于验证，也用于记录行为契约。它不是教程目录。

常见对应关系：

- `test/field/`：验证 `simple_grid`、`host_field`、`device_field`、插值和沉积。
- `test/field_solver/`：验证 Poisson solver、边界条件、ghost 处理和后端求解。
- `test/particle_boundary/`：验证 STL 载入、几何相交、空间分区、boundary router 和材料行为。
- `test/particle_collision/`：验证 MCC/DPMCC、pairing engine 和碰撞相关工具。
- `test/pypsum/`：验证 Python/C++ 序列化互操作。

批量验证入口为 `test/runCheck.sh`，会编译并短超时运行所有可执行目标（冒烟测试）。

如果新增 `src/<module>/`，原则上也应考虑新增或扩展 `test/<module>/`。
示例程序则应放在 `example/` 或 `application/`，不要把测试代码当作用户教程来维护。
此外，测试代码中一般不使用`#include <psum/psum.hpp>`, 因其引入了测试内容之外的依赖。


### PSuM 中的主要模块

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
- **基础元编程层** — `tag` 系统和 `serialization` 系统不直接参与 PIC 计算，但为上层提供了编译期信息和持久化能力。
几乎所有模块都依赖 tag。
- **容器层** — `device_vector` / `device_array` / `particle_group` 在设备端管理数据。
它们本身不实现 PIC 算法，只是高性能数据结构。
- **PIC 模块层** — 五个核心模块："3P2F"。`particle_container` 和 `field` 既属于容器层也是 PIC 模块（它们同时提供数据结构和算法接口）；
`particle_boundary` 处理粒子-几何边界交互；`particle_collision` 处理粒子间碰撞；`field_solver` 求解泊松方程。
除此之外, 还有一些较小的模块或实用工具:
- **工具层** — `random`（随机数生成与物理分布）、`timer`（性能计时）、`ui`（键盘输入）、`utils_sycl`（SYCL 工具函数）。不依赖上层模块。

文件组织遵循**目录 = 命名空间，文件名 = 子命名空间或主类型**的约定。
例如 `src/field/device_field.hpp` 对应 `psum::field::device_field`，`src/tag/foundation.hpp` 对应 `psum::tag::foundation`（包含多个工具类型）。

### tag 系统的实现

PSuM 之所以称之为元编程框架正是因为 tag 系统.

**类型层级：**
```
abstract_tag                 ← 标签的基类
    ├── position             ← 内置属性标签
    ├── velocity             ← 内置属性标签
    ├──   ...                ← 内置属性标签
    └── 用户自定义标签

concrete_tag                 ← 绑定实例的基类
    └── tag_bind<Tag, Type>  ← 标签+类型的配对
```

**核心概念：**
- `a_tag_concept<T>` — 要求 `T` 继承 `abstract_tag` 且有 `static std::string tag_name`
- `c_tag_concept<T>` — 要求 `T` 继承 `concrete_tag`

**tagged_struct 的原理：**

`tagged_struct<tag_bind<T1,V1>, tag_bind<T2,V2>, ...>` 展开为继承 `std::tuple<V1, V2, ...>` 的类。访问 `get<Tag>(obj)` 的工作流程：

1. `tag_bind` 同时暴露 `front`（标签类型）和 `back`（数据类型）两个别名
2. `dual_tuple` 利用 `front`/`back` 将这些对分离为两个 tuple：`tuple_of_tag`（标签集）和 `tuple_of_type`（类型集）
3. `index_in_tuple<Tag, tuple_of_tag>` 通过递归模板在编译期查找标签的索引位置
4. `get<Tag>(obj)` 获取该索引 → 调用 `std::get<index>(tuple_base)` → 返回对应字段的引用

整个过程完全在编译期完成，生成的代码等价于 `obj.field_N`，无运行时开销。

**标签验证：** `try_check<Tag, Type>()` 在 `tag_bind` 的 `static_assert` 中被调用.
当用户写下 `tagged_struct<tag_bind<Tag, Type>, ...>` 的时候，编译时触发.
如果标签携带了 `check<Type>()` 且检查未通过，在此处编译失败。
这使得不合理的类型绑定在编译期被拦截。

### 粒子容器的实现

#### device_vector 的内存模型

`device_vector<T>` 使用 SYCL USM 在设备端分配内存：

- **构造时**：调用 `sycl::malloc_device` 分配原始设备内存；析构时 `sycl::free` 释放
- **容量管理**：维护 `data_`（原始指针）、`size_`（当前元素数）、`capacity_`（总容量）、`overflow_`（溢出标志）
- **线程安全 push_back**：`device_vector_acc` 的 `push_back` 使用 `sycl::atomic_ref<size_t>` 原子自增 `size_`。
获得唯一的写入位置后直接拷贝数据。多个线程同时 push 不会冲突，但容量无法自动增长——超出 capacity 时设置 `overflow_` 标志。
主机端操作（`size()`、`data()`、`to_host()`、`copy()` 等）都会首先检查该标志并抛出异常，确保溢出错误被*尽快*举起。

**双层 lambda 的机制：**

```cpp
vec.for_each([&](handler& h) {
    auto acc = get_access(h);  // 将指针+size打包为轻量value
    return [=](T& elem) {
        elem = ...;
    };
});
```

- 外层被转换为 SYCL 的主机端`h.parallel_for`, 用于提交作业
- 内层 lambda 必须 `[=]` 值捕获，因为设备端无法通过引用访问主机端的变量.
- `acc`（包含裸指针+size）须是 **trivially copyable**，可以被安全传送到设备端

#### particle_group 的双数组设计

`particle_group` 维护两个 `device_vector`：

- `data_` — 存储所有粒子（有效 + 无效）
- `invalid_indexes_` — 存储无效粒子的索引

**size() = data_.size() - invalid_indexes_.size()** — 有效粒子数

`device_vector` 是唯一支持设备端 `push_back` 的容器（通过 `device_vector_acc` 原子操作）。`particle_group` 只能在主机端通过 `insert()` 批量添加粒子。

**insert 三策略：**
1. 无无效槽位（`invalid_indexes_.size() == 0`）：`data_.resize()` 追加 → `memcpy` 到末尾
2. 无效槽位足够（`insert_size <= invalid_indexes_.size()`）：从栈中弹出对应数量的索引，将新粒子拷贝到这些位置
3. 无效槽位存在但不足：先填充无效槽位，剩余追加

**惰性删除：** 在 `for_each` 中调用 `validator::make_invalid(p)` 时，将当前粒子的索引 push 到 `invalid_indexes_` 中。
遍历结束后，无效粒子的位置可被后续 `insert` 复用，但物理内存不改变。validator 同时能保证后续的 for_each 不会再访问到这个粒子。

**compress 算法：** 从后向前扫描，将末尾有效粒子移动到前方的无效位置，截断数组。

**shuffle 算法**：使用 Feistel 网络方法进行重排，实现较高的性能。

### 场量的实现

#### simple_grid

`simple_grid<Dim>` 描述一个正则笛卡尔网格，内部核心数据：
- `lower_bounds` / `upper_bounds` — 起止坐标，`std::array<double, Dim>`
- `cell_num` — 每维格子数，`std::array<int, Dim>`

并储存 `delta`（格子边长=1/cell_num）、`delta_r`（倒数）、`length`（总长）。

**flat indexing** 将多维网格索引映射到一维线性数组：
- 节点总数 = ∏(cell_num[i] + 1)
- 格子总数 = ∏(cell_num[i])
- `n2i(n)` / `c2i(c)` 将节点/格子索引转为线性索引（对应于数组存储）
- `i2n(i)` / `i2c(i)` 反向映射

#### device_field 与访问器

`device_field<Dim, Location, Scalar, QSize>` 内部持有：
- `device_array<Value>` — 设备端线性存储的场值
- `simple_grid<Dim>` — 网格

场量本身是**不可直接访问**的——必须通过访问器：

```cpp
decltype(auto) get_access(sycl::handler& h) {
    return acc_type(device_content.get_access(h), grid);
}
```

访问器 `device_field_acc` 包装了 `device_array_acc` + `simple_grid`，是 trivially copyable 的值类型。内核函数通过 `acc(idx)` 读写场值，`acc.getGrid()` 获取网格引用。

这样做的好处是：场量对象（`device_field`）可以留在主机端，访问器被拷贝到设备端执行插值和沉积——避免了设备端持有复杂的场量对象。

#### 插值与沉积

用户在核函数中可用使用多个插值/沉积相关函数，均接受位置和场访问器：

- **`interp(pos, field_acc)`** — 在粒子位置线性插值场量值（仅节点场）
- **`interp_diff(pos, field_acc)`** — 插值场量梯度，返回 `Dim` 组值（仅节点场）, 为各个分量上的梯度
- **`add_back(pos, weight, field_acc)`** — 将粒子权量沉积到网格（原子加，仅节点场）
- **`interp_nearest(pos, field_acc)`** / **`add_back_nearest(pos, weight, field_acc)`** — 最近邻变体（节点场和格子场均可用）

`interp, interp_diff, add_back` 函数内部通过 `interp_tools::generate_interp` / `generate_add_back` 等统一实现，模板参数指定具体的 1D 插值核。

**插值核心 `interp_tensor`：** 将 1D 插值函数扩展到 N 维的张量积框架。

1D 线性插值函数 `linear_interp_1D(x)` 返回 `{1-x, x}`（两个系数，对应左右节点）。
`interp_tensor` 对每个维度独立调用 1D 函数，将结果做笛卡尔积，产生 `2^Dim` 个支持点及其系数。

例如 2D 时，支持点为 4 个角节点，系数为 `(1-x)(1-y), (1-x)y, x(1-y), xy`。

**Gather（`generate_interp`）：**
```
对每个支持点 i：
    res += field_acc(index[i]) * coeff[i]
```

**Scatter（`generate_add_back`）：**
```
对每个支持点 i：
    atomic_add(field_acc(index[i]), value * coeff[i])
```

原子加封装在 `_atomic_add_` 中，内部使用 `sycl::atomic_ref` 的 `fetch_add`。

**梯度插值 `generate_interp_diff`：** 对每个维度返回各自的一组系数（1D 插值核的导数与该维度正常插值核的张量积），产生 `Dim` 组结果。

### 场量求解器的实现

场量求解器的职责是把网格、边界条件和源项组织成线性方程组，并把“如何求解线性方程组”留给后端。
PSuM 目前主要求解静电 PIC 中常见的 Poisson 方程，接口按维度分为 `Poisson_solver_1d`、`Poisson_solver_2d`、`Poisson_solver_3d`。

求解器内部有两层结构：
- **前端**：由 C++ 模板和 Eigen 稀疏矩阵组成，负责离散化、边界条件装配、右端项修正，以及与 PSuM 的 `field` 数据结构衔接。
- **后端**：由 `solver_backend` 封装，负责具体线性方程求解。默认后端是 `native`，使用 Eigen；也可以通过统一函数表接入 CUDA 或其他求解器实现。

#### 离散化与矩阵装配

`Poisson_solver_2d` 和 `Poisson_solver_3d` 使用规则网格上的有限差分离散。
以二维笛卡尔坐标为例，网格节点上的未知量为电势 `phi`，内部节点对应的离散方程大致是：

```text
(phi_E - 2 phi_C + phi_W) / dx^2
+ (phi_N - 2 phi_C + phi_S) / dy^2
= -rho / epsilon
// 这里忽略了 epsilon 的不均匀. 在Poisson_solver_2d中，epsilon 可以具有分布。
```

初始化过程中将形成稀疏线性系统, 以`Eigen::SparseMatrix<double>`为系统的系数矩阵。
这样求解器只需要处理标准稀疏矩阵, 由不同的后端进行求解。

二维求解器还支持 `Cartesian` 与 `Cylindrical` 两种坐标形式。
圆柱坐标下的径向项带有几何系数，因此矩阵装配会根据节点位置调整相邻节点的系数；但从外部看，用户仍然只是在一个二维网格上求解 `phi`。

#### 固定边界与混合边界

PSuM 将边界条件拆成两类：

- **固定边界**：例如 Dirichlet 条件，直接指定某些节点的值。实现上会把对应矩阵行替换成 `phi_i = value`，并在右端项中写入边界值。
- **混合边界**：例如 Robin 条件，形式上包含函数值和法向导数。它不能简单替换矩阵行，需要引入 ghost 节点表达边界外侧的虚拟值。

固定边界优先级高于混合边界。

#### ghost_element_manager

混合边界的核心工具是 `ghost_element_manager`。
它把主系统变量 `x` 和 ghost 变量 `x_ghost` 组织成分块线性系统：

```text
[A, B] [x      ] = [b             ]
[C, D] [x_ghost]   [boundary_value]
```

其中：
- `A` 是主网格节点方程。
- `B` 是主方程中 ghost 节点的贡献。
- `C, D` 来自边界方程，用于表达 ghost 节点和主节点之间的关系。

`ghost_element_manager::get_elimination_system()` 会消去 ghost 变量，得到只含主变量的系统：

```text
A' x = b + M * boundary_value
A' = A - B * D^-1 * C
M  = -B * D^-1
```

这就是 `Poisson_solver_*` 中 `A_` 和 `M_` 的来源。
`A_` 交给求解后端；`M_` 在每次 `solve()` 前参与右端项修正，因为 Robin 边界的边界值可能随时间变化。

#### solver_backend 插件系统

`solver_backend` 是 PSuM 中对多种求解器后端的统一接口类。
它内部保存两个对象：

- `psum_field_solver_handle handle`：后端求解器实例。
- `func_table funcs`：一组 C 风格函数指针，包括 `init`、`set_matrix`、`solve`、`set_options`、`set_source_replace`、`set_source_addback`、`free`。

构造 `solver_backend(kind)` 时，前端通过 `get_solver_funcs(kind)` 获得函数表，调用 `init()` 创建后端实例。
析构时调用 `free(handle)` 释放资源。
这使得后端可以用不同技术实现，只要遵守同一套 C ABI 接口即可。

求解流程则是：

```text
1. 根据输入源项构造右端项
2. native 后端调用 Eigen 求解；其他后端调用 backend.solve(source, phi)
```

关于后端插件系统的完整接口规范、资源管理约定和实现细节，请参阅 [场求解器后端设计文档](detailed_design/field_solver_backend.md)。

### 粒子边界系统的实现

粒子边界模块处理“粒子从旧位置走到新位置时是否穿过几何表面，以及穿过后应该发生什么”。
它由两个相互独立的部分组成：

- **trigger**：检测轨迹和三角面片是否相交，并返回最近交点、法线和材料类型。
- **material_set**：根据材料类型修改粒子，例如吸收、镜面反射、漫反射或按壁面温度重新采样速度。

#### boundary_router

`boundary_router<MaterialSet, ParticleType>` 是用户侧接触的入口。
它内部持有：

- `triangle_mesh_trigger<material_type>`：几何触发器。
- `MaterialSet`：材料行为集合。

在核函数中，用户通过 `router.get_access(h)` 获得 `boundary_router_acc`，然后调用 `router_acc.deal(old_pos, new_pos, p)` 处理粒子边界。

`deal()` 的执行流程是：

```text
1. trigger_acc.detect(...) 检测线段和网格表面的相交
2. 如果没有相交，返回 false
3. 如果相交，根据参数 k 计算命中点 hit = p1 + k * (p2 - p1)
4. 将材料类型、命中点、法线传给 material_set_acc.action(...)
5. material_set 修改粒子并返回 true
```

注意这里传入的是旧位置和新位置，而不是单个粒子位置。
这样可以避免粒子一步跨过薄壁时漏检。

#### 空间分区与加速

直接检查每个粒子轨迹和每个三角形的相交会非常慢。
`triangle_mesh_trigger` 因此会先把几何体放入规则空间分区中。
用户调用 `boundary_router::set(meshes, I, J, K)` 时，`I/J/K` 指定三维分区数量。

空间加速主要依赖两步：

1. **三角形预分配**：把三角形放入可能相交的结构化网格空间分区。这在 deal 函数外、trigger 对象配置时完成。
2. **线段遍历分区**：粒子轨迹从旧位置到新位置形成一条线段，`trace_to_idxs_begin` / `trace_to_idxs_next` 按顺序遍历这条线段穿过的分区。

线段遍历类似 DDA 网格遍历算法。
它维护当前位置所在分区、运动方向、到下一条网格面的参数距离 `t`，每次跨过最近的网格面并返回新的分区编号。
这样一条粒子轨迹只需要检查它实际穿过的少量分区。

真正的线段-三角形相交检测由 `line_seg_tri_intersect_test` 完成。采用 Moller-Trumbore 算法.

如果一条轨迹命中多个三角形，trigger 会选择最近交点。此外，还提供了一些性能调优的选项, 如开启无符号距离场。

#### 材料行为分发

材料行为在 `basic_material_set::acc_type::action()` 中通过 `material_type` 枚举分发。

材质集的编写是常规且容易的. PSuM 提供了一个简单的 `basic_material_set` , 内置四种材料行为：

- `absorb`：调用 `Validator<ParticleType>::make_invalid(p)`，将粒子标记为无效。
- `specular`：镜面反射，速度更新为 `v' = v - 2 (v dot n) n`。
- `diffuse`：按余弦分布采样新的反射方向，保持速度大小近似不变。
- `maxwell_reflect`：按壁面温度和粒子质量从半空间 Maxwell 分布采样新的速度。


### 碰撞系统的实现

PSuM 的碰撞模块分为两类：

- **MCC**：Monte Carlo Collision，处理粒子与背景物种之间的碰撞，例如电子与中性气体碰撞。
- **DPMCC**：Deferred Pairing MCC，处理需要显式配对的粒子-粒子碰撞，例如 DSMC 中的分子对碰撞。

二者都围绕“用户提供碰撞模型，框架负责并行遍历、上下文访问和新粒子缓冲”这一思路设计。

#### collision_model 与上下文

`collision_model<CollisionTypes...>` 直接列出唯一的碰撞类型。每个类型声明
`incident_species_tag` 和 `background_species_tag`，框架据此分组并选择执行器。

| 继承的标记 | 通道类型 | 回调 |
| --- | --- | --- |
| 无 | 普通 MCC | `collide` |
| `reaction_collision_tag` | 反应 MCC | `collide` |
| `deferred_pairing_collision_tag` | 普通 DPMCC | `request_probability`, `pair_collision_cross_section` 和 `pair_collide` |
| 两者 | 反应 DPMCC | `request_probability`, `pair_collision_cross_section` 和 `pair_collide` |

反应通道必须用 `std::tuple` 声明 `consumed_species_tags` 和 `product_species_tags`。
MCC 只能消耗入射物种，DPMCC 可以消耗配对端点的物种。产物也可以是独立的物种。
没有消耗物或产物时，对应项写成 `std::tuple<>`。

碰撞上下文在 `src/particle_collision/context/` 下按职责组织：

- `species_context` 统一绑定粒子组和所需密度场。入射和背景可以使用同一份绑定，配对视图读取粒子组的当前首地址和槽位范围。
- `species_contexts<Model, PG, Field>` 在构造时接收按 `Model::species_tuple` 排列的绑定，并用第一个粒子组的队列校验其余粒子组。在任一通道充当背景的物种需要绑定密度场，其余物种只需绑定粒子组。模型声明的产物物种会分配插入缓冲。
- `collision_channel_context` 持有单条延迟通道的请求缓冲和配对引擎。由 `collision_channel_contexts` 按碰撞通道类型管理。
- `tagged_context` 把资源绑定到标签。`context_composition` 组合这些上下文，核函数通过 `ctx_acc.get<Tag>()` 获取对应资源。

框架选中具体反应通道后，按该通道的标签开放增删接口。
`ctx.template insert_particle<ProductTag>(p)` 将产物加入该物种的缓冲区。
MCC 反应调用 `ctx.template erase_particle<IncidentTag>()` 删除当前粒子。
DPMCC 反应调用 `ctx.template erase_particle<IncidentTag>(pair.first)` 或 `ctx.template erase_particle<BackgroundTag>(pair.second)`。
每次调用只请求删除一个端点。只有当前配对中属于该物种的粒子指针才会登记删除。同种配对的两次调用使用同一个物种标签。
删除在回调返回时生效，重复删除同一粒子只登记一次。

完整碰撞步调用 `execute_collision_model<Model>(ctx, dt)`。其中的入射阶段 `execute_incident_collisions<Model::collision_types_tuple>(ctx, dt)` 按入射物种遍历并运行对应碰撞通道。
完整碰撞步中，每个入射物种遍历一次。每个粒子先执行 MCC，再按更新后的状态生成全部 DPMCC 请求。背景批次在配对前过滤失效请求并重新计数，每批最多两段：第一段包含全部同种通道，第二段包含全部异种通道，前一段完成后才开始后一段。所有产物在全部批次结束后统一回填，从下一步开始参与碰撞。产物缓冲在碰撞开始前预留容量，回填时直接插入并清空，不再根据占用率提前翻倍扩容。
执行前复用抽样通道组：入射物种、背景物种及 MCC/DPMCC 类型相同的通道属于同组。对每种产物，组内取单次最大产物数，乘以本步开始时的入射粒子数，再跨组累加并加上缓冲已有产物数。product_species_tags 中同一 Tag 的重复次数表示单次碰撞对此物种的最大插入次数；例如 tuple<Electron, Electron> 声明最多新增两个电子。各产物分别取最大值，MCC 和 DPMCC 分组分别累加。容量估算仅覆盖正常生成的请求，不考虑手工预置请求。插入次数上限由回调遵守，不新增逐次计数，仍保留缓冲溢出检查。容量计算在本步开始时按需读取各入射物种的有效粒子数，同一物种在产物和请求预留中共用一个宿主局部缓存；该缓存不参与碰撞中的失效过滤，下一步重新建立。
请求缓冲预留先用宿主端 extent() 作为本步新增请求数的上界，加上已有请求数；容量足够时不回读有效粒子数。只有上界不满足时才读取并复用上述有效粒子数缓存，实际不足时按 max(所需容量, 原容量 + 原容量/2) 扩容。已有容量不缩小；该增长策略仅用于请求缓冲，产物预留保持原规则。
同种配对的入射标记数组由对应物种上下文持有，仅在出现有效请求时按需分配，并按约1.5倍增长复用。每个非空同种段先清零当前extent范围，再由过滤后的请求写标记；后续通道共享本段标记，不重复清零。数组保留到物种上下文销毁，不进入设备访问器；异种通道和纯MCC不分配。
框架提供了模板元编程工具, 可以在提供的类型元组范围上进行遍历, 允许写出声明式的模型定义。

一次 MCC 执行的流程是：

```text
1. 对入射物种的 particle_group 执行 for_each
2. 在核函数中插值背景密度，计算碰撞概率
3. 随机选择具体碰撞通道
4. 修改当前粒子，或把新粒子 push_back 到某一物种的缓冲区
5. 所有碰撞结束后，flush_product_buffers 把 buffer 批量 insert 回 particle_group
```

buffer 的存在很重要。
在设备端不能直接向 `particle_group` 插入新粒子。
`device_vector` 支持设备端 `push_back`，可以作为临时缓冲区。

#### deferred pairing 与 pairing_engine

有些碰撞不能只看一个粒子和背景场，而必须找到另一个真实粒子作为碰撞对象。
DPMCC 通过“延迟配对”处理这一类问题。

一个延迟配对碰撞类型会继承 `deferred_pairing_collision_tag`。

DPMCC 必须提供 request_probability(p,ctx) -> double、
pair_collision_cross_section(pair,ctx) -> double（物理截面，单位 m^2）和
pair_collide(pair,ctx) -> void，不再使用单粒子 collision_cross_section(v2)。
MCC 仍保留原单粒子截面接口。

请求概率是每条通道的实际概率，必须有限且非负，同一抽样组的总和不得超过 1；
剩余概率表示不发请求。非法概率组不产生请求，不自动归一化。
框架不再为请求概率乘速度、密度或 dt；返回固定值（例如 0.25）时，
静止入射粒子也可发请求。payload 记录选中通道的概率，不是组总概率。

配对后框架计算：
p = factor * sigma * |v_first - v_second| * n_background * dt / request_probability。
同种通道 factor=0.5*(N-1)/N，异种通道 factor=1；N 小于 2 时不执行同种碰撞。
各物种的粒子位置使用相同的物理坐标约定。请求入格、背景粒子入格和背景密度插值均采用背景物种的投影。密度取自入射粒子所在的位置。
截面回调不得再包含同种系数、密度、dt 或请求概率补偿。
保留原有非法概率警告和超过 1 的截断；非法截面会警告并拒绝碰撞。
没有引入 M/K 补偿。

`collision_channel_contexts` 会从 `collision_model` 筛选这些类型，以碰撞类型为键为每种通道准备一个 buffer。

执行流程是：

```text
1. 入射粒子触发延迟碰撞时，把 {uint32_t 网格编号, {&p, nullptr, 请求概率}} 写入 DPMCC buffer
2. pairing_engine 根据空间网格把这些请求映射到分区
3. 遍历背景粒子，抢占每个分区中的请求，把选中的背景粒子指针填入第二个端点
4. 框架根据双粒子物理截面计算接受概率，无锁抽样；仅接受后调用 pair_collide
```

从统一生成请求到全部背景批次执行完毕，粒子组须保持端点地址和槽位归属稳定。手工分阶段调用时，应等涉及该粒子组的待处理请求全部完成后，再扩容、重排或复用空槽。
异种请求要求不同的粒子组。同种段先过滤失效请求，再把所有同种通道中仍有效的请求端点统一标记为入射组；其余有效粒子构成背景组。标记按粒子原始槽位索引，每槽一字节，由物种上下文保留并复用，每个非空同种段清零当前 extent 范围，完成标记后再遍历背景，配对时只读。正常请求生成保证每个入射粒子在同种段只发一个请求，背景粒子也最多领取一个请求，因此两个端点以及不同通道的配对均不重叠；不需要锁表、原子占用或解锁。手工添加请求也必须保证同段入射端点不重复。第一段存活的粒子仍可参与第二段的异种碰撞；下一批重新按请求划分角色。绕过 deal 手工提交配对时，调用者须保证同一并行执行中的端点不重叠。

正常模型的每段至多一个碰撞执行核，配对与请求分区仍先单独完成。同一背景物种的每段内，所有通道统一互斥：每个背景粒子按原通道顺序尝试，首次配对成功后跳过其余通道，不区分是否反应或消耗背景；随后概率拒绝也不重新分配。同种段和异种段都直接遍历各通道引擎中的请求，背景端点为空就跳过，否则执行碰撞；不再建立额外的任务数组，也不按消耗类型排序。

原始请求缓冲保存请求生成时的 uint32_t 网格编号和 payload，不再保存完整位置；生成时位于网格外的请求不入队。后续分区保留该编号，不因粒子位置变化而重新归类，仍过滤失效粒子。分区后的引擎数组只保存 payload，即两个粒子指针和请求概率；有效长度由宿主 total_size_ 管理，容量足够时不写设备端长度，扩容时直接丢弃旧批次内容。不存在背景槽位索引、任务数组或任务计数回读；执行核覆盖过滤后的请求总数，未匹配项直接返回。

species_contexts 构造时要求不同物种标签绑定不同粒子组，重复绑定直接抛出 invalid_argument。此要求覆盖入射、背景和产物物种，也适用于纯 MCC 模型。同一物种标签可以用于多个通道，其同种通道使用同一组入射标记，构造上下文时不再分配锁资源。执行阶段不再保留重复绑定检查、串行标记或串行回退。通道绑定必须提供碰撞回调，配对完成后直接执行，不再支持只配对并返回结果的路径，也不把配对结果复制回原 buffer；执行完成后只清空原请求。发请求只写入缓冲，不统计网格计数，也不需要单独准备计数器。每次 deal 对非空请求清零票据和计数，过滤失效请求后统计并分区；不再统计 K 或按配对不足放大接受概率。

只有模型中存在同种 DPMCC 通道的物种才在 species_context 内分配 uint32_t 单元中心粒子计数场，网格与该物种密度场一致。每次请求生成前清零，在同种通道组内、请求抽样前统计 MCC 后仍有效且位于配对网格内的粒子；多个同种通道共享该组，每个粒子只计一次，未发请求的粒子也计入。该场在请求核完成后供碰撞接受概率读取，表示请求生成时的快照，不随后续 DPMCC 消耗更新，也不计入最后回填的产物。异种 DPMCC 和纯 MCC 不分配该资源。直接调用底层同种碰撞执行接口的测试需自行提供计数。此有限粒子数修正不补偿背景不足或配对偏好；若密度与计数不对应同一状态，不能据此宣称概率严格无偏。

同种通道按已发请求划分两组，不额外随机分组，不修改请求概率；有限粒子数系数由框架统一应用。在一个有 N 个有效粒子的单元中，若 R 个粒子发出同种请求，最多形成 min(R,N-R) 对；全部粒子都发请求时没有背景候选，配对数为零。配对完成后执行碰撞，不补偿未匹配请求。

pairing_engine 的每格请求计数、前缀和及独立票号均使用 uint32_t，适用于每通道总请求数小于 2^32 的规模。前缀数组包含 C+1 个元素，首项为零，对 C 个计数作 inclusive scan 后只回读末项得到有效总数。同种段的入射标记在第一个存在有效请求的通道分区写入前才分配并清零；各通道在分区写入核中顺便标记有效入射端点，不再单独遍历请求写标记。标记数组由物种上下文持有并复用，配对结束后不释放；下一非空同种段重新清零并写入。
空请求通道只把有效请求总数设为零，跳过清零、扫描及分区；过滤后总数为零时也跳过分区写入。选择器先检查总数，不读取空通道遗留的分区信息。下一次有请求时正常重建计数和分区。deal 使用粒子组宿主端 extent() 获取槽位范围，空段也保持宿主端种子生成。
背景粒子按槽位置换遍历，每段在宿主端通过 std::random_device 生成随机种子，不依赖粒子随机数流。有效且位于网格内的粒子直接尝试领取请求，不再读取密度或随机跳过；票号直接对应请求编号，不增加每格请求置换。背景随机遍历不保证请求队列前缀是均匀随机子集；背景不足造成的未匹配请求直接跳过；同种入射组排除规则会减少可用背景候选，通道优先顺序也会影响哪些请求获得配对。

该模块也提供了许多元编程工具。

### 工具: 随机数

随机数模块为设备端计算提供可复现的伪随机序列。在 PSuM 中，通过为每一个粒子分配随机种子属性，可获得更好的可复现性。

#### rander 与设备端约束

`rander` 是设备端随机数生成器，使用 LCG（线性同余）算法：

```cpp
x = x * a + c;
```

由于 SYCL 设备端无法使用 `std::mt19937` 等标准库引擎（它们依赖动态内存分配和系统调用），PSuM 实现了轻量级的 `rander_core` 模板。它只依赖基本整数运算，满足设备端执行约束。

#### seed 回写机制

粒子随机种子常以 `property::random_seed` 属性表示。使用随机数的典型模式是：

```cpp
particles.for_each([&](sycl::handler& h) {
    return [=](Particle& p) {
        auto R = random::view_as_rander(get<property::random_seed>(p));
        // 使用 R() 生成随机数
        // R 析构时自动将更新的种子写回 p
    };
});
```

`view_as_rander`将生成一个`rander_wrapper` ，后者在析构时调用 `sync_seed()`，将引擎内部状态写回粒子的 `random_seed` 字段。
这确保了：
- 每次运行从相同初始种子产生相同序列（可复现性）
- 不同粒子的随机序列独立（因为初始种子不同）
- 种子状态随粒子持久化保存

#### 物理分布采样

`rand_functions.hpp` 提供了常用的物理分布采样函数：

- `RandV_Maxwell(r, T, m)`：Maxwell 速度分布
- `RandV_spherical(r)`：均匀球面方向
- `RandV_Cosine(r, axis)`：余弦分布反射方向
- `RandV_halfMaxw(r, axis, T, m)`：半空间 Maxwell 分布

这些函数接受 `rander&` 引用，在 1D/2D/3D 命名空间中分别实现。

### 序列化系统

序列化模块负责把 PSuM 中的状态保存为文件，并在下一次运行时恢复。
它承担两类任务：

- 将基础数组、Eigen 对象、STL 容器、tagged_struct、device 容器保存到 MAS 文件。
- 用 `object_manager` 和 `json_loader` 管理模拟状态和配置参数。

#### mas 二进制格式

MAS 是 Multi-Array-Set 的缩写。
一个 MAS 文件由多个数据块顺序追加组成，每个数据块包含：

```text
magic string: #MAS!#
key length / type length / shape length
key string
type string
shape array
data bytes
```

`mas_file::readHead()` 会从文件头到文件尾扫描所有 block，只读取元信息并记录每个 block 在文件中的位置。
真正读取数据时，再根据 block 位置跳转并读出内容。
这样可以避免打开文件时把所有大数组一次性读入内存。

MAS 默认不允许同名数据块重复写入。
如果需要替换数据，`replaceData()` 会先调用 `smashData()`。
`smashData()` 并不移动文件内容，而是把旧 block 的 key 第一个字符改成 `!`，使后续扫描时忽略它。
这种设计让替换操作接近 O(1)，代价是文件中可能留下被屏蔽的旧数据块。

#### 泛型 save/load

序列化接口以重载的 `save(fp, key, obj)` 和 `load(fp, key, obj)` 为中心。
不同类型的支持分散在几个文件中：

- `foundation.hpp`：基础类型和类型擦除指针。
- `container_sl.hpp`：`std::vector`、`device_vector` 等容器。
- `tag_sl.hpp`：`tagged_struct` 的按标签序列化。
- `object_manager.hpp`：对象级管理和批量保存。

`tagged_struct` 的序列化特别依赖 tag 系统。
保存时按标签名写入字段；读取时按标签名匹配，而不是依赖字段在 tuple 中的顺序。
不同字段对应的数据可以在 mas_file 中相当容易地找到。

`device_vector` 和 `particle_group` 序列化时会把设备端数据拷贝回主机侧再写入文件。
文档和接口中都强调：不同硬件上的 device 容器数据不应被视为可无条件交换，因为文件中也会记录设备相关信息。

#### object_manager

`object_manager` 面向“断点续跑”和“统一管理模拟状态”。
它内部维护：

- `objects_`：`std::unordered_map<std::string, std::any>`，保存已经注册的对象。
- `savers_`：每个对象对应的保存函数。
- `file_`：可选的已有 MAS 文件，用于恢复初始状态。

用户通过 `mng.obj<T>(name)` 取得对象引用。
如果对象已经在内存中，就返回已有对象；如果没有，就默认构造一个对象，并尝试从关联的 MAS 文件中读取同名同类型数据。
保存时，`object_manager::save()` 会遍历所有注册过的 saver，把对象写入新的 MAS 文件。

为了避免覆盖正在读取的断点文件，`object_manager` 会检查保存路径。
如果当前 manager 关联了一个已有文件，保存目标必须是另一个合法路径。
这可以减少“读取旧状态时顺手把旧状态覆盖掉”的风险。

#### json_loader

`json_loader` 继承自 `object_manager`，用于把 JSON 配置转成可查询的参数表。
`key_condition` 提供了带前缀的查询方式，使用户可以写出类似 `loader["solver"].obj<double>("dt")` 的访问逻辑。

`json_loader` 只承担配置读取，不承担任意 JSON 对象建模。
