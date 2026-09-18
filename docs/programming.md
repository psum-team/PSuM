[English](programming_en.md) | 中文

## PSuM 编程指南

*如果你还不熟悉 C++ 模板、lambda 和 SYCL 的基本写法，可以先读 [PSuM 中的 C++](cpp-in-psum.md)。本文只讲“如何使用 PSuM 写程序”，内部实现细节见 [框架设计](design.md)。*

本文面向已经能编译 PSuM 示例并准备写自己算例的用户。如果环境还没配置好，可优先考虑 [Docker 部署](../docker_install/readme.md)（如果你的环境可控如个人电脑，这比手动配置简单得多）。
如果 Docker 不适用，阅读 [依赖安装](dependency.md) 和 [构建系统说明](building.md) 手动配置。

### 环境确认

在写代码前，通常需要先加载项目生成的环境变量：

```bash
source env_load.sh
```

`env_load.sh` 通常由 `env_scan.sh` 生成，用来设置 AdaptiveCpp、Eigen、CUDA、UMFPACK 等依赖的路径。若要关闭 CUDA 或调整架构，应在 `config.mk.local` 中覆盖 `USE_CUDA`、`CUDA_ARCH` 等选项，而不是修改 `config.mk`。

确认环境是否基本可用，可以先运行：

```bash
cd test
bash runCheck.sh
```

*该测试通常需要几分钟时间，请耐心等待。*

或进入一个示例目录执行 `make`。`programming.md` 中的代码默认已经处在上述环境下。

### 代码入口与命名空间

大多数用户程序只需要包含主入口：

```cpp
#include <psum/psum.hpp>

using namespace psum::prelude;
```

`psum::prelude` 汇总了常用模块的命名空间，包括 tag、序列化、粒子容器、场量、边界、场求解器、碰撞、随机数和若干 SYCL 工具。
使用 `prelude` 可以让代码更精炼。

如果你希望显式控制依赖，也可以包含模块入口：

```cpp
#include <psum/field.hpp>
#include <psum/particle_container.hpp>
#include <psum/field_solver.hpp>
```

并相应地使用模块命名空间：

```cpp
using namespace psum::field;
using namespace psum::particle_container;
using namespace psum::field_solver;
```

应当优先使用 `include/psum/` 下的公共入口。
`src/` 下的头文件是项目内部组织方式，当前许多实现以 header-only 形式存在，但它们不一定都是稳定的用户入口。

### Hello PSuM!

下面是一个最小程序：创建计算设备、在设备端数组中写入数据，再拷贝回主机端。

```cpp
#include <psum/psum.hpp>
#include <iostream>

using namespace psum::prelude;

int main() {
    sycl::queue q{sycl::default_selector_v};
    std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    device_vector<double> values(q, 16);   // 第二个参数是容量 capacity，初建 size = 0
    values.resize(16);                     // 须先设定元素个数，for_each 才有作用范围
    values.for_each([&](sycl::handler& h) {
        return [=](double& x) {
            x = 3.14;
        };
    });

    auto host_values = values.to_host();
    std::cout << host_values[0] << std::endl;
}
```

`for_each` 使用 PSuM 中常见的“双层 lambda”写法：外层在主机端准备资源，内层在设备端并行执行。内层通常应使用 `[=]` 值捕获。更详细的解释见 [PSuM 中的 C++](cpp-in-psum.md)。

### 定义粒子类型

PSuM 使用 tag 系统描述粒子属性。使用 `tagged_struct`：

```cpp
using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector3d>,
    tag_bind<property::velocity, Eigen::RowVector3d>,
    tag_bind<property::weight, double>,
    tag_bind<property::random_seed, uint32_t>
>;
```

访问属性时使用 `get<Tag>(particle)`：

```cpp
Particle p;
get<property::position>(p) = Eigen::RowVector3d(0.0, 0.0, 0.0);
get<property::velocity>(p) = Eigen::RowVector3d::Zero();
get<property::weight>(p) = 1.0;
get<property::random_seed>(p) = 1;
```

常用内置 tag 包括：

- `property::position`
- `property::velocity`
- `property::acceleration`
- `property::mass`
- `property::charge`
- `property::weight`
- `property::random_seed`
- `property::species_id`
- `property::internal_energy`

自定义 tag 只需要继承 `abstract_tag` 并提供 `tag_name`：

```cpp
struct init_position : tag::foundation::abstract_tag {
    inline const static std::string tag_name = "init_position";
};
```

如果某个 tag 需要限制绑定类型，可以实现 `check<T>()`。tag 系统的编译期查找和类型验证细节属于内部机制，见 [框架设计](design.md)。

一个常见坑是局部变量名和 tag 名冲突。例如局部变量 `mass` 会遮蔽 `property::mass`。实际代码中建议保留 `property::` 前缀。

### 使用 device_vector

`device_vector<T>` 是设备端动态数组，类似 `std::vector<T>`，但数据存放在 SYCL 设备内存中。

常见构造方式：

```cpp
std::vector<Particle> host_particles(100);
device_vector<Particle> from_host(q, host_particles);

device_vector<Particle> buffer(q, 1000);  // 预留长度：1000
```

主机和设备之间的数据移动：

```cpp
auto host_copy = from_host.to_host();
from_host.copy(host_particles);
```

设备端遍历：

```cpp
from_host.for_each([&](sycl::handler& h) {
    return [=](Particle& p) {
        get<property::velocity>(p) = Eigen::RowVector3d::Zero();
    };
});
```

设备端填充新元素需要先在外层获取访问器：

```cpp
device_vector<Particle> selected(q, 1000);

from_host.for_each([&](sycl::handler& h) {
    auto out = selected.get_access(h);
    return [=](Particle& p) {
        if (get<property::position>(p).x() > 0.0) {
            out.push_back(p);
        }
    };
});
```

`push_back` 是线程安全的，但容量不会自动增长。如果写入数量超过预留容量，容器会进入 overflow 状态，后续主机端操作会尽早暴露这个错误。

### 使用 particle_group

`particle_group` 是 PSuM 中主要的粒子容器。它建立在 `device_vector` 之上，额外支持惰性删除和空间复用。

如果:
- 数据有频繁的删除需求
- 不需要设备端填充

则建议使用 `particle_group`。

定义粒子组时需要指定粒子类型和 validator：

```cpp
using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;

ParticleGroup particles(q);
particles.insert(host_particles);
```

`pos_x_nan_is_invalid` 使用 `position.x()` 是否为 NaN 判断粒子是否有效。遍历 `particle_group` 时，无效粒子会被自动跳过：

```cpp
particles.for_each([&](sycl::handler& h) {
    return [=](Particle& p) {
        if (get<property::position>(p).x() > 1.0) {
            ParticleGroup::validator::make_invalid(p);
        }
    };
});
```

删除只是标记无效，并不会立刻移动内存。后续可以使用：

```cpp
particles.compress();
particles.shrink(particles.size() * 1.2);
```

`size()` 表示有效粒子数，`capacity()` 表示已分配容量。`particle_group` 的双数组结构和压缩算法见 [框架设计](design.md)。

### 使用网格和场量

场量系统负责在规则网格上存储电荷密度、电势、电场等物理量。

创建网格：

```cpp
grid2D grid({0.0, 0.0}, {1.0, 1.0}, {64, 64});
```

常见场量有两类：

- `device_field`：设备端场量，适合在核函数中读写。
- `host_field`：主机端场量，在 host 端直接操作会更方便。

对于粒子-场交互的场景，通常都应使用`device_field` 。

示意：

```cpp
node_field2D<double> rho(q, grid);
node_field2D<double> phi(q, grid);
```

使用时关注三件事：维度、位置类型（节点或单元中心）、数值类型。

### 插值与沉积

PIC 中粒子和场之间有两个基本操作：

- interp：从网格场插值到粒子位置。
- add_back：把粒子权重沉积回网格。

PSuM 提供的核心函数包括：

```cpp
auto value = interp(pos, field_acc);
auto grad = interp_diff(pos, field_acc);
add_back(pos, weight, field_acc);
```

这些函数通常在粒子遍历内使用：

```cpp
particles.for_each([&](sycl::handler& h) {
    auto rho_acc = rho.get_access(h);
    return [=](Particle& p) {
        auto pos = get<property::position>(p);
        auto w = get<property::weight>(p);
        add_back(pos, w, rho_acc);  // 使用 rho_acc 而非 rho!
    };
});
```

不同场量支持的插值/沉积方式有差异。

如果只需要最近邻版本，可以使用 `interp_nearest` 和 `add_back_nearest`。插值核、张量积展开和原子沉积的实现见 [框架设计](design.md)。

### 求解 Poisson 方程

典型静电 PIC 程序可以概括为：

```text
for each time step:
    rho.setZero()
    particles.for_each(deposit charge to rho)
    solver.solve(phi, rho)
    particles.for_each(push particles by -interp_diff(phi); apply boundary conditions)
    execute collisions
```

粒子边界有时可以相当简单并直接写在推动函数中; 粒子碰撞有时也可忽略, 但场求解器是必须的. Poisson 求解器按维度分为：

- `Poisson_solver_1d`
- `Poisson_solver_2d`
- `Poisson_solver_3d`

使用流程通常是：

```text
1. 创建 grid
2. 创建固定边界或混合边界列表
3. solver.init(grid, boundaries, ...)
4. 每个时间步调用 solver.solve(phi, source)
```

固定边界（Dirichlet）用于直接指定电势值；混合边界（Robin）用于包含函数值和法向导数的边界条件。
求解器可以使用默认 `native` 后端，也可以在初始化时选择其他后端。

矩阵装配、ghost 节点消元和 backend 插件机制属于内部实现，见 [框架设计](design.md)。

### 处理粒子边界

粒子边界系统处理粒子轨迹和几何表面的相交。使用思路是：

```text
1. 加载或生成三角面片几何
2. 为三角面片指定材料类型
3. 创建 boundary_router
4. 在粒子推进后，用旧位置和新位置调用 router_acc.deal(...)
```

核函数中的典型调用形式是：

```cpp
particles.for_each([&](sycl::handler& h) {
    auto router_acc = router.get_access(h);
    return [=](Particle& p) {
        auto old_pos = get<property::position>(p);
        // update position here
        auto new_pos = get<property::position>(p);
        router_acc.deal(
            old_pos.x(), old_pos.y(), old_pos.z(),
            new_pos.x(), new_pos.y(), new_pos.z(),
            p
        );
    };
});
```

内置材料行为包括吸收、镜面反射、漫反射和 Maxwell 反射。许多反射类行为还需要粒子具有 `random_seed` 属性。

### 处理碰撞

PSuM 的碰撞模块主要包括：

- MCC：粒子与背景物种的 Monte Carlo Collision。
- DPMCC：需要显式配对的延迟配对碰撞。

用户侧需要理解三个概念：

- 碰撞模型：描述有哪些入射物种和碰撞通道。
- 物种上下文：绑定粒子组、背景密度场和新粒子 buffer。
- 执行函数：在时间步中调用碰撞模型，并在结束后清理 buffer。

简化流程：

```text
prepare species contexts
execute_mcc_model<Model>(ctx, dt)
execute_all_clean_buffers<SpeciesTuple>(ctx)
```

自定义碰撞截面、通道选择和配对算法通常较长，不适合放在主教程里；完整写法可参考[示例总览](examples.md)与 `application/` 下的具体应用。

### 保存、恢复与配置

基础保存/读取使用 `mas_file`：

```cpp
mas_file fp("state.mas", mas_file::replaceMode);
save(fp, "particles", particles);
load(fp, "particles", particles);
```

需要管理多个对象时，可以使用 `object_manager`。它适合断点续跑：使用的对象从 object_manager 对象中注册，下一次运行从已有 MAS 文件中恢复。

配置文件可以用 `json_loader`：

```cpp
json_loader loader;
loader.load_json("config.json");
double dt = loader["solver"].obj<double>("dt");
```

`device_vector` 和 `particle_group` 的序列化会涉及设备信息，不同硬件之间不能默认认为可以无条件交换。

### MPI 支持（可选）

仓库在 `src/mpi/` 提供一个**可选的** MPI 通信层：一个薄纯 C 接口（全仓唯一 `include <mpi.h>` 的编译单元），其上为通信包装与分布式线性代数对象（分布式向量、稀疏矩阵 SpMV）。它默认**不参与** CPU-only 构建，按需编译链接；通信只走宿主内存，无 GPU-direct 路径。

理念：把 MPI 隔在一层薄接口之后，使 MPI 头与运行时都不扩散到整个代码库；它是可选能力，而非默认依赖。

最小使用流程分三步：**(1)** 用 `mpicc` 编译全仓唯一含 `mpi.h` 的 C 编译单元 `src/mpi/mpi_implement.c`；**(2)** 用 `acpp` 编译你的 C++ 代码并与 `mpi_implement.o` 链接——用到 PSuM 头文件时必须用 `acpp`（`mpicxx` 无法编译 SYCL），以 OpenMPI 为例，MPI 的头文件路径与库由 `$(mpicc --showme:compile)` / `$(mpicc --showme:link)` 提供；**(3)** 用 `mpirun -np N` 运行。

```bash
mpicc -O2 -fPIC -I src/mpi -c src/mpi/mpi_implement.c
# $(mpicc --showme:*)＝让 mpicc 只打印、不执行它平时代加的开关：
#   --showme:compile → -I<OpenMPI 头文件目录>；--showme:link → -L<…> -lmpi
acpp your_main.cpp mpi_implement.o $(mpicc --showme:compile) $(mpicc --showme:link)
mpirun -np 4 ./your_app
```

### 下一步读什么

- 想看完整示例：读 [示例](examples.md)。
- 想理解双层 lambda、模板和 concept：读 [PSuM 中的 C++](cpp-in-psum.md)。
- 想理解内部实现：读 [框架设计](design.md)。
- 想贡献代码：读 [代码风格规范](coding-style.md)。
