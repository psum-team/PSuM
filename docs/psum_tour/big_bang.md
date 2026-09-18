[English](big_bang_en.md) | 中文

## Big Bang

本文档将展示一个二维静电 PIC 小算例的开发过程。

算例在 `[-1, 1] x [-1, 1]` 的方形区域中放入一小团同号带电粒子。
粒子先集中在原点附近，随后在自洽电场作用下相互排斥并向外膨胀；到达边界的粒子会被吸收。
这里例子将包含静电 PIC 的主要流程，但不包含碰撞、磁场、复杂几何边界等。

![同号电荷相互排斥](rho_step.gif)

使用到的模块, 特性或典型用法:
- 粒子定义和粒子容器
- 粒子与场的交互
- 泊松方程求解

### 准备工作

在 `example/big_bang/` 下新建 `big_bang.cpp`，然后按照每个 stage 逐步修改这个文件。
`example/big_bang/stage_1.cpp` 到 `stage_6.cpp` 是每个阶段结束时的参考答案，用来和自己的代码对照；不必当作必须运行的程序。

推荐从项目根目录进入算例目录，并加载编译环境：

```bash
cd example/big_bang
source ../../env_load.sh
```

这里的`output/` 里可能残留旧的 `.plt`、`.png` 或 `.gif` 文件。
可以先手动清空 `output/`。

### Stage 1：创建网格和场量

在 `big_bang.cpp` 中，首先 include 必要的头文件并 using namespace psum::prelude。
```cpp
#include <iostream>
#include <sycl/sycl.hpp>
#include <Eigen/Core>
#include <psum/psum.hpp>
using namespace std;
using namespace psum::prelude;
```
添加粒子类型定义：

```cpp
using Particle = tagged_struct<
    tag_bind<property::position, Eigen::RowVector2d>,  // 2D 位置
    tag_bind<property::velocity, Eigen::RowVector2d>,  // 2D 速度
    tag_bind<property::charge, double>,                // 电荷量
    tag_bind<property::mass, double>                 // 粒子质量
>;
```
在 main 函数中添加网格定义：
```cpp
int main() {
    grid2D grid({-1.0, -1.0}, {1.0, 1.0}, {256, 256});
    sycl::queue q{sycl::default_selector_v};

    // 可以看到设备信息
    cout << "device: " << q.get_device().get_info<sycl::info::device::name>() << endl;
    
    // 节点场量用于存储电势和电荷密度（device端，可在kernel中使用）
    node_field2D<double> phi(q, grid);      // 电势
    node_field2D<double> rho(q, grid);      // 电荷密度

    return 0;
}
```

编写 makefile 文件：
```makefile
-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(LIBS) $(LDFLAGS)
	./bb
```
(这些命令已经存在于 example/big_bang/makefile 中, 只需要取消注释即可)

后续每个阶段都这样编译和运行：

```bash
make big_bang
```

编译前确认已经 source 项目根目录的 `env_load.sh`，然后运行 `make big_bang`。
如果看到 `make: acpp: No such file or directory`，通常就是还没有执行 `source ../../env_load.sh`, 或依赖/安装没有成功。

运行成功后，应该能看到设备信息被打印出来，例如 `AdaptiveCpp OpenMP host device`(如果你的设备有 gpu，也可能显示 gpu 型号)。
第一次运行时 AdaptiveCpp 可能会打印 kernel cache 的 warning；这通常只是 JIT 编译提示，不是程序失败。

这一阶段你只需要知道：
- `Particle` 定义一个粒子有哪些属性。
- `grid2D` 定义二维计算区域和网格数。
- `node_field2D` 是定义在网格节点上的场量，这里用于电势 `phi` 和电荷密度 `rho`。

这一阶段结束时的代码状态可以参考`example/big_bang/stage_1.cpp`.

### step 2: 添加粒子容器并初始化粒子和电荷统计

从 Stage 1 的 `big_bang.cpp` 继续修改：
- 在 `Particle` 定义后添加 `ParticleGroup`。
- 在 `main` 前添加 `make_particles` 和 `deposit_charge`。
- 在 `main` 中创建粒子容器，初始化粒子，沉积电荷，然后输出 `output/initial_rho.plt`。
- 把 makefile 改成先创建 `output/`，再运行程序并生成图片。

在`Particle`后定义 `particle_group` 作为粒子容器：

```cpp
using ParticleGroup = particle_group<Particle, pos_x_nan_is_invalid>;
```

这里的 `pos_x_nan_is_invalid` 会在后续边界吸收阶段使用：当粒子被标记为无效后，`particle_group` 的遍历会自动跳过它。

粒子初始化函数直接接收 `ParticleGroup&`，在函数内部生成 host 端 `vector<Particle>`，最后一次性插入粒子组：

```cpp
void make_particles(size_t count, ParticleGroup& particles) {
    rander R;
    vector<Particle> particle_vec(count);
    for (auto& p : particle_vec) {
        get<property::position>(p) = {R() - 0.5, R() - 0.5};
        auto rand_vec = RandFunction3D::RandV_Maxwell(R, 100000, 2.18e-25);
        get<property::velocity>(p) = {rand_vec.x(), rand_vec.y()};
        get<property::charge>(p) = 1.602e-19 * 1e7;
        get<property::mass>(p) = 2.18e-25 * 1e7;
    }
    particles.insert(particle_vec);
}
```

这里的位置是 `[-0.5, 0.5] x [-0.5, 0.5]` 上的均匀分布。
速度使用 PSuM 随机工具中的 `RandFunction3D::RandV_Maxwell` 生成 Maxwell 分布速度，再取 `x/y` 分量作为二维速度。
电荷和质量都乘以 `1e7`，代表许多真实粒子（宏粒子）。

接下来添加电荷沉积函数：

```cpp
void deposit_charge(ParticleGroup& particles, node_field2D<double>& rho) {
    rho.setZero();
    double cell_area = rho.getGrid().del<0>() * rho.getGrid().del<1>();
    particles.for_each([&](sycl::handler& h) {
        auto rho_acc = rho.get_access(h);
        return [=](Particle& p) {
            add_back(
                get<property::position>(p),
                get<property::charge>(p) / cell_area,
                rho_acc
            );
        };
    });
}
```

注意 `cell_area` 直接从 `rho.getGrid()` 中取得，这样 `deposit_charge` 不需要额外传入网格参数。
`rho.setZero()` 用来清空旧的沉积结果，`add_back` 将粒子电荷按插值权重加回网格节点。

在 `main` 中添加：

```cpp
ParticleGroup particles(q);
make_particles(100000, particles);
deposit_charge(particles, rho);
rho.plot("output/initial_rho.plt", "rho", 0.0);
```

调整 makefile：

```makefile
big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output # 创建output目录
	./bb
	python3 ../pltview.py output/initial_rho.plt # 将生成的plt文件转成png文件
```

运行成功后，应该能看到 `output/initial_rho.plt`，并生成 `output/initial_rho.png`。
如果没有生成 `.png`，先确认 `python3 ../pltview.py output/initial_rho.plt` 是否被执行，以及当前目录是否是 `example/big_bang/`。

这一阶段你只需要知道：
- `ParticleGroup` 是装很多粒子的容器。
- `make_particles` 先在 host 端生成粒子，再一次性插入容器。
- `deposit_charge` 把粒子的电荷按插值权重分配到网格节点上。

这一阶段结束时的代码状态可以参考 `example/big_bang/stage_2.cpp`.

### step 3：添加粒子运动和边界条件处理

从 Stage 2 的 `big_bang.cpp` 继续修改：
- 在 `deposit_charge` 后添加 `move_particles`。
- 把 `main` 中“只沉积一次并输出”的代码改成时间循环。
- makefile 改成把 `rho_step_*.plt` 合成动画。

这一阶段先不加入 Poisson 求解器，因此 `phi` 仍然是零场；但我们仍然按完整 PIC 推进的写法，从 `phi` 中得到电场。
这样下一阶段只需要让 `phi` 变成求解器算出来的电势，粒子推进函数不用再改。

在 `deposit_charge` 后添加粒子运动函数：

```cpp
void move_particles(ParticleGroup& particles, node_field2D<double>& phi, double dt) {
    particles.for_each([&](sycl::handler& h) {
        auto phi_acc = phi.get_access(h);
        return [=](Particle& p) {
            auto& position = get<property::position>(p);
            auto& velocity = get<property::velocity>(p);

            auto grad_phi = interp_diff(position, phi_acc);
            Eigen::RowVector2d electric_field(-grad_phi[0], -grad_phi[1]);
            velocity += electric_field * get<property::charge>(p) / get<property::mass>(p) * dt;
            position += velocity * dt;

            if (!phi_acc.getGrid().inGrid(position)) {
                ParticleGroup::validator::make_invalid(p);
            }
        };
    });
}
```

这里 `interp_diff(position, phi_acc)` 返回电势梯度，电场是 `-grad(phi)`。当前 `phi` 为零，所以电场为零。
粒子只按初始 Maxwell 速度飞行。
边界处理使用 `ParticleGroup::validator::make_invalid(p)`，被标记为无效的粒子在后续 `particle_group` 遍历中会被跳过。

然后把 step 2 中只沉积一次的部分改成时间循环：

```cpp
double dt = 2.0e-7;
int steps = 400;
int output_interval = 10;

for (int step = 0; step <= steps; ++step) {
    deposit_charge(particles, rho);
    if (step % output_interval == 0) {
        rho.plot("output/rho_step_" + to_string(step) + ".plt", "rho", step * dt);
        cout << "step = " << step << ", alive = " << particles.size() << endl;
    }
    move_particles(particles, phi, dt);
}
```

调整 makefile，在运行程序后用 `pltview.py` 合成动图：

```makefile
big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./bb
	python3 ../pltview.py rho_step output
```

`python3 ../pltview.py rho_step output` 会在 `output/` 中查找以 `rho_step_` 开头的 `.plt` 文件，并把它们合成动画。
运行成功后，终端会打印 `step = ...`、`alive = ...`，并生成 `output/rho_step.gif`。
这一阶段虽然 `phi` 是零场，粒子仍会按初始 Maxwell 速度飞行，所以后期少量粒子会离开边界。

这一阶段你只需要知道：
- `interp_diff(position, phi_acc)` 从电势场中计算梯度。
- 电场是 `-grad(phi)`；当前 `phi` 为零，所以还没有自洽电场加速。
- 离开网格的粒子会被标记为无效，后续遍历会跳过它们。

这一阶段结束时的代码状态可以参考 `example/big_bang/stage_3.cpp`.

### step 4：添加泊松求解器

从 Stage 3 的 `big_bang.cpp` 继续修改：
- 在 namespace 部分添加边界条件工具。
- 在创建粒子后创建 `Poisson_solver_2d`，并设置四条 Dirichlet 边界。
- 添加 host 端临时场 `phi_host` 和 `rho_host`。
- 在每次电荷沉积后求解 `phi`，并暂时保留 `if (step == 0) break;`，先只检查初始电势。
- makefile 的编译命令从这一阶段开始加入 `$(USE_BACKENDS)`。

这一阶段把 step 3 中一直为零的 `phi` 改成由电荷密度 `rho` 求解得到的电势。
为了先单独检查泊松求解器和电势输出，这一步在完成第 0 步的 `rho -> phi` 求解并输出后就退出程序。

首先在命名空间部分增加边界条件工具：

```cpp
using namespace psum::field_solver::boundary_creator;
```

在创建粒子之后，创建 Poisson 求解器，并设置四条边为定值边界：

```cpp
Poisson_solver_2d psolver;
psolver.init(
    grid,
    Poisson_solver_2d::Cartesian,
    {
        Dirichlet_line(grid, boundary_direction_2d::N) = 0,
        Dirichlet_line(grid, boundary_direction_2d::S) = 0,
        Dirichlet_line(grid, boundary_direction_2d::E) = 0,
        Dirichlet_line(grid, boundary_direction_2d::W) = 0
    }
);
```

当前使用的 native 求解后端在 host 端工作，因此还需要准备 host 端的临时场：

```cpp
host_node_field2D<double> phi_host(grid);
host_node_field2D<double> rho_host(grid);
```

然后在主循环中，在每次沉积电荷之后求解电势：

```cpp
for (int step = 0; step <= steps; ++step) {
    deposit_charge(particles, rho);
    rho_host.copy(rho.getContent().to_host());
    psolver.solve(phi_host.data(), rho_host.data());
    phi.copy(phi_host.getContent());

    if (step % output_interval == 0) {
        rho.plot("output/rho_step_" + to_string(step) + ".plt", "rho", step * dt);
        phi.plot("output/phi_step_" + to_string(step) + ".plt", "phi", step * dt);
        cout << "step = " << step << ", alive = " << particles.size() << endl;
    }
    if (step == 0) break; // 临时逻辑，只输出初始分布
    move_particles(particles, phi, dt);
}
```

这里 `rho_host.copy(rho.getContent().to_host())` 将 device 端的 `rho` 拷贝到 host；
`psolver.solve(phi_host.data(), rho_host.data())` 求解泊松方程；
`phi.copy(phi_host.getContent())` 再把 host 端电势拷贝回 device 端的 `phi`。

是否需要 host 端副本取决于求解器后端是否支持 device 端指针。
对于`cuda_sparselu_gpu`这样的 gpu 后端不需要这一步(而且会导致出错)。

调整 makefile，在运行程序后查看初始电势：

```makefile
big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./bb
	python3 ../pltview.py phi_step output
```

这里加入 `$(USE_BACKENDS)`，便于后续使用非 native 后端。
如果链接时报找不到 `registry.o`、`impls.o` 或其他 field solver backend 文件，说明求解器后端还没有构建好。
需要先按构建文档生成这些 backend 目标文件。

运行后应当得到 `output/phi_step.png`。这一阶段只输出第 0 步，所以 `pltview.py` 会生成单张图片而不是动画。电势分布应当比电荷分布更平滑，并且在四条边界处达到 0.
本教程默认使用 native 后端，所以需要 `rho_host.copy(...)` 和 `phi.copy(...)` 这两次 host/device 拷贝。
如果以后切到 GPU 后端，这部分代码可以去掉。

这一阶段你只需要知道：
- Poisson 求解器把电荷密度 `rho` 转换成电势 `phi`。
- `$(USE_BACKENDS)` 会把求解器后端链接进程序。
- 这里先在第 0 步退出，是为了单独确认电势求解和输出没有问题。

这一阶段结束时的代码状态可以参考 `example/big_bang/stage_4.cpp`.

### step 5：使用求解器计算出的电场更新粒子位置

从 Stage 4 的 `big_bang.cpp` 继续修改：
- 删除主循环中的 `if (step == 0) break;`。
- makefile 中同时生成 `rho_step` 和 `phi_step` 的动画。

step 4 已经能够从当前粒子分布求出电势，但程序在第 0 步就退出了。这一阶段要做的事情很少：只要去掉主循环中的

```cpp
if (step == 0) break;
```

即可。

调整 makefile，在运行后同时生成电荷密度和电势动画：

```makefile
big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./bb
	python3 ../pltview.py rho_step output
	python3 ../pltview.py phi_step output
```

运行后应当能看到粒子团快速膨胀，到达边界的粒子会被 `ParticleGroup::validator::make_invalid(p)` 标记为无效，因此 `alive` 数量会下降。

这一阶段你只需要知道：
- 每一步先沉积 `rho`，再求解 `phi`，最后用 `phi` 推动粒子。
- `output/rho_step.gif` 看电荷密度（对应于粒子空间分布）怎么扩散。

这一阶段结束时的代码状态可以参考 `example/big_bang/stage_5.cpp`.

### step 6：观察能量守恒

从 Stage 5 的 `big_bang.cpp` 继续修改：
- 添加 `<fstream>`。
- 在 `move_particles` 后添加 `kinetic_energy` 和 `field_energy`。
- 在 `main` 中打开 `output/energy.plt`; 创建一个可复用的 shared 变量。
- 在求解出 `phi` 后统计总能量并写入文件。
- makefile 中再增加 `python3 ../pltview.py energy output`。

step 5 已经完成了自洽的粒子推进：每一步先由粒子沉积出 `rho`，再求解 `phi`，最后用电场推动粒子。
接下来可以加入一个能量诊断，用它观察数值过程在粒子大量离开边界之前是否基本守恒。

先加入文件输出所需的`<fstream>`：

```cpp
#include <fstream>
```

粒子动能可以直接在粒子容器上并行累加。
由于所有粒子都会同时写入同一个标量，需要把这个标量放在 device 可访问的 shared memory 中，并用 `atomic_add` 做归约：

```cpp
double kinetic_energy(ParticleGroup& particles, double* energy) {
    *energy = 0.0;
    particles.for_each([&](sycl::handler& h) {
        return [=](Particle& p) {
            const auto& velocity = get<property::velocity>(p);
            atomic_add(*energy, 0.5 * get<property::mass>(p) * velocity.squaredNorm());
        };
    });
    return *energy;
}
```

场能用电荷密度与电势之积的积分(累积), 可以用 field 提供的 for_each 函数实现：

```cpp
double field_energy(node_field2D<double>& phi, node_field2D<double>& rho, double* energy) {
    *energy = 0.0;
    double cell_area = phi.getGrid().del<0>() * phi.getGrid().del<1>();
    phi.for_each([&](sycl::handler& h) {
        auto rho_acc = rho.get_access(h);
        return [=](size_t i, double& phi_value) {
            atomic_add(*energy, 0.5 * phi_value * rho_acc(i) * cell_area);
        };
    });
    return *energy;
}
```

这里使用 `phi.for_each` 的第一个参数 `i` 取得当前网格点的 index，再通过 `rho_acc(i)` 访问同一位置的电荷密度。

在 `main` 中打开能量文件，并创建一个可复用的 shared 变量：

```cpp
ofstream energy_file("output/energy.plt");
energy_file << "variables=time,relative_energy" << endl;
double initial_energy = 0.0;
double* energy = shared_variable<double>(q);
```

`energy` 在主函数中只创建一次，之后分别传给动能和场能统计函数重复使用。每次统计函数开始时都会先把 `*energy` 清零。

在主循环中，求解出 `phi` 后立刻统计总能量，能量的相对值：

```cpp
double kinetic = kinetic_energy(particles, energy);
double electric = field_energy(phi, rho, energy);
double total = kinetic + electric;
if (step == 0) {
    initial_energy = total;
}
energy_file << step * dt << "\t" << (total / initial_energy) << endl;
```

输出文件 `output/energy.plt` 只有两列：时间和 `relative_energy`。这样可以直接复用 `pltview.py` 的一维曲线绘图逻辑。

最后调整 makefile，让 `big_bang` 目标同时生成电荷密度、电势和能量误差图：

```makefile
big_bang: big_bang.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o bb big_bang.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
	mkdir -p output
	./bb
	python3 ../pltview.py rho_step output
	python3 ../pltview.py phi_step output
	python3 ../pltview.py energy output
```

运行后会得到 `output/energy.png`。在大量粒子倍被边界吸收之前，`relative_energy` 应当接近 1；
之后由于粒子不断离开计算区域，总能量会明显下降。

这一阶段你只需要知道：
- 动能来自粒子速度，场能来自 `rho` 和 `phi`。
- `atomic_add` 用来安全地把很多粒子的贡献加到同一个数上。
- `output/energy.png` 不是要求全程接近 0；粒子离开边界后，总能量会快速下降。

这一阶段结束时的代码状态可以参考 `example/big_bang/stage_6.cpp`.

### 输出和常见错误

`pltview.py` 有两种常用方式：

```bash
python3 ../pltview.py output/initial_rho.plt
python3 ../pltview.py rho_step output
```

第一种用于单个 `.plt` 文件，通常生成对应的 `.png`。
第二种用于一组文件，会在 `output/` 中查找以 `rho_step_` 开头的 `.plt` 文件并合成动画；`phi_step output` 和 `energy output` 的含义类似。

常见错误可以先这样排查：

- `make: acpp: No such file or directory`：通常是没有执行 `source ../../env_load.sh`。
- `No rule to make target 'big_bang'`：当前目录不对，或者 makefile 中还没有添加 `big_bang` 目标。
- `No rule to make target 'big_bang.cpp'`：已经有 `big_bang` 目标，但还没有在当前目录新建 `big_bang.cpp`。
- 没有生成 `output/...` 文件：确认 makefile 里有 `mkdir -p output`，并且程序是在 `example/big_bang/` 下运行。
- `pltview.py` 没有打印 `wrote ...`：说明绘图脚本没有成功跑完，往上看终端中的 Python 报错。
- 链接时报找不到 backend `.o` 文件：先构建 field solver backend，或者检查 `config.mk.local` 中启用的后端是否和本机环境匹配。
- 看到 `AdaptiveCpp Warning`：通常是运行时 JIT 编译提示，不一定是程序错误；只要程序继续输出并生成文件即可。
