[English](neon_light_en.md) | 中文

## Neon Light

本文档将展示一个二维 Poisson 求解器的复杂边界算例。

算例从一个结构化 `.plt` 位图文件中读入图案，并把值为 1 的区域设置成内部 Dirichlet 边界区域。
这个定值区域不是一个常数，而是随空间和时间快速变化的函数；最终输出会像一块闪烁的霓虹灯牌。

![复杂边界条件泊松求解](phi_step.gif)

使用到的模块, 特性或典型用法:
- 外部 plt 文件的读取
- 二维泊松方程的复杂边界设置

### 准备工作

在 `example/neon_light/` 下新建 `neon_light.cpp`，然后按照每个 stage 逐步修改这个文件。
`example/neon_light/stage_1.cpp` 到 `stage_5.cpp` 是每个阶段结束时的参考答案，用来和自己的代码对照；不必当作必须运行的程序。

输入位图文件位于：

```text
docs/psum_tour/psum_bitmap.plt
```

这是 100 x 100 的 node field，变量名为 `n`，是一个二值化图案。

推荐从项目根目录进入算例目录，并加载编译环境：

```bash
cd example/neon_light
source ../../env_load.sh
```

这里的`output/` 里可能残留旧的 `.plt`、`.png` 或 `.gif` 文件.
可以先手动清空 `output/`。

### Stage 1：读取位图并恢复 node field

在 `neon_light.cpp` 中，先 include 必要头文件并使用 PSuM 的 prelude：

```cpp
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <psum/psum.hpp>

using namespace std;
using namespace psum::prelude;
```

接下来实现一个小函数，把 PSuM `.plot` 输出的 point 型 `.plt` 文件恢复成 `host_node_field2D<double>`：

```cpp
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
```

这里需要注意：
- 二维 plt 文件开头的 `zone i` 对应第二个坐标方向的节点数，`zone j` 对应第一个坐标方向的节点数，所以恢复 `grid2D` 时使用 `{nj - 1, ni - 1}`。

在 `main` 中读入位图并立刻输出，先确认读取方向和图案都正确：

```cpp
int main() {
    const string bitmap_file = "../../docs/psum_tour/psum_bitmap.plt";
    auto mask = read_node_field_plt(bitmap_file);
    mask.plot("output/mask.plt", "mask", 0.0);
    return 0;
}
```

makefile 中为这个阶段创建 `output/`，运行 `neon_light.cpp`，并把 `.plt` 转成图片：

```makefile
-include ../../config.mk

INCLUDES = -I ../../include -I ../../src

neon_light: neon_light.cpp
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o neon_light neon_light.cpp $(LIBS) $(LDFLAGS)
	mkdir -p output
	./neon_light
	python3 ../pltview.py output/mask.plt

clean:
	rm -f neon_light
	rm -rf output
```

(这些命令已经存在于 example/neon_light/makefile 中, 只需要取消注释即可)

后续每个阶段都这样编译和运行：

```bash
make neon_light
```

运行成功后，应看到 `output/mask.plt`(用 tecplot 打开) 和 `output/mask.png`(可以直接看)。

这一阶段你只需要知道：

- `host_node_field2D<double>` 是 host 端 node-centered 二维场。
- `.plt` 文件中的 `x y n` 数据可以恢复成 field 的 content。
- `mask.plot(...)` 用来快速检查读入结果。

这一阶段结束时的代码状态可以参考 `example/neon_light/stage_1.cpp`。

### Stage 2：创建 Poisson 网格并求解一次

从 Stage 1 的 `neon_light.cpp` 继续修改：

- 增加 `using namespace psum::field_solver::boundary_creator;`。
- 从 `mask.getGrid()` 继承物理范围，创建 200 x 200 的 Poisson 网格。
- 配置四周为零的 Dirichlet 边界。
- 临时把 mask 区域作为 source，求解一次并输出结果。

Poisson 网格可以和输入位图网格分辨率不同。这里使用更细的 200 x 200 网格，但物理范围继承自位图：

```cpp
const auto& mask_grid = mask.getGrid();
grid2D grid(
    {mask_grid.lowerBound<0>(), mask_grid.lowerBound<1>()},
    {mask_grid.upperBound<0>(), mask_grid.upperBound<1>()},
    {200, 200}
);
```

创建并初始化求解器。暂时把四边设为 0 定值边界：

```cpp
Poisson_solver_2d solver;
solver.init(
    grid,
    Poisson_solver_2d::Cartesian,
    {
        Dirichlet_line(grid, boundary_direction_2d::N) = 0.0,
        Dirichlet_line(grid, boundary_direction_2d::S) = 0.0,
        Dirichlet_line(grid, boundary_direction_2d::E) = 0.0,
        Dirichlet_line(grid, boundary_direction_2d::W) = 0.0
    }
);
```

为了先确认求解器和输出流程可用，可以把图案区域设置成 source：

```cpp
host_node_field2D<double> phi(grid);
host_node_field2D<double> source(grid);
source.for_each([&](size_t, double& rho, const auto& pos) {
    rho = interp(host_node_field2D<double>::Position{pos[0], pos[1]}, mask) > 0.5 ? 10.0 : 0.0;
});

solver.solve(phi.data(), source.data());
phi.plot("output/phi.plt", "phi", 0.0);
```

从这一阶段开始需要链接 Poisson solver backend，所以把 makefile 中的编译行改成：

```makefile
	$(ACPP) $(COMMON_FLAGS) $(INCLUDES) -o neon_light neon_light.cpp $(USE_BACKENDS) $(LIBS) $(LDFLAGS)
```

如果编译时报找不到 Poisson 相关实现，优先确认 makefile 中已经加入了 `$(USE_BACKENDS)`。

运行：

```bash
make neon_light
```

运行成功后，应看到 `output/phi.plt` 和 `output/phi.png`。
与 mask 输出相比, phi 输出看起来更加柔和, 甚至看不清图案的轮廓. 这是泊松方程的性质决定的。

这一阶段你只需要知道：

- Poisson 网格的物理范围可以来自已有 field。
- `interp(..., mask)` 可以在不同分辨率网格之间取值。
- `solver.solve(phi.data(), source.data())` 会把求解结果写入 `phi`。

这一阶段结束时的代码状态可以参考 `example/neon_light/stage_2.cpp`。

### Stage 3：把图案变成内部 Dirichlet 边界

从 Stage 2 的 `neon_light.cpp` 继续修改：

- 不再把图案作为 source。
- 使用 `Dirichlet_func` 把图案区域设置成内部 Dirichlet 边界。

内部边界的选择函数直接写在 `solver.init(...)` 中, 跟在原本的西向边界条件设置之后：

```cpp
        Dirichlet_line(grid, boundary_direction_2d::W) = 0.0,   // <-- 多了个逗号
        Dirichlet_func(grid, [&](double x, double y) {
            return interp(host_node_field2D<double>::Position{x, y}, mask) > 0.01;
        }) = 1.5
```

这里没有手写插值工具函数，而是直接使用 PSuM field 模块已有的 `interp`。
线性插值会在 0-1 边缘的区域产生介于 0 和 1 之间的中间值，因此用 `0.01` 作为阈值。

这一阶段 source 设为零，只看边界条件产生的电势：

```cpp
host_node_field2D<double> phi(grid);
host_node_field2D<double> source(grid);
source.setZero();

solver.solve(phi.data(), source.data());
phi.plot("output/phi.plt", "phi", 0.0);
```

运行：

```bash
make neon_light
```

运行成功后，应看到 `output/phi.plt` 和 `output/phi.png` 有所变化。
与上一阶段的输出相比, 整个图案区域看起来都在"发光".

这一阶段你只需要知道：

- `Dirichlet_func` 可以用任意布尔函数选择内部定值区域。
- mask 和 Poisson 网格不必一致，命中判断由 `interp` 完成。
- 多个边界条件可以放在同一个 `solver.init(...)` 的列表中。

这一阶段结束时的代码状态可以参考 `example/neon_light/stage_3.cpp`。

### Stage 4：让内部边界值随时间变化

从 Stage 3 的 `neon_light.cpp` 继续修改：

- 把内部 Dirichlet 值从常数改成 lambda。
- 增加时间变量 `t`。
- 增加循环，反复求解并输出 `phi_step_*.plt`。
- makefile 中把 `phi_step_*.plt` 合成动画。

在定义求解器前, 先定义一个随时间和空间快速变化的电势函数：

```cpp
double t = 0.0;
auto voltage = [&](double time, double x, double y) {
    double high_freq = sin(3000.0 * (x + y) + 100.0 * time) + 1;
    double low_freq = cos(30.0 * (x - y) - 10.0 * time) + 1;
    return (high_freq * 0.2 + low_freq) * cos(60 * y - 3);
};
```

然后把内部边界的赋值改成 lambda：

```cpp
Dirichlet_func(grid, [&](double x, double y) {
    return interp(host_node_field2D<double>::Position{x, y}, mask) > 0.01;
}) = [&](const auto& pos) {
    return voltage(t, pos[0], pos[1]);
},
```

`solver.init(...)` 只需要调用一次。
后续循环中更新 `t`，再调用 `solver.solve(...)`：

```cpp
int steps = 80;
double dt = 0.02;
for (int step = 0; step <= steps; ++step) {
    t = step * dt;
    source.setZero();
    solver.solve(phi.data(), source.data());
    phi.plot("output/phi_step_" + to_string(step) + ".plt", "phi", t);
    cout << "step = " << step << ", t = " << t << endl;
}
```

运行：

```bash
make neon_light
```

同时把 makefile 中的绘图命令调整为：

```makefile
	python3 ../pltview.py output/mask.plt
	python3 ../pltview.py phi_step output
```

运行成功后，应看到终端打印 `step = ...`，并生成 `output/phi_step.gif`。

这一阶段你只需要知道：

- 边界值 lambda 捕获变量 `t`，所以每一步求解时都能得到不同的定值边界。
- 动画中的动态效果来自时变 Dirichlet 边界。

这一阶段结束时的代码状态可以参考 `example/neon_light/stage_4.cpp`。

### Stage 5：增加边缘弱电势并完成霓虹灯效果

从 Stage 4 的 `neon_light.cpp` 继续修改：

- 叠加部分选中的 Dirichlet 边界。
- 利用线性插值得到的中间值添加一个较弱的边缘 Dirichlet 边界。

即使已经设置了边界条件, 还可以用后设置的覆盖先设置的. 在原本的北向边界后添加：
```cpp
        Dirichlet_line(grid, boundary_direction_2d::N) = 0.0,
        Dirichlet_line(grid, boundary_direction_2d::N,
            {grid.lowerBound<0>() + 0.3 * grid.span<0>(),
             grid.upperBound<0>() - 0.3 * grid.span<0>()}) = 2.5, // 在北向边界的一小段设置了更高的定值
```

函数型边界条件也可以叠加设置。线性插值`interp(..., mask)` 会在 mask 的边缘得到 `0` 到 `1` 之间的值, 选中这一区域并放在原本设置的后面:

```cpp
Dirichlet_func(grid, [&](double x, double y) {
    return interp(host_node_field2D<double>::Position{x, y}, mask) > 0.01;  // <-- 原本的边界条件
}) = [&](const auto& pos) {
    return voltage(t, pos[0], pos[1]);
},                                                                          // <-- 多了个逗号
Dirichlet_func(grid, [&](double x, double y) {
    double mask_val = interp(host_node_field2D<double>::Position{x, y}, mask);
    return mask_val > 0.01 && mask_val < 0.99;                          // <-- 选中mask的边缘
}) = [&](const auto& pos) {
    return voltage(t, pos[0], pos[1]) * 0.5;                            // <-- 使用不同的定值
}
```

运行：

```bash
make neon_light
```

运行成功后，最终输出包括：

```text
output/mask.png
output/phi_step.gif
```

应当能看到相当清晰的图案轮廓，以及闪烁效果。

这一阶段你只需要知道：

- 边界条件可以叠加设置, 后设置的覆盖先设置的。
- 整个例子的关键 API 路线是 `read_node_field_plt`、`interp`、`Dirichlet_func` 和 `Poisson_solver_2d`。

这一阶段结束时的代码状态可以参考 `example/neon_light/stage_5.cpp`。

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
- `No rule to make target 'neon_light'`：当前目录不对，或者 makefile 中还没有添加 `neon_light` 目标。
- `No rule to make target 'neon_light.cpp'`：已经有 `neon_light` 目标，但还没有在当前目录新建 `neon_light.cpp`。
- 没有生成 `output/...` 文件：确认 makefile 里有 `mkdir -p output`，并且程序是在 `example/neon_light/` 下运行。
- `pltview.py` 没有打印 `wrote ...`：说明绘图脚本没有成功跑完，往上看终端中的 Python 报错。
- 链接时报找不到 backend `.o` 文件：先构建 field solver backend，或者检查 `config.mk.local` 中启用的后端是否和本机环境匹配。
- 看到 `AdaptiveCpp Warning`：通常是运行时 JIT 编译提示，不一定是程序错误；只要程序继续输出并生成文件即可。
