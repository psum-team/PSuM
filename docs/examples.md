[English](examples_en.md) | 中文

## PSuM 示例

PSuM 的自带的示例程序分为两类：**tour 系列**和**benchmark 系列**。

### Tour 系列 （Quick Start）

Tour 系列的引导文档位于 `docs/psum_tour/`，详细介绍了如何从零开始编写 tour 算例。
此外，每个阶段的参考代码都放在 `example/` 下对应的子目录，便于与自己的代码比较。
它们不追求物理上的严谨和完整，而是强调**可交互性**和**直观可检验性**, 你看一眼可视化输出就能判断程序是否符合预期。

阅读 tour 文档时，建议先自己新建文件并跟随指引改动. 
example 下已经存在的文件 (如`stage_1.cpp`、`stage_2.cpp`) 是"参考答案", 不是必须一开始直接运行的入口。

- **Big Bang** (`example/big_bang/`, [详细说明](psum_tour/big_bang.md))：二维静电 PIC 入门算例。
在方形区域中放入一小团同号带电粒子，自洽电场使粒子团向外膨胀，到达边界的粒子被吸收。
文档以 6 个 stage 逐步展开：从网格和场量创建、粒子初始化与电荷沉积，到粒子运动与边界处理、Poisson 求解器接入，最后加入能量守恒诊断。

- **Neon Light** (`example/neon_light/`, [详细说明](psum_tour/neon_light.md))：二维 Poisson 复杂边界算例。
从结构化 `.plt` 位图恢复 `host_node_field2D`，用 `Dirichlet_func` 和 `interp` 把图案区域设置为内部时变 Dirichlet 边界。
文档以 5 个 stage 展开：从位图读入、Poisson 求解器建立、内部边界配置，到时变边界值和最终动画效果。

- **Telescope** (`example/telescope/`, [详细说明](psum_tour/telescope.md))：二维光线追踪算例。
模拟简化版 Maksutov 望远镜：平行光从左侧入射，经过校正器折射、主镜反射、次镜反射后会聚到焦平面。
文档以 5 个 stage 展开：从光子定义与基本运动循环、边界系统接入，到弧形边界面反射实现，最后加入折射与校正器，全程有动画输出。

- **Avalanche** (`example/avalanche/`, [详细说明](psum_tour/avalanche.md))：二维 MCC 碰撞算例。
在区域中放置高速电子，背景为高斯分布的气团；电子进入气团后发生弹性散射与电离碰撞，次级电子继续碰撞电离，形成雪崩级联。
文档以 5 个 stage 展开：从粒子初始化、场量与沉积，到 MCC 碰撞模型接入、次级粒子产生与 buffer 机制。

> 如果你有兴趣写一个新的 tour 算例：在 `example/` 下创建代码目录（含 makefile 与各 stage），在 `docs/psum_tour/` 下新建对应的中英双语引导文档，并在 `docs/examples*.md` 中加入索引。新增目录应能干净构建，并保持文档链接有效。

### Benchmark 系列

基准算例位于 `application/benchmark/` 下。它们是项目自带的基准算例，用于检验 PSuM 的功能与性能，覆盖不同求解器与物理场景。
每个算例目录附有简要说明及其对比文献（见该目录 README）。

- **CCP** (`application/benchmark/CCP/`)：容性耦合等离子体（1D），He 放电 + MCC，RF 偏压 450 V / 13.56 MHz，含文献对照。
- **DSMC** (`application/benchmark/DSMC/`)：稀薄气体 DSMC 算例，N₂ 超声速射流撞击平板，VHS 碰撞模型 + 内能交换。
- **EDI** (`application/benchmark/EDI/`)：ExB 漂移放电（2D），模拟霍尔推进器通道内电子/离子输运，含磁场剖面与电离源项。
- **TSI** (`application/benchmark/TSI/`)：双流不稳定性（1D），经典静电动理学基准，用于检验能量守恒与相空间演化。
- **sheath** (`application/benchmark/sheath/`)：等离子体鞘层（1D），偏压壁面 12.5 V, 含文献对照。
- **particle-impulse-integration** (`application/benchmark/particle-impulse-integration/`)：粒子脉冲积分；自由分子流稳态快速算法，与高精度参考解对照。
