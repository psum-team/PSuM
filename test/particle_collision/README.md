# 碰撞模块测试 / Collision module tests

在本目录执行 `make all` 运行十一项回归，或用表中的目标单独运行。两个子目录也接受各自的目标。构建需要 AdaptiveCpp、C++20 和 Eigen，可用 `CPPFLAGS=-I/usr/include/eigen3` 指定 Eigen 路径。

Run `make all` from this directory to execute all eleven regressions, or select an individual target from the table. Each subdirectory also accepts its own targets. Building requires AdaptiveCpp, C++20 and Eigen. Set `CPPFLAGS=-I/usr/include/eigen3` to specify the Eigen include path.

| 测试目录 | 目标 | 验证内容 |
|---|---|---|
| [库仑碰撞](coulomb_collision/) | test9 | 散射、守恒、网格外粒子、德拜长度、低速上限与解析散射衰减 |
| 库仑碰撞 | test_coulomb_pool | 混池配对频率、端点及真实物种参数 |
| 库仑碰撞 | test_coulomb_relaxation | 双电子与 Spitzer 电子–离子解析温度弛豫 |
| [MCC / DPMCC](mcc_dpmcc/) | test6 | 物种上下文、删除权限与反应端点 |
| MCC / DPMCC | test_dpmcc_stages | 阶段顺序、端点互斥、产物容量与回填 |
| MCC / DPMCC | test_dpmcc_requests | 请求生成、概率、存储与配对 |
| MCC / DPMCC | test_reaction_network | 竞争反应、通道拆分与产物反馈 |
| MCC / DPMCC | test7 | 氮气碰撞守恒 |
| MCC / DPMCC | test10 | 轴对称网格投影与球面反射 |
| MCC / DPMCC | test_frozen_beams | 氮气冻结双束配对、碰撞率与消融对照 |
| MCC / DPMCC | test_n2_relaxation | 氮气速度分布与温度弛豫 |

| Test directory | Target | What it checks |
|---|---|---|
| [Coulomb collisions](coulomb_collision/) | test9 | Scattering, conservation, out-of-grid particles, Debye length, the low-speed cross-section cap and analytical scattering decay |
| Coulomb collisions | test_coulomb_pool | Pairing frequencies in mixed pools, endpoint identities and physical species parameters |
| Coulomb collisions | test_coulomb_relaxation | Two-electron-population and Spitzer electron–ion temperature relaxation against theory |
| [MCC / DPMCC](mcc_dpmcc/) | test6 | Species contexts, erasure permissions and reacting endpoints |
| MCC / DPMCC | test_dpmcc_stages | Stage order, exclusive endpoints, product capacity and insertion |
| MCC / DPMCC | test_dpmcc_requests | Request generation, probabilities, storage and pairing |
| MCC / DPMCC | test_reaction_network | Competing reactions, channel splitting and subsequent reactions of products |
| MCC / DPMCC | test7 | Conservation in nitrogen collisions |
| MCC / DPMCC | test10 | Axisymmetric grid projection and reflection from a sphere |
| MCC / DPMCC | test_frozen_beams | Frozen nitrogen beam pairing, collision rates and permutation ablation tests |
| MCC / DPMCC | test_n2_relaxation | Nitrogen velocity distributions and temperature relaxation |

`make test_coulomb_relaxation COULOMB_RELAXATION_ARGS=--full` 运行论文中的两个完整解析算例，曲线写入 `coulomb_collision/build/`。双电子独立初态的 Maxwell 热流对照见 [复现脚本](coulomb_collision/maxwell_relaxation/run.sh)。

`make test_coulomb_relaxation COULOMB_RELAXATION_ARGS=--full` runs both full-length analytical comparisons used in the paper and writes the curves to `coulomb_collision/build/`. The two-electron heat-flux study compares independent initial states with Maxwellian resampling. [Reproduction script](coulomb_collision/maxwell_relaxation/run.sh)

MCC / DPMCC 的两个氮气研究入口分别为 `mcc_dpmcc/build/test_frozen_beams --pair-order 新输出目录` 和 `mcc_dpmcc/build/test_n2_relaxation --study 新输出目录`。冻结双束的 Python 工具用于碰撞率统计、逐对分布和置换消融，依赖 NumPy 和 SciPy。两类测试的历史实验、可执行文件与输出均保存在各自 Git 忽略的 `build/` 中。

The two extended nitrogen studies run with `mcc_dpmcc/build/test_frozen_beams --pair-order NEW_OUTPUT_DIR` and `mcc_dpmcc/build/test_n2_relaxation --study NEW_OUTPUT_DIR`. The frozen-beam Python tools use NumPy and SciPy for collision-rate statistics, pair distributions and permutation ablations. Historical experiments, executables and output are kept in each test group’s Git-ignored `build/` directory.
