language: [English](CONTRIBUTING.md) | 中文

# 贡献指南

PSuM 是一个持续进行中的研究项目，目前由北航 JLPP 团队（Joint Laboratory of Plasma and Propulsion）主导开发。

我们非常期待来自社区的反馈与讨论。

PSuM 当前最有帮助的参与方式是提交 **Issue**——报告 Bug、提出功能建议、讨论设计思路，这些都能帮助项目改进。

**我们暂时还不接收未经沟通的 Pull Request。** 如果你有一个具体的改动想法，请先开一个 Issue 讨论，让我们看看是否合适、以及如何实现。

本文档说明如何搭建开发环境并运行测试。无论你是否打算回馈贡献，只要你在基于 PSuM 构建或研究它，这些内容都有用。文档同时也记录了维护者遵循的约定，供参考。

PSuM 是一个面向 [PIC（Particle-in-Cell）](docs/pic.md) 方法的 C++20/SYCL 框架。它以头文件为主，外加两个需要编译的组件（场求解器后端和 `pypsum` Python 绑定）。背景知识可阅读[编程指南](docs/programming.md)。

## 项目结构

| 路径 | 用途 |
|------|------|
| `src/<module>/` | 模块实现（头文件）。每个目录对应一个 `psum::<module>` 命名空间。 |
| `include/psum/<module>.hpp` | 每个模块的公共聚合头文件。 |
| `test/<module>/makefile` | 各模块的测试。 |
| `application/` | 完整的物理应用（benchmark 算例）。 |
| `example/` | 教学示例（Tour 系列）。 |
| `pypsum/` | Python 互操作 / 序列化模块。 |
| `third_party/` | 内嵌第三方库：`json.hpp`（MIT）。 |

## 开发环境配置

构建系统基于 Make。编译前必须先配置环境。
依赖：g++（≥11）、AdaptiveCpp（`acpp`）、Eigen，可选 CUDA/UMFPACK。Python 组件另需 python3-venv（Debian/Ubuntu）或 virtualenv——构建脚本会自动创建虚拟环境。
完整说明见[依赖项安装](docs/dependency.md)和[环境配置与构建](docs/building.md)。
仓库内提供了 env_scan.sh 这样的脚本来辅助管理环境变量。

```bash
bash env_scan.sh                 # 自动探测依赖，生成 env_load.sh
source env_load.sh               # 设置 PATH、CPATH、LIBRARY_PATH、LD_LIBRARY_PATH
bash build.sh                    # 编译场求解器后端和 pypsum，并运行基础测试
```

- 在 `config.mk.local`（已被 gitignore）中覆盖默认配置，不要直接编辑 `config.mk`。
- 没有 CUDA？`echo "USE_CUDA := 0" > config.mk.local`
- `env_load.sh` 是自动生成且被 gitignore 的。依赖变化后须重新生成。

## 运行测试

```bash
cd test && bash runCheck.sh                    # 冒烟测试：编译并运行 test/ 下所有目标
bash test/runCheck.sh ../application            # 也可以对 application/ 运行
cd test/field_solver && make testpsolver2d      # 单个测试目标
```

- `runCheck.sh` 的环境变量：`TIMEOUT`（默认 1s）、`JOBS`（默认 8）、`SKIP_DIRS`、`RUNCHECK_STRICT`（默认 0；设为 `1` 时超时计为失败）、`RUNCHECK_SKIP_TARGETS`（逗号分隔的 `目录/目标名` 跳过清单，用于长测试分流）、`RUNCHECK_SKIP_FILE`（跳过清单文件路径，每行一条 `目录/目标名`；忽略 `#` 注释与空行；文件不存在则直接报错）。
- 大面积失败（>50%）几乎总是因为缺少环境——先运行 `source env_load.sh`。
- 提交必需的数据文件（`.plt`、`.mas`、`.csv` 等）：`.gitignore` 全局忽略这些格式。
  请放在 `reference_results/` 或 `necessary_input/` 下（两者已豁免），或在 `.gitignore`
  中为你的路径添加显式 `!` 例外行。

在 `test/<module>/` 下编写新的测试目标时，注意 `runCheck.sh` 会解析 `make -B -n` 的 dry-run 输出，因此：

1. recipe 的第一行必须是编译器命令（前面不能有 `@echo`）。
2. 编译命令行必须有显式的 `-o <output_file>`。
3. 目标名 `all`、`clean`、`test_cov`、`py_test` 会被过滤掉——不要用它们作为真正的测试。聚合目标（如 `test_multigrid`）须加入同一排除名单，否则 runCheck 会重复计数其首个依赖。
4. 不要在 `-o` 之后使用自动变量（`$@`、`$<`）——dry-run 不会展开它们。

## 代码风格

PSuM 对 C++ 默认习惯有一组刻意的偏离。写代码前请完整阅读 [`docs/coding-style.md`](docs/coding-style.md)。要点：

- **大部分情况：snake_case**：文件、类、函数、命名空间。
- **成员变量和私有方法**：尾部下划线（`data_`、`set_size_()`）。
- **枚举值**：小驼峰 camelCase（`cellCentered`）。
- **与类型别名冲突的模板参数**：前导下划线（`_Dimension`、`_Scalar`）。
- **头文件保护符**：`PSUM_<DIR>_<FILE>_HPP`（如 `PSUM_TAG_FOUNDATION_HPP`）。
- **文件→命名空间映射**：`src/<module>/<file>.hpp` → `namespace psum { namespace <module> { ... } }`。
- 不要使用 `using namespace sycl;`。

## 报告 Issue

当前最好的贡献方式是提交 Issue：

- **Bug 报告**——描述你做了什么、期望发生什么、实际发生了什么。附上你运行的命令、环境信息（编译器、GPU、操作系统）以及任何错误输出。一个最小的复现样例会非常有帮助。
- **功能建议**——说明使用场景和你想达到的目的，而不仅仅是你心里的那个解决方案。
- **设计讨论**——对于较大的想法，Issue 是在动手之前讨论思路的好地方。

如果你在考虑一个代码改动，请先开 Issue 讨论。这有助于避免重复劳动，并确保改动符合项目方向。

## 改动约定

改动代码请遵循下面的约定。这些约定同时也记录了维护者自身的工作方式，供参考。

- 为新功能**编写测试**，放在 `test/<module>/` 下，遵循现有的各模块 makefile 约定。
- **提交前在本地跑测试门禁**：
  ```bash
  cd test && bash runCheck.sh
  ```
  默认是快层冒烟（短超时，长测试按 `test/rcheck_skip_targets.list` 跳过），
  必须零失败。若改动涉及长时或重负载目标，请用
  `TIMEOUT=20 RUNCHECK_STRICT=1 bash runCheck.sh` 或直接运行对应目标另行复核。
- **保持改动聚焦。** 一次贡献一个逻辑改动，能让 review 更快、历史更干净。
- **匹配周围代码。** 先读几个邻近文件，遵循它们的命名、注释密度和惯用法。避免重新格式化无关代码。

本项目没有贡献者许可协议（CLA）。任何被接受的贡献都按项目的 [MIT 协议](LICENSE)授权。