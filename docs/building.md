[English](building_en.md) | 中文

## 构建系统说明

### 概述

正常使用前，需完成依赖安装、配置管理、组件编译几个环节。

- 依赖安装：安装 cuda/llvm/Eigen/AdaptiveCpp 等库，参见其他文件。
- 配置管理：引入必要的环境变量并根据项目需求设置编译选项。
- 组件编译：编译需要用到的组件并简单测试。

这里主要介绍配置管理和组件编译。

---

### 文件说明

#### env_scan.sh - 依赖检测工具

**作用**: 依赖安装后，这个脚本可以生成环境变量配置文件。
- 环境变量对于 psum 中的编译几乎是必须的.

**功能**:
- 在预定义路径中搜索依赖库
- 检测以下依赖: LLVM, AdaptiveCpp, Eigen, CUDA, BLAS, UMFPACK
- 自动生成 `env_load.sh` 环境配置文件
- 为编译工具、头文件、库文件设置相应的环境变量配置
- 支持搜索深度设置（默认 3 层，可通过 `-d` 参数调整）

**使用方法**:
```bash
bash env_scan.sh           # 默认深度 3
bash env_scan.sh -d 4      # 最大搜索深度 4 层
```

**搜索优先级**:
- 1、环境变量提到的路径
- 2、用户家目录下的路径
- 3、`/usr`, `/opt`, `/usr/local`, `/data/apps`, `/share`, `/public` 等启发式搜索路径


env_scan.sh 在所有依赖良好安装、环境简单的情况下应当能够正常工作。对于复杂情况，例如集群上运行时，建议先用 module load 或使用其他配置方法，再调用 env_scan.sh。这可能改善搜索结果，因为 env_scan.sh 会优先搜索已经引入环境变量的路径。

---

#### env_load.sh - 环境加载器（自动生成）

**作用**: 设置所有依赖库的环境变量

**功能**:
- 由 `env_scan.sh` 自动生成，包含生成时间戳
- 设置以下环境变量:
  - `PATH`: 可执行文件路径
  - `CPATH`: 头文件搜索路径
  - `LIBRARY_PATH`: 编译时库搜索路径
  - `LD_LIBRARY_PATH`: 运行时库搜索路径

**使用方法**:
```bash
source env_load.sh
```

如果 env_load.sh 不可靠，应当手动设置环境变量。这个文件不在 git 的管理下。

---

#### config.mk - 默认构建配置

**作用**: 为 Makefile 提供编译器和构建选项的默认配置

**配置项**:
- 编译器设置: `CXX`, `NVCC`, `ACPP`
- 功能开关: `USE_EIGEN`, `USE_UMFPACK`, `USE_CUDA` (0=禁用, 1=启用)
- CUDA 架构: `CUDA_ARCH` (默认 sm_75)
- 后端选择: `USE_CUDA_SPARSELU_GPU`, `USE_EIGEN_SPARSELU_CPU`
- 通用编译选项: `-O3 -std=c++20`
- 库链接: CUDA, UMFPACK, Eigen 相关库

用户可以在此看到默认配置，并根据需要，**在 `config.mk.local` 中覆盖设置**。

---

#### config.mk.local - 用户本地配置（可选）

**作用**: 覆盖默认编译配置

**功能**:
- 该文件优先级高于 `config.mk`，用户可以根据自己的需求覆盖默认配置

**示例内容**:
```makefile
CUDA_ARCH := sm_80  # 更改默认的 CUDA 架构
USE_CUDA := 0       # 如果没有cuda环境，就应当覆盖默认值
```

该文件不在 git 的管理下。

---

#### build.sh - 组件编译脚本

**作用**: 编译和测试项目中的组件. psum 主要功能以 header only 形式提供，只有少数组件需要编译。

**执行步骤**:
1. 加载 `env_load.sh` 环境变量
2. 编译 field solver 后端 (C++/CUDA)
3. 测试 field solver 组件
4. 编译 pypsum 序列化模块 (Python/C++ 互操作)
5. 测试 pypsum 模块互操作性

**使用方法**:
```bash
bash build.sh
```

build.sh 应当在已经创建好 `env_load.sh` 环境变量的情况下运行。

---

#### runCheck.sh - 冒烟测试脚本

使用`test/runCheck.sh`来进一步确认环境是否配置正确。

---

### 构建流程

#### 首次构建

适用于：首次克隆项目、系统环境发生变化、需要调整组件编译配置等情况

```bash
# 0. 环境预配置 (如果需要)
# module load cuda llvm openblas umfpack eigen

# 1. 扫描系统依赖
bash env_scan.sh

# 2. 检查生成的环境配置是否正确
cat env_load.sh

# 3. 创建自定义配置 (如果需要)
# 如需修改 CUDA 架构、禁用某些功能，创建 config.mk.local

# 4. 执行组件构建
bash build.sh

# 5. 加载环境变量
source env_load.sh # 虽然build.sh会自动加载，但当前bash会话中需要手动加载

# 编译某个应用或测试目标，例如：
cd application/collision-less-flow && make main   # 应用
cd test/field && make test_simplegrid             # 单个测试（编译并运行）
cd test && bash runCheck.sh                        # 门禁测试

```

#### 使用

适用于：环境已配置，仅需要编译项目

```bash
# 1. 加载环境变量
source env_load.sh

# 2. 编译并运行任意目标，例如：
cd application/collision-less-flow && make main && make run   # 运行示例算例
# 或
cd test && bash runCheck.sh   # 快层测试门禁（预期零失败）
```

---

### 文件关系

- `env_scan.sh` 扫描系统后自动生成 `env_load.sh`

- `build.sh` 依赖于`env_load.sh`和组件的 `makefile` 或 `build.sh`

- 组件中的 `makefile` 会读入 `config.mk`

- `config.mk` 会尝试获取 `config.mk.local` 并覆盖默认值

---