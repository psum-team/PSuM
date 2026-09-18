[English](dependency_en.md) | 中文

## 环境配置

**前置要求：cmake/g++版本**

要求 cmake 3.16 或更高版本; 要求 g++ 11 或更高版本.

在安装依赖前，运行以下指令检查基础工具是否满足要求：
```bash
fail=0

for c in bash git wget curl sed awk grep tar make cmake g++; do
  command -v $c >/dev/null || { echo "[FAIL] missing $c"; fail=1; }
done

if command -v cmake >/dev/null; then
  v=$(cmake --version | awk 'NR==1{print $3}')
  major=$(echo $v | cut -d. -f1)
  minor=$(echo $v | cut -d. -f2)
  if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 16 ]; }; then
    echo "[FAIL] cmake >=3.16 required (found $v)"; fail=1
  fi
fi

if command -v g++ >/dev/null; then
  v=$(g++ -dumpversion | cut -d. -f1)
  [ "$v" -lt 11 ] && { echo "[FAIL] g++ >=11 required (found $v)"; fail=1; }
fi

command -v lsb_release >/dev/null || echo "[WARN] lsb_release not found, you may need it."

command -v add-apt-repository >/dev/null || echo "[WARN] add-apt-repository not found, you may need it."

[ "$fail" -eq 0 ] && echo "Environment check PASSED" || { echo "Environment check FAILED"; exit 1; }
```

### 安装 clang/llvm

项目对 clang/llvm 的版本需求为>=15 且<=20。这里使用 llvm-16 版本。

#### Ubuntu 安装

使用 apt 安装：
```bash
wget https://apt.llvm.org/llvm.sh
sudo chmod +x llvm.sh
sudo ./llvm.sh 16
sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev lld-16
```

#### 源码编译安装

对于系统默认软件仓库中的 clang/llvm 版本可能不符合要求的情况，推荐从源码编译.

- 对于 CentOS 7/8, 由于其已经进入 EOL, 几乎必须从源码编译 llvm/clang; 且可能需要修复仓库以使用 dnf.
- Python3 required for LLVM build.
- 源码安装 llvm 所需时间很长。

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz
tar -xf llvm-project-16.0.6.src.tar.xz
cd llvm-project-16.0.6.src
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" -DLLVM_ENABLE_RUNTIMES="openmp" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLIBOMP_OMPD_SUPPORT=OFF -DPython3_EXECUTABLE="" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
make -j$(nproc)
sudo make install
```

*如果第一步下载速度过慢，可以在浏览器中下载：[GitHub llvm-16](https://github.com/llvm/llvm-project/releases/tag/llvmorg-16.0.6)，选择`llvm-project-16.0.6.src.tar.xz`。*

### AdaptiveCpp

AdaptiveCpp（hipSYCL）实现 SYCL 标准，是 PSuM 跨平台特性的基础。依赖项：clang/llvm、boost。
安装时确保当前 g++版本>=11。

#### 安装 boost

**Ubuntu：**
```bash
sudo apt update
sudo apt install -y libboost-all-dev
```

**CentOS 8：**
```bash
sudo dnf install -y boost-devel
```

#### 编译安装 AdaptiveCpp

```bash
git clone --depth=1 https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp ..
sudo make install
```

如果 cmake 未检测到正确的 llvm 也会报错。可以手动指定路径：
```bash
cmake -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp -DLLVM_DIR=/usr/lib/llvm-16/cmake -DCLANG_EXECUTABLE_PATH=/usr/lib/llvm-16/bin/clang++ ..
```

其中 LLVM_DIR 参数可以用`llvm-config --cmakedir`获取，CLANG_EXECUTABLE_PATH 参数可以用`which clang++`获取。

### 安装 Eigen

依赖版本为 Eigen 3.4.0.

已知 3.3.7 版本无法通过编译。
某些系统版本（如 Ubuntu 20.04）下通过 apt 安装的 Eigen 可能存在版本问题，建议手动编译安装。

### 安装 OpenBLAS

Ubuntu：
```bash
sudo apt install libopenblas-dev
```
CentOS：
```bash
sudo dnf install openblas-devel
```
### 安装 SuiteSparse(或仅安装 UMFPACK)
Ubuntu：
```bash
sudo apt install libsuitesparse-dev
```
CentOS：
```bash
sudo dnf install suitesparse-devel
```

### 安装 python 支持 (可选)

需要安装 python3 与 pybind11 以编译 python 相关模块。

构建脚本会自动创建虚拟环境（`.pypsum_venv`），该步骤依赖 `python3 -m venv`：

- Debian/Ubuntu 需要先安装：`sudo apt install python3-venv`；
- 无法安装时，脚本会自动回退到 `virtualenv`（`pip install --user virtualenv`）。

顶层 `./build.sh` 的 Python 相关行为：

- 它总是先运行 `env_scan.sh` 重新检测环境并再生成 `env_load.sh`（幂等，但会覆盖已有文件）；
- 已存在的 `.pypsum_venv` **默认保留并复用**；设 `PSUM_FORCE_VENV=1` 强制删除重建；
- 设 `PSUM_SKIP_PYTHON=1` 可完全跳过 Python 模块的构建与测试（无 Python 也能完成 C++ 部分构建）。

## 环境检测和导入

在完成上述依赖安装后，运行`env_scan.sh`检测能否获取到依赖。该脚本会创建`env_load.sh`用于加载环境变量。
正确的`env_load.sh`对后续构建是必须的。详见[构建文档](building.md)。