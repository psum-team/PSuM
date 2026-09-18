English | [中文](dependency.md)

## Environment Setup

**Prerequisites: cmake/g++ versions**

cmake 3.16 or later is required; g++ 11 or later is required.

Before installing dependencies, run the following commands to check whether the basic tools meet the requirements:
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

### Installing clang/llvm

The project requires clang/llvm version >=15 and <=20. Here we use llvm-16.

#### Ubuntu Installation

Install using apt:
```bash
wget https://apt.llvm.org/llvm.sh
sudo chmod +x llvm.sh
sudo ./llvm.sh 16
sudo apt install -y libclang-16-dev clang-tools-16 libomp-16-dev llvm-16-dev lld-16
```

#### Building from Source

If the clang/llvm version in the system's default repositories does not meet the requirements, building from source is recommended.

- For CentOS 7/8, since they have reached EOL, building llvm/clang from source is almost mandatory; you may also need to fix repositories to use dnf.
- Python3 is required for LLVM build.
- Building llvm from source takes a long time.

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz
tar -xf llvm-project-16.0.6.src.tar.xz
cd llvm-project-16.0.6.src
mkdir build && cd build
cmake -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra" -DLLVM_ENABLE_RUNTIMES="openmp" -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON -DLIBOMP_OMPD_SUPPORT=OFF -DPython3_EXECUTABLE="" -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" ../llvm
make -j$(nproc)
sudo make install
```

*If the download in the first step is too slow, you can download it in a browser: [GitHub llvm-16](https://github.com/llvm/llvm-project/releases/tag/llvmorg-16.0.6), select `llvm-project-16.0.6.src.tar.xz`.*

### AdaptiveCpp

AdaptiveCpp (hipSYCL) implements the SYCL standard and is the foundation of PSuM's cross-platform capabilities. Dependencies: clang/llvm, boost.
Make sure the current g++ version is >=11 before installing.

#### Installing Boost

**Ubuntu:**
```bash
sudo apt update
sudo apt install -y libboost-all-dev
```

**CentOS 8:**
```bash
sudo dnf install -y boost-devel
```

#### Building and Installing AdaptiveCpp

```bash
git clone --depth=1 https://github.com/AdaptiveCpp/AdaptiveCpp.git
cd AdaptiveCpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp ..
sudo make install
```

If cmake does not detect the correct llvm, it will also report an error. You can manually specify the paths:
```bash
cmake -DCMAKE_INSTALL_PREFIX=/opt/AdaptiveCpp -DLLVM_DIR=/usr/lib/llvm-16/cmake -DCLANG_EXECUTABLE_PATH=/usr/lib/llvm-16/bin/clang++ ..
```

The LLVM_DIR parameter can be obtained with `llvm-config --cmakedir`, and the CLANG_EXECUTABLE_PATH parameter can be obtained with `which clang++`.

### Installing Eigen

The required version is Eigen 3.4.0.

It is known that version 3.3.7 fails to compile.
On some systems (such as Ubuntu 20.04), the Eigen version installed via apt may have issues; manual compilation and installation is recommended.

### Installing OpenBLAS

Ubuntu:
```bash
sudo apt install libopenblas-dev
```
CentOS:
```bash
sudo dnf install openblas-devel
```
### Installing SuiteSparse (or only UMFPACK)
Ubuntu:
```bash
sudo apt install libsuitesparse-dev
```
CentOS:
```bash
sudo dnf install suitesparse-devel
```

### Installing Python Support (Optional)

You need to install python3 and pybind11 to compile Python-related modules.

The build script creates a virtual environment (`.pypsum_venv`) automatically; this step relies on `python3 -m venv`:

- On Debian/Ubuntu, install it first: `sudo apt install python3-venv`;
- If that is not possible, the script falls back to `virtualenv` (`pip install --user virtualenv`).

Python-related behavior of the top-level `./build.sh`:

- It always runs `env_scan.sh` first to re-detect the environment and regenerate `env_load.sh` (idempotent, but it overwrites an existing file);
- An existing `.pypsum_venv` is **kept and reused by default**; set `PSUM_FORCE_VENV=1` to force a rebuild from scratch;
- Set `PSUM_SKIP_PYTHON=1` to skip the Python module build and tests entirely (the C++ part builds fine without Python).

## Environment Detection and Import

After completing the above dependency installations, run `env_scan.sh` to check whether the dependencies can be detected. This script will create `env_load.sh` for loading environment variables.
A correct `env_load.sh` is required for subsequent builds. See [Building Documentation](building_en.md).
