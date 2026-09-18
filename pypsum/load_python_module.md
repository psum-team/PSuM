[English](load_python_module_en.md) | 中文

## 构建与安装

在项目根目录下运行以下脚本。该脚本会在**仓库根目录**创建虚拟环境（`.pypsum_venv`），安装依赖（`pybind11`、`numpy`），并把模块编译到 `pypsum/bin`。

```bash
cd pypsum/serialization
chmod +x build.sh
./build.sh
```

## 在 Python 中使用

由于该模块是针对虚拟环境中的 Python 版本编译的，请使用以下方法之一确保所有依赖（如 `numpy`）正确加载。

### 方法 A：推荐（激活虚拟环境）
这是最可靠的方法，可以确保环境与编译的二进制文件完全一致：

```bash
# 1. 激活脚本创建的虚拟环境（位于仓库根目录；按当前目录给正确路径）
source ../../.pypsum_venv/bin/activate

# 2. 运行你的 Python 脚本
python your_script.py
```

### 方法 B：动态路径加载（在脚本内部）
如果需要从其他环境加载该模块，必须手动将 `bin` 目录添加到 `sys.path`。建议使用相对路径计算来避免硬编码：

```python
import sys
import os

# 按你的脚本与仓库的相对位置调整层级（例如仓库根下的 tests/ 用一级 "../" 即可）
current_dir = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.normpath(os.path.join(current_dir, "..", "pypsum", "bin"))

if bin_path not in sys.path:
    sys.path.insert(0, bin_path)

# 现在可以导入编译好的模块
import mas_file_py
```
