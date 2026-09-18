English | [中文](load_python_module.md)

## Build and Installation

Run this from the repository root:

```bash
cd pypsum/serialization
chmod +x build.sh
./build.sh
```
The script creates .pypsum_venv/ in the repository root, installs pybind11 and numpy, and writes the compiled extension to pypsum/bin/.

## Usage in Python

Since this module is compiled specifically for the Python version within the virtual environment, please use one of the following methods to ensure all dependencies (like `numpy`) load correctly.

### Method A: Recommended (Activate Virtual Environment)
This is the most reliable method as it ensures the environment is perfectly aligned with the compiled binary:

```bash
# 1. Activate the virtual environment created by the script (in the repo root; adjust for your cwd)
source ../../.pypsum_venv/bin/activate

# 2. Run your Python script
python your_script.py
```

### Method B: Dynamic Path Loading (Inside Script)
If you need to load the module from a different environment, you must manually add the `bin` directory to `sys.path`. We recommend using a relative path calculation to avoid hardcoding:

```python
import sys
import os

# Absolute path of the 'bin' directory relative to this script.
# The extension module is built into 'pypsum/bin/'; adjust the
# number of "../" levels to where your script lives relative to the repository.
current_dir = os.path.dirname(os.path.abspath(__file__))
bin_path = os.path.normpath(os.path.join(current_dir, "..", "pypsum", "bin"))

if bin_path not in sys.path:
    sys.path.insert(0, bin_path)

# Now you can import the compiled module
import mas_file_py
```
