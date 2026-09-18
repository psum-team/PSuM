import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_bin_dir = os.path.join(_current_dir, "bin")

if _bin_dir not in sys.path:
    sys.path.insert(0, _bin_dir)

try:
    import mas_file_py
    MasFile = mas_file_py.MasFile
    del mas_file_py
except ImportError as e:
    raise ImportError(f"Cannot load C++ extension module. Please make sure that the .so file exists in the {_bin_dir} directory. Error: {e}")

__all__ = ["MasFile"]