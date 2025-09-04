from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

try:
    import numpy as _np
    numpy_includes = [_np.get_include()]
except Exception:
    # Numpy might not be available at import-time; allow build to fail loudly later
    numpy_includes = []


ext_modules = [
    Pybind11Extension(
        "occupance._occupancy",
        [
            "src/occupance/occupancy_module.cpp",
        ],
        cxx_std=17,
        include_dirs=numpy_includes,
    )
]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)


