"""
Build script for the C++ CPU extension with AVX-512 support.
"""
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import sys

extra_compile_args = []
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/std:c++17", "/arch:AVX512"]
else:
    extra_compile_args = ["-O3", "-std=c++17", "-mavx512f", "-mavx512vl"]

setup(
    name="ternary_cpu_ext",
    ext_modules=[
        CppExtension(
            "ternary_cpu_ext",
            [
                "cpp/src/ternary_kernel_avx512.cpp",
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)