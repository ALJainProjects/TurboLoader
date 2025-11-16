"""
Setup script for TurboLoader Python bindings

Builds the turboloader module using pybind11.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""

    def __str__(self):
        import pybind11
        return pybind11.get_include()


def find_jpeg_turbo():
    """Find libjpeg-turbo installation"""
    # Try common locations
    possible_paths = [
        '/opt/homebrew/opt/jpeg-turbo',  # Homebrew on Apple Silicon
        '/usr/local/opt/jpeg-turbo',      # Homebrew on Intel
        '/usr',                           # System
    ]

    for base_path in possible_paths:
        include_path = os.path.join(base_path, 'include')
        lib_path = os.path.join(base_path, 'lib')

        if os.path.exists(include_path) and os.path.exists(lib_path):
            return include_path, lib_path

    # Try pkg-config
    try:
        include_path = subprocess.check_output(
            ['pkg-config', '--variable=includedir', 'libjpeg'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        lib_path = subprocess.check_output(
            ['pkg-config', '--variable=libdir', 'libjpeg'],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if include_path and lib_path:
            return include_path, lib_path
    except:
        pass

    raise RuntimeError(
        "Could not find libjpeg-turbo installation.\n"
        "Please install it:\n"
        "  macOS: brew install jpeg-turbo\n"
        "  Linux: sudo apt-get install libjpeg-turbo8-dev\n"
    )


# Find JPEG library
jpeg_include, jpeg_lib = find_jpeg_turbo()

print(f"Found libjpeg-turbo:")
print(f"  Include: {jpeg_include}")
print(f"  Library: {jpeg_lib}")

ext_modules = [
    Extension(
        'turboloader',
        sources=['src/python/turboloader_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            jpeg_include,
            'src',  # For pipeline_v2 headers
        ],
        library_dirs=[jpeg_lib],
        libraries=['jpeg'],
        language='c++',
        extra_compile_args=[
            '-std=c++20',
            '-O3',
            '-march=native',  # Enable CPU-specific optimizations
            '-fvisibility=hidden',
        ],
    ),
]


class BuildExt(build_ext):
    """Custom build extension to set C++20 flag"""

    def build_extensions(self):
        # Set C++20 standard
        ct = self.compiler.compiler_type
        opts = []

        if ct == 'unix':
            opts.append('-std=c++20')
        elif ct == 'msvc':
            opts.append('/std:c++20')

        for ext in self.extensions:
            ext.extra_compile_args = opts + ext.extra_compile_args

        build_ext.build_extensions(self)


setup(
    name='turboloader',
    version='0.4.0',
    author='TurboLoader Contributors',
    description='High-performance data loading for PyTorch (TurboLoader)',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=[
        'pybind11>=2.10.0',
        'numpy>=1.20.0',
    ],
    extras_require={
        'torch': ['torch>=1.10.0'],
        'dev': ['pytest', 'black', 'mypy'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    zip_safe=False,
)
