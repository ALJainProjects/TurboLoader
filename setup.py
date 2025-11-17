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


def find_library(name, brew_name=None, pkg_config_name=None):
    """Find a library installation (works on macOS and Linux)"""
    if brew_name is None:
        brew_name = name
    if pkg_config_name is None:
        pkg_config_name = name

    # Try Homebrew first (macOS)
    try:
        brew_prefix = subprocess.check_output(
            ['brew', '--prefix', brew_name],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        include_path = os.path.join(brew_prefix, 'include')
        lib_path = os.path.join(brew_prefix, 'lib')

        if os.path.exists(include_path) and os.path.exists(lib_path):
            return include_path, lib_path
    except:
        pass

    # Try common system locations
    possible_paths = [
        '/usr/local',
        '/usr',
    ]

    for base_path in possible_paths:
        include_path = os.path.join(base_path, 'include')
        lib_path = os.path.join(base_path, 'lib')

        if os.path.exists(include_path) and os.path.exists(lib_path):
            return include_path, lib_path

    # Try pkg-config
    try:
        include_path = subprocess.check_output(
            ['pkg-config', '--variable=includedir', pkg_config_name],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        lib_path = subprocess.check_output(
            ['pkg-config', '--variable=libdir', pkg_config_name],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        if include_path and lib_path:
            return include_path, lib_path
    except:
        pass

    return None, None


# Find all required libraries
print("Detecting dependencies...")

jpeg_include, jpeg_lib = find_library('jpeg-turbo', 'jpeg-turbo', 'libjpeg')
if not jpeg_include:
    raise RuntimeError(
        "Could not find libjpeg-turbo installation.\n"
        "Please install it:\n"
        "  macOS: brew install jpeg-turbo\n"
        "  Linux: sudo apt-get install libjpeg-turbo8-dev\n"
    )
print(f"  libjpeg-turbo: {jpeg_include}")

png_include, png_lib = find_library('libpng', 'libpng', 'libpng')
if not png_include:
    raise RuntimeError(
        "Could not find libpng installation.\n"
        "Please install it:\n"
        "  macOS: brew install libpng\n"
        "  Linux: sudo apt-get install libpng-dev\n"
    )
print(f"  libpng: {png_include}")

webp_include, webp_lib = find_library('webp', 'webp', 'libwebp')
if not webp_include:
    raise RuntimeError(
        "Could not find libwebp installation.\n"
        "Please install it:\n"
        "  macOS: brew install webp\n"
        "  Linux: sudo apt-get install libwebp-dev\n"
    )
print(f"  libwebp: {webp_include}")

curl_include, curl_lib = find_library('curl', 'curl', 'libcurl')
if not curl_include:
    raise RuntimeError(
        "Could not find libcurl installation.\n"
        "Please install it:\n"
        "  macOS: brew install curl\n"
        "  Linux: sudo apt-get install libcurl4-openssl-dev\n"
    )
print(f"  libcurl: {curl_include}")

ext_modules = [
    Extension(
        '_turboloader',
        sources=['src/python/turboloader_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            jpeg_include,
            png_include,
            webp_include,
            curl_include,
            'src',  # For pipeline headers
        ],
        library_dirs=[
            jpeg_lib,
            png_lib,
            webp_lib,
            curl_lib,
        ],
        libraries=[
            'jpeg',
            'png',
            'webp',
            'webpdemux',
            'curl',
        ],
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
    version='1.2.1',
    author='TurboLoader Contributors',
    description='High-performance data loading for ML frameworks with 19 SIMD-accelerated transforms',
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
