#!/usr/bin/env python3
"""
Setup script for TurboLoader - High-Performance ML Data Loading Library
"""

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess
from pathlib import Path

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        # Place the .so file inside the turboloader/ package directory
        library_output_dir = os.path.join(extdir, 'turboloader')

        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={library_output_dir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DTURBOLOADER_BUILD_TESTS=OFF',
            '-DTURBOLOADER_BUILD_BENCHMARKS=OFF',
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # Parallel build
        build_args += ['--', '-j8']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


# Read long description from README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='turboloader',
    version='0.3.7',
    author='Arnav Jain',
    author_email='arnav@arnavjain.com',
    description='High-performance ML data loading library with SIMD optimizations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/arnavjain/turboloader',
    project_urls={
        'Bug Reports': 'https://github.com/arnavjain/turboloader/issues',
        'Source': 'https://github.com/arnavjain/turboloader',
        'Documentation': 'https://github.com/arnavjain/turboloader/blob/main/README.md',
    },
    ext_modules=[CMakeExtension('turboloader')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'webdataset>=0.2.0',
            'pillow>=8.0.0',
        ],
        'benchmarks': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
            'webdataset>=0.2.0',
            'pillow>=8.0.0',
            'matplotlib>=3.3.0',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: C++',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
    ],
    license='MIT',
    keywords='machine-learning data-loading simd performance pytorch tensorflow',
)
