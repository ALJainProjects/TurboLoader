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


def find_library(name, brew_name=None, pkg_config_name=None, header_subdir=None):
    """Find a library installation (works on macOS and Linux)

    Args:
        name: Library name
        brew_name: Homebrew package name
        pkg_config_name: pkg-config name
        header_subdir: Subdirectory where headers are located (e.g., 'curl' for curl/curl.h)
    """
    if brew_name is None:
        brew_name = name
    if pkg_config_name is None:
        pkg_config_name = name

    def verify_include(include_path):
        """Verify the include path actually has the required headers"""
        if header_subdir:
            check_path = os.path.join(include_path, header_subdir)
        else:
            check_path = include_path
        return os.path.exists(check_path)

    # Try pkg-config first (most reliable on Linux)
    try:
        cflags = (
            subprocess.check_output(
                ["pkg-config", "--cflags", pkg_config_name], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        libs = (
            subprocess.check_output(
                ["pkg-config", "--libs", pkg_config_name], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

        # Parse -I flag for include path
        include_path = None
        for flag in cflags.split():
            if flag.startswith("-I"):
                path = flag[2:]
                if verify_include(path):
                    include_path = path
                    break

        # If no -I flag found, try to get includedir variable
        if not include_path:
            try:
                inc_dir = (
                    subprocess.check_output(
                        ["pkg-config", "--variable=includedir", pkg_config_name],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
                if inc_dir and verify_include(inc_dir):
                    include_path = inc_dir
            except Exception:
                pass

        # Parse -L flag for library path (or use default)
        lib_path = None
        for flag in libs.split():
            if flag.startswith("-L"):
                lib_path = flag[2:]
                break

        # If no -L flag, try to get libdir
        if not lib_path:
            try:
                lib_path = (
                    subprocess.check_output(
                        ["pkg-config", "--variable=libdir", pkg_config_name],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except Exception:
                pass

        # If we found include but not lib, use system default
        if include_path and not lib_path:
            for lp in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/lib"]:
                if os.path.exists(lp):
                    lib_path = lp
                    break

        if include_path and lib_path:
            return include_path, lib_path
    except Exception:
        pass

    # Try Homebrew (macOS)
    try:
        brew_prefix = (
            subprocess.check_output(["brew", "--prefix", brew_name], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )

        include_path = os.path.join(brew_prefix, "include")
        lib_path = os.path.join(brew_prefix, "lib")

        if (
            os.path.exists(include_path)
            and os.path.exists(lib_path)
            and verify_include(include_path)
        ):
            return include_path, lib_path
    except Exception:
        pass

    # Try common system locations (verify headers exist)
    # Order matters - check more specific paths first
    possible_paths = [
        "/usr/local",
        "/usr",
    ]

    for base_path in possible_paths:
        include_path = os.path.join(base_path, "include")
        lib_path = os.path.join(base_path, "lib")

        # Also check lib64 on Linux
        if not os.path.exists(lib_path) and os.path.exists(os.path.join(base_path, "lib64")):
            lib_path = os.path.join(base_path, "lib64")

        if (
            os.path.exists(include_path)
            and os.path.exists(lib_path)
            and verify_include(include_path)
        ):
            return include_path, lib_path

    return None, None


def get_extensions():
    """Build extension modules - only called when actually building wheels"""
    print("Detecting dependencies...")

    jpeg_include, jpeg_lib = find_library("jpeg-turbo", "jpeg-turbo", "libjpeg")
    if not jpeg_include:
        raise RuntimeError(
            "Could not find libjpeg-turbo installation.\n"
            "Please install it:\n"
            "  macOS: brew install jpeg-turbo\n"
            "  Linux: sudo apt-get install libjpeg-turbo8-dev\n"
        )
    print(f"  libjpeg-turbo: {jpeg_include}")

    png_include, png_lib = find_library("libpng", "libpng", "libpng")
    if not png_include:
        raise RuntimeError(
            "Could not find libpng installation.\n"
            "Please install it:\n"
            "  macOS: brew install libpng\n"
            "  Linux: sudo apt-get install libpng-dev\n"
        )
    print(f"  libpng: {png_include}")

    webp_include, webp_lib = find_library("webp", "webp", "libwebp")
    if not webp_include:
        raise RuntimeError(
            "Could not find libwebp installation.\n"
            "Please install it:\n"
            "  macOS: brew install webp\n"
            "  Linux: sudo apt-get install libwebp-dev\n"
        )
    print(f"  libwebp: {webp_include}")

    curl_include, curl_lib = find_library("curl", "curl", "libcurl", header_subdir="curl")
    if not curl_include:
        raise RuntimeError(
            "Could not find libcurl installation.\n"
            "Please install it:\n"
            "  macOS: brew install curl\n"
            "  Linux: sudo apt-get install libcurl4-openssl-dev\n"
        )
    print(f"  libcurl: {curl_include}")

    lz4_include, lz4_lib = find_library("lz4", "lz4", "liblz4")
    if not lz4_include:
        raise RuntimeError(
            "Could not find lz4 installation.\n"
            "Please install it:\n"
            "  macOS: brew install lz4\n"
            "  Linux: sudo apt-get install liblz4-dev\n"
        )
    print(f"  lz4: {lz4_include}")

    return [
        Extension(
            "_turboloader",
            sources=[
                "src/python/turboloader_bindings.cpp",
            ],
            include_dirs=[
                get_pybind_include(),
                jpeg_include,
                png_include,
                webp_include,
                curl_include,
                lz4_include,
                "src",  # For pipeline headers
            ],
            library_dirs=[
                jpeg_lib,
                png_lib,
                webp_lib,
                curl_lib,
                lz4_lib,
            ],
            libraries=[
                "jpeg",
                "png",
                "webp",
                "webpdemux",
                "curl",
                "lz4",
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++20",
                "-O3",
                "-fvisibility=hidden",
            ],
            extra_link_args=[],
        ),
    ]


class LazyExtensionList(list):
    """Lazily evaluate extensions only when needed for building"""

    def __init__(self):
        super().__init__()
        self._extensions = None

    def _get_extensions(self):
        if self._extensions is None:
            self._extensions = get_extensions()
        return self._extensions

    def __iter__(self):
        return iter(self._get_extensions())

    def __len__(self):
        return len(self._get_extensions())

    def __getitem__(self, key):
        return self._get_extensions()[key]

    def __bool__(self):
        # Return True to indicate we have extensions
        # This prevents setuptools from skipping build_ext
        return True


class BuildExt(build_ext):
    """Custom build extension to set C++20 flag and platform-specific optimizations"""

    def build_extensions(self):
        import platform

        ct = self.compiler.compiler_type
        arch = platform.machine().lower()
        system = platform.system().lower()

        for ext in self.extensions:
            opts = list(ext.extra_compile_args)
            link_opts = list(ext.extra_link_args)

            if ct == "unix":
                # macOS-specific flags
                if system == "darwin":
                    opts.append("-mmacosx-version-min=10.15")
                    link_opts.append("-mmacosx-version-min=10.15")
                    if "arm64" in arch:
                        opts.append("-mcpu=apple-m1")
                else:
                    # Linux - use march=native for x86
                    if "x86" in arch or "amd64" in arch:
                        opts.append("-march=native")
            elif ct == "msvc":
                opts.append("/std:c++20")

            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


# Check if we're in a metadata-only operation (sdist, egg_info, etc.)
# These operations don't need the native libraries
def is_metadata_only():
    """Check if this is a metadata-only operation that doesn't need libraries"""
    # Check command line arguments
    metadata_commands = {
        "sdist",
        "egg_info",
        "--version",
        "--name",
        "--author",
        "--author-email",
        "--maintainer",
        "--maintainer-email",
        "--url",
        "--license",
        "--description",
        "--long-description",
        "--classifiers",
        "--keywords",
        "--platforms",
        "--fullname",
    }

    for arg in sys.argv[1:]:
        if arg in metadata_commands:
            return True
        # Also check for pip's metadata extraction
        if "egg_info" in arg or "dist_info" in arg:
            return True

    # Check environment variable that pip sets during metadata extraction
    if os.environ.get("_PYPROJECT_HOOKS_BUILD_BACKEND"):
        # We're being called by pyproject-hooks for metadata
        # Check if it's just for getting requirements
        if any("get_requires" in arg for arg in sys.argv):
            return True

    return False


# Use lazy extension loading to defer library detection
if is_metadata_only():
    ext_modules = []
else:
    ext_modules = LazyExtensionList()

setup(
    name="turboloader",
    version="2.3.6",
    author="TurboLoader Contributors",
    description="High-performance data loading for ML with pipe operator, HDF5/TFRecord/Zarr, GPU transforms, Azure support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    install_requires=[
        "pybind11>=2.10.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "torch": ["torch>=1.10.0"],
        "dev": ["pytest", "black", "mypy"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
