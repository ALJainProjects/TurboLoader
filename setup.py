"""
Setup script for TurboLoader Python bindings

Builds the turboloader module using pybind11.
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import subprocess

# The package version is derived from the git tag by setuptools_scm (see pyproject
# [tool.setuptools_scm]); setup() does NOT set it. We only need the resolved string
# here to pass to the C++ extension as -DTURBOLOADER_VERSION so the native module's
# version()/features() always match the package (this skew once shipped as 2.5.0 vs
# 2.25.0). Resolve from scm (git checkout) or the generated _version.py (sdist).
def _resolve_version():
    # 1. Honor the pretend-version env (set in CI) so the C++ macro matches the wheel
    #    metadata exactly — the wheel smoke-test asserts version() == __version__.
    for _k in ("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TURBOLOADER",
               "SETUPTOOLS_SCM_PRETEND_VERSION"):
        if os.environ.get(_k):
            return os.environ[_k]
    # 2. setuptools_scm from a git checkout.
    try:
        from setuptools_scm import get_version
        return get_version(root=".", relative_to=__file__)
    except Exception:
        pass
    # 3. The version file written into an sdist by setuptools_scm.
    try:
        ns = {}
        with open(os.path.join("turboloader", "_version.py")) as fh:
            exec(fh.read(), ns)
        return ns.get("__version__") or ns.get("version") or "0.0.0"
    except Exception:
        return "0.0.0"


VERSION = _resolve_version()


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

    # Explicit prefix override: TURBOLOADER_DEPS_PREFIX points at dependencies built
    # from source (used to make portable macOS wheels — Homebrew bottles target the
    # runner's OS, which makes wheels non-portable). Only used when the prefix actually
    # contains this library's headers; otherwise we fall through (e.g. system libcurl).
    deps_prefix = os.environ.get("TURBOLOADER_DEPS_PREFIX")
    if deps_prefix:
        inc = os.path.join(deps_prefix, "include")
        lib = os.path.join(deps_prefix, "lib")
        if os.path.isdir(lib) and verify_include(inc):
            return inc, lib

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


def find_openmp():
    """Find OpenMP installation for the current platform"""
    import platform
    system = platform.system().lower()

    if system == "darwin":
        # macOS: OpenMP requires Homebrew libomp
        try:
            omp_prefix = (
                subprocess.check_output(
                    ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
                )
                .decode()
                .strip()
            )
            omp_include = os.path.join(omp_prefix, "include")
            omp_lib = os.path.join(omp_prefix, "lib")
            if os.path.exists(omp_include) and os.path.exists(omp_lib):
                return {
                    "include": omp_include,
                    "lib": omp_lib,
                    "compile_flags": ["-Xpreprocessor", "-fopenmp"],
                    "link_flags": [f"-L{omp_lib}", "-lomp"],
                }
        except Exception:
            pass
        return None
    else:
        # Linux: OpenMP is built into GCC/Clang
        return {
            "include": None,
            "lib": None,
            "compile_flags": ["-fopenmp"],
            "link_flags": ["-fopenmp"],
        }


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

    # OpenMP is OFF by default. Linking a second OpenMP runtime (Homebrew libomp)
    # into the same process as PyTorch (which bundles its own libomp) causes an
    # import-order-dependent hard crash on macOS. Since TurboLoader is meant to be
    # used alongside PyTorch, correctness wins over the marginal OpenMP speedup.
    # Power users on non-PyTorch setups can opt in with TURBOLOADER_ENABLE_OPENMP=1.
    # Without -fopenmp the `#pragma omp` directives simply compile to serial code.
    enable_omp = os.environ.get("TURBOLOADER_ENABLE_OPENMP", "0") == "1"
    omp = find_openmp() if enable_omp else None
    if enable_omp and omp:
        print("  OpenMP: ENABLED via TURBOLOADER_ENABLE_OPENMP=1")
    elif enable_omp:
        print("  OpenMP: requested but not found; building without it")
    else:
        print("  OpenMP: disabled by default (avoids libomp/PyTorch crash); "
              "set TURBOLOADER_ENABLE_OPENMP=1 to enable")

    # On macOS, do NOT add a system SDK include root (e.g. the SDK's usr/include where
    # libcurl lives) as an explicit -I: it shadows libc++'s own headers and breaks
    # <cmath> ("tried including <math.h> but didn't find libc++'s <math.h>") when the
    # active Xcode SDK differs from the one found. System headers come via -isysroot.
    # (On Linux the yum/apt deps genuinely live in /usr/include, so keep them there.)
    _macos = sys.platform == "darwin"

    def _keep_include(inc):
        if not inc:
            return False
        # get_pybind_include() returns a lazy object, not a str — stringify before testing.
        s = str(inc)
        if _macos and (".sdk/usr/include" in s.lower() or s.rstrip("/") == "/usr/include"):
            return False
        return True

    include_dirs = [
        inc
        for inc in (
            get_pybind_include(),
            jpeg_include,
            png_include,
            webp_include,
            curl_include,
            lz4_include,
        )
        if _keep_include(inc)
    ]
    include_dirs.append("src")  # For pipeline headers
    # Likewise drop a system SDK lib root on macOS (curl): -L against a mismatched
    # SDK's usr/lib can pick the wrong libcurl stub. System libs come via -isysroot.
    def _keep_lib(lib):
        if not lib:
            return False
        s = str(lib)
        if _macos and (".sdk/usr/lib" in s.lower() or s.rstrip("/") == "/usr/lib"):
            return False
        return True

    library_dirs = [
        lib for lib in (jpeg_lib, png_lib, webp_lib, curl_lib, lz4_lib) if _keep_lib(lib)
    ]
    # Detect compiler for LTO flag: Clang uses -flto=thin, GCC uses -flto
    import platform as _plat

    lto_flag = "-flto=thin" if _plat.system() == "Darwin" else "-flto"

    compile_args = [
        "-std=c++20",
        "-O3",
        "-fvisibility=hidden",
        "-funroll-loops",
        lto_flag,
    ]
    if not enable_omp:
        # `#pragma omp` lines are harmless no-ops without -fopenmp; silence the warning.
        compile_args.append("-Wno-unknown-pragmas")
    link_args = [lto_flag]

    if omp:
        compile_args.extend(omp["compile_flags"])
        link_args.extend(omp["link_flags"])
        if omp["include"]:
            include_dirs.append(omp["include"])
        if omp["lib"]:
            library_dirs.append(omp["lib"])

    # Metal GPU transform path: macOS arm64 only. Adds the Obj-C++ .mm (clang compiles it
    # as Objective-C++ by file extension) plus a -DTURBOLOADER_METAL define so the binding
    # exposes the GPU entry points; build_ext links the Metal framework. Opt out with
    # TURBOLOADER_ENABLE_METAL=0. On Linux/Intel it is simply never added, so those wheels
    # are byte-for-byte unaffected.
    extra_sources = []
    extra_macros = []
    if (
        _macos
        and _plat.machine().lower() in ("arm64", "aarch64")
        and os.environ.get("TURBOLOADER_ENABLE_METAL", "1") == "1"
    ):
        extra_sources.append("src/metal/metal_transforms.mm")
        extra_macros.append(("TURBOLOADER_METAL", "1"))

    return [
        Extension(
            "_turboloader",
            sources=[
                "src/python/turboloader_bindings.cpp",
            ]
            + extra_sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=[
                "jpeg",
                "png",
                "webp",
                "curl",
                "lz4",
            ],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[("TURBOLOADER_VERSION", '"{}"'.format(VERSION))] + extra_macros,
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
        import re

        ct = self.compiler.compiler_type
        arch = platform.machine().lower()
        system = platform.system().lower()

        # Teach distutils to accept Objective-C++ (.mm) sources on macOS (the Metal
        # transform unit). The Extension language is "c++", so every source — including
        # .mm — is compiled with the C++ frontend (clang++), which handles Objective-C++
        # by file extension.
        if system == "darwin" and ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")

        # On macOS, strip problematic flags from Python's embedded compiler configuration
        # python.org Python embeds CFLAGS/CPPFLAGS that can interfere with libc++ headers
        if system == "darwin" and ct == "unix":
            # Filter out problematic include paths from preprocessor args
            if hasattr(self.compiler, "preprocessor"):
                self.compiler.preprocessor = [
                    arg
                    for arg in self.compiler.preprocessor
                    if not (arg.startswith("-I") and "Python.framework" in arg)
                ]

            # Filter compiler_so (used for C++ compilation)
            if hasattr(self.compiler, "compiler_so"):
                filtered = []
                skip_next = False
                for arg in self.compiler.compiler_so:
                    if skip_next:
                        skip_next = False
                        continue
                    # Skip -isysroot with problematic paths and -I flags pointing to Python framework
                    if arg == "-isysroot":
                        skip_next = True
                        continue
                    if arg.startswith("-I") and "Python.framework" in arg:
                        continue
                    # Keep other flags but filter problematic -iwithsysroot
                    if arg.startswith("-iwithsysroot"):
                        continue
                    filtered.append(arg)
                self.compiler.compiler_so = filtered

            # Same for compiler_cxx
            if hasattr(self.compiler, "compiler_cxx"):
                filtered = []
                skip_next = False
                for arg in self.compiler.compiler_cxx:
                    if skip_next:
                        skip_next = False
                        continue
                    if arg == "-isysroot":
                        skip_next = True
                        continue
                    if arg.startswith("-I") and "Python.framework" in arg:
                        continue
                    if arg.startswith("-iwithsysroot"):
                        continue
                    filtered.append(arg)
                self.compiler.compiler_cxx = filtered

        for ext in self.extensions:
            opts = list(ext.extra_compile_args)
            link_opts = list(ext.extra_link_args)

            if ct == "unix":
                # macOS-specific flags
                if system == "darwin":
                    # Respect MACOSX_DEPLOYMENT_TARGET when the build sets it
                    # (cibuildwheel does, matched to the runner's Homebrew dylibs).
                    # Only fall back to an explicit minimum otherwise, and never force
                    # 10.15 — it disables C++20 libc++ availability and breaks the
                    # build inside <memory> on Xcode 15.x.
                    if not os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
                        opts.append("-mmacosx-version-min=11.0")
                        link_opts.append("-mmacosx-version-min=11.0")
                    if "arm64" in arch:
                        opts.append("-mcpu=native")
                    # Add rpath for Homebrew libraries on macOS
                    for lib_dir in ext.library_dirs:
                        if lib_dir and os.path.exists(lib_dir):
                            link_opts.append(f"-Wl,-rpath,{lib_dir}")
                    # Link the Metal framework when the GPU transform path is compiled in
                    # (presence of the -DTURBOLOADER_METAL macro is the single source of truth).
                    if any(name == "TURBOLOADER_METAL" for name, _ in ext.define_macros):
                        link_opts += ["-framework", "Metal", "-framework", "Foundation"]
                else:
                    # Linux x86 SIMD flags
                    if "x86" in arch or "amd64" in arch:
                        if os.environ.get("TURBOLOADER_PORTABLE", "0") == "1":
                            # Portable: SSE4.2 baseline (compatible with all modern x86_64)
                            opts.append("-msse4.2")
                        else:
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
    # version omitted: provided dynamically by setuptools_scm (pyproject dynamic).
    author="TurboLoader Contributors",
    description="High-performance data loading for ML with pipe operator, HDF5/TFRecord/Zarr, GPU transforms, Azure support",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    # Project metadata (name, version, dependencies, classifiers, license, python
    # requirement) is defined once in pyproject.toml [project] — keeping it here too
    # made setuptools warn that it was "overwritten". setup.py only carries the C++
    # extension build here.
    zip_safe=False,
)
