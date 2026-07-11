# Installation Guide

Complete installation instructions for TurboLoader on all supported platforms.

## Quick Install

```bash
pip install turboloader
```

That's it! On Linux this installs a prebuilt wheel; on other platforms pip falls
back to the source distribution (see below).

PyTorch is an **optional** dependency. Install the extra only if you want the
PyTorch helpers / `output_format='pytorch'` convenience integration:

```bash
pip install turboloader[torch]
```

TurboLoader itself works framework-agnostically with NumPy and also outputs
TensorFlow-style HWC arrays, so `torch` is not required for the core loader.

---

## System Requirements

### Supported Platforms

- **Linux**: Ubuntu 20.04+, Debian 10+, RHEL/CentOS 8+, Amazon Linux 2 —
  prebuilt **manylinux** wheels for `x86_64` and `aarch64`.
- **macOS**: 11+ (Big Sur and later) — portable wheels built from the source
  distribution; self-contained portable binary wheels are being added.
- **Windows**: Not officially supported yet (use WSL2).

A source distribution (sdist) is also published for platforms without a
matching prebuilt wheel.

### Python Versions

- Python 3.10, 3.11, 3.12, 3.13

### Compilers (for building from source)

- **GCC**: 11.0+ (Linux)
- **Clang**: 14.0+ (macOS)
- **MSVC**: 19.29+ (Windows, experimental)

---

## Installation Methods

### Method 1: PyPI (Recommended)

Install the latest stable release:

```bash
pip install turboloader
```

Install a specific version (latest published on PyPI is `2.33.0`):

```bash
pip install turboloader==2.33.0
```

Upgrade to latest:

```bash
pip install --upgrade turboloader
```

With the optional PyTorch integration:

```bash
pip install "turboloader[torch]"
```

### Method 2: From Source

Clone and install in development mode:

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

---

## Platform-Specific Instructions

### Ubuntu / Debian

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libjpeg-turbo8-dev \
    libpng-dev \
    libwebp-dev \
    liblz4-dev \
    libcurl4-openssl-dev

# Install TurboLoader
pip install turboloader
```

### RHEL / CentOS / Fedora

```bash
# Install system dependencies
sudo yum install -y \
    gcc-c++ \
    cmake \
    libjpeg-turbo-devel \
    libpng-devel \
    libwebp-devel \
    lz4-devel \
    libcurl-devel

# Install TurboLoader
pip install turboloader
```

### macOS

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install jpeg-turbo libpng webp lz4 cmake

# Install TurboLoader
pip install turboloader
```

### Windows (WSL2)

Windows is not officially supported yet. Use WSL2 (Windows Subsystem for Linux):

```powershell
# In PowerShell (Administrator)
wsl --install

# Then follow Ubuntu instructions inside WSL2
```

---

## Optional Dependencies

### PyTorch integration

PyTorch is optional. Install the extra to enable the PyTorch helpers and the
`output_format='pytorch'` (CHW tensor) path:

```bash
pip install "turboloader[torch]"
```

The core loader, SIMD transforms, and the NumPy / TensorFlow-HWC output paths
work without PyTorch installed.

### GPU image loader (NVIDIA CUDA) — build from source

> **Prebuilt CUDA wheel (no compile):** Linux x86_64 / Python 3.10 / CUDA 13.x runtime —
> attached to the [GitHub Release](https://github.com/ALJainProjects/TurboLoader/releases/tag/v2.34.1):
> ```bash
> pip install https://github.com/ALJainProjects/TurboLoader/releases/download/v2.34.1/turboloader-2.34.1+cu13-cp310-cp310-linux_x86_64.whl
> pip install nvidia-nvimgcodec-cu12
> ```
> Other Pythons/CUDA versions: build from source below.


> **Not in the published wheels** (they are portable CPU/Metal; CUDA needs a toolkit + GPU at
> build time). Built from source on a CUDA box, `CudaImageLoader(decode="nvimgcodec")` is an
> end-to-end GPU loader on **nvImageCodec** that **beats DALI** on a 3090 (~28.5k vs ~25.5k
> img/s). Build with `nvcc` + the CUDA toolkit (gcc 10+ for C++20):

```bash
pip install nvidia-nvimgcodec-cu12      # nvImageCodec runtime + header (auto-discovered)

CUDA_HOME=/usr/local/cuda \
TURBOLOADER_ENABLE_CUDA=1 \             # transform kernels + cudart
TURBOLOADER_ENABLE_NVJPEG=1 \          # nvJPEG decoder (decode="gpu")
TURBOLOADER_ENABLE_NVIMGCODEC=1 \      # nvImageCodec pipeline (decode="nvimgcodec")
TURBOLOADER_CUDA_ARCH=native \         # required on CUDA 13+; or sm_86 (3090) / sm_87 (Orin)
  pip install -e . --no-build-isolation
```

> `turboloader.cuda_available()` then returns `True`. The header for
> `TURBOLOADER_ENABLE_NVIMGCODEC` is auto-discovered from the installed `nvidia-nvimgcodec-cu12`
> wheel (override with `TURBOLOADER_NVIMGCODEC_INCLUDE`). The CPU decode path remains
> libjpeg-turbo + SIMD (NEON / AVX2 / AVX-512) with automatic DCT scaled decode. See
> [GPU acceleration](GPU_ACCELERATION.md) for the full guide and benchmarks.

### Cloud / specialized storage (source-only / optional)

Cloud and specialized storage backends (S3, GCS, Azure, HDF5, Zarr, TFRecord)
are **not bundled in the prebuilt wheel**. Where supported they are optional,
source-only integrations that require their own client libraries and
credentials. The built-in loaders read local files and WebDataset-style TAR
archives. If you need object storage today, stage data to a local path (or an
NFS mount) and point TurboLoader at it.

---

## Verification

Verify your installation:

```bash
# Run verification script
python scripts/verify_installation.py
```

Or test manually:

```python
import turboloader
import numpy as np

print(f"TurboLoader version: {turboloader.__version__}")

# Test basic functionality
img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
resize = turboloader.Resize(224, 224)
output = resize.apply(img)

print(f"✓ Transform test passed: {output.shape}")
```

---

## Building from Source

### Prerequisites

Install build tools:

**Ubuntu/Debian:**
```bash
sudo apt-get install -y build-essential cmake git
```

**macOS:**
```bash
xcode-select --install
brew install cmake
```

### Build Steps

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Install Python package
cd ..
pip install -e .
```

### Build Options

```bash
# Build with AVX-512 support
cmake -DUSE_AVX512=ON ..

# Release build (optimized)
cmake -DCMAKE_BUILD_TYPE=Release ..
```

---

## Docker

Use TurboLoader in Docker:

```dockerfile
FROM python:3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libjpeg-turbo8 \
    libpng16-16 \
    libwebp7 \
    liblz4-1 \
    && rm -rf /var/lib/apt/lists/*

# Install TurboLoader
RUN pip install turboloader

# Verify installation
RUN python -c "import turboloader; print(turboloader.__version__)"
```

Build and run:

```bash
docker build -t my-turboloader-app .
docker run -it my-turboloader-app
```

---

## Troubleshooting

### "No module named 'turboloader'"

**Solution:** Ensure you're using the correct Python environment:

```bash
which python
pip list | grep turboloader
```

### "ImportError: cannot import name..."

**Solution:** Version mismatch. Reinstall:

```bash
pip uninstall turboloader
pip install turboloader
```

### "Library not loaded" (macOS)

**Solution:** Install system dependencies:

```bash
brew install jpeg-turbo libpng webp lz4
```

### Build errors when installing from source

**Solution:** Ensure you have C++20 compiler:

```bash
# Check compiler version
gcc --version  # Should be 11.0+
clang --version  # Should be 14.0+
```

If compiler is too old, install from PyPI instead (Linux gets a prebuilt
manylinux wheel; other platforms build from the sdist):

```bash
pip install turboloader
```

---

## Next Steps

After installation:

1. **Quick Start**: Read [Quick Start Guide](quickstart.md)
2. **Examples**: Check [examples/](../examples/) directory
3. **API Docs**: See [API Reference](api/index.md)
4. **Verification**: Run `python scripts/verify_installation.py`

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **Troubleshooting**: [Troubleshooting Guide](TROUBLESHOOTING.md)
