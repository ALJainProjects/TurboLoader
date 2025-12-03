# Installation Guide

Complete installation instructions for TurboLoader on all supported platforms.

## Quick Install

```bash
pip install turboloader
```

That's it! TurboLoader will be installed with pre-built wheels for most platforms.

---

## System Requirements

### Supported Platforms

- **Linux**: Ubuntu 20.04+, Debian 10+, RHEL/CentOS 8+, Amazon Linux 2
- **macOS**: 11+ (Big Sur and later)
- **Windows**: Not officially supported yet (use WSL2)

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

Install a specific version:

```bash
pip install turboloader==2.7.0
```

Upgrade to latest:

```bash
pip install --upgrade turboloader
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

### GPU Support (NVIDIA nvJPEG)

For GPU-accelerated JPEG decoding:

```bash
# Install CUDA Toolkit (11.0+)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install TurboLoader (will detect CUDA automatically)
pip install turboloader
```

### Cloud Storage (S3, GCS)

Dependencies are included by default. Configure credentials:

**AWS S3:**
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

**Google Cloud Storage:**
```bash
# Install gcloud SDK
# Follow: https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
```

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
# Build with GPU support
cmake -DUSE_CUDA=ON ..

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

If compiler is too old, install from PyPI instead:

```bash
pip install turboloader  # Uses pre-built wheels
```

### Missing CUDA libraries (GPU support)

**Solution:** Install CUDA Toolkit:

```bash
# Download from nvidia.com
# Then reinstall TurboLoader
pip install --force-reinstall turboloader
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
