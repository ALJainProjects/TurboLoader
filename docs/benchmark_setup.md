# Benchmark Setup Guide

This guide explains how to set up all the data loading frameworks for benchmarking against TurboLoader.

## Table of Contents

1. [Core Requirements](#core-requirements)
2. [TurboLoader Setup](#turboloader-setup)
3. [PyTorch Setup](#pytorch-setup)
4. [TensorFlow Setup](#tensorflow-setup)
5. [FFCV Setup](#ffcv-setup)
6. [NVIDIA DALI Setup](#nvidia-dali-setup)
7. [Running Benchmarks](#running-benchmarks)

---

## Core Requirements

### System Requirements
- Python 3.9+
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- CMake 3.15+
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA 11.0+ for DALI

### Python Dependencies

```bash
pip install numpy pillow psutil
```

---

## TurboLoader Setup

### Build from Source

```bash
# Clone repository
git clone https://github.com/yourusername/turboloader.git
cd turboloader

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Install Python bindings
cd ..
pip install -e .
```

### Verify Installation

```bash
python3 -c "import _turboloader; print('TurboLoader installed successfully')"
```

---

## PyTorch Setup

### Standard Installation

```bash
# CPU-only version
pip install torch torchvision

# CUDA 11.8 version (for GPU support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### macOS (Apple Silicon)

```bash
pip install torch torchvision
```

### Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

---

## TensorFlow Setup

### Standard Installation

```bash
# CPU-only version
pip install tensorflow

# GPU version (requires CUDA 11.2+, cuDNN 8.1+)
pip install tensorflow[and-cuda]
```

### macOS (Apple Silicon)

```bash
# TensorFlow with Metal acceleration
pip install tensorflow-macos
pip install tensorflow-metal
```

### Verify Installation

```bash
python3 -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
```

---

## FFCV Setup

FFCV (Fast Forward Computer Vision) is a high-performance data loading library from MIT.

### Prerequisites

FFCV requires:
- **Rust/Cargo** for compilation
- **NumPy** and **PyTorch**
- **libjpeg-turbo** for fast JPEG decoding

### Install Rust (if not already installed)

```bash
# Linux/macOS
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify Rust installation
rustc --version
cargo --version
```

### Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libturbojpeg-dev
```

**macOS:**
```bash
brew install jpeg-turbo pkg-config
```

**Fedora/RHEL:**
```bash
sudo dnf install -y \
    gcc-c++ \
    pkg-config \
    libjpeg-turbo-devel \
    libpng-devel
```

### Install FFCV

```bash
# Install FFCV from PyPI
pip install ffcv

# Or build from source for latest features
git clone https://github.com/libffcv/ffcv.git
cd ffcv
pip install -e .
```

### Verify Installation

```bash
python3 -c "from ffcv.loader import Loader; print('FFCV installed successfully')"
```

### FFCV Dataset Preparation

FFCV requires converting datasets to `.beton` format:

```bash
# Convert images to FFCV format
python3 << EOF
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from pathlib import Path

# Create writer
writer = DatasetWriter('dataset.beton', {
    'image': RGBImageField(max_resolution=256),
    'label': IntField()
})

# Write dataset
# ... (see benchmark scripts for full example)
EOF
```

### Common FFCV Issues

**Issue: "cargo not found"**
```bash
# Make sure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

**Issue: "libjpeg-turbo not found"**
```bash
# Ubuntu/Debian
sudo apt-get install libturbojpeg-dev

# macOS
brew install jpeg-turbo
export PKG_CONFIG_PATH="/opt/homebrew/opt/jpeg-turbo/lib/pkgconfig:$PKG_CONFIG_PATH"
```

---

## NVIDIA DALI Setup

NVIDIA DALI (Data Loading Library) provides GPU-accelerated data loading.

### Prerequisites

DALI requires:
- **NVIDIA GPU** with compute capability 3.5+
- **CUDA 11.0+** or **CUDA 12.0+**
- **cuDNN** (automatically handled by DALI package)
- **Driver version** 450.80.02+ (Linux) or 452.39+ (Windows)

### Check GPU Compatibility

```bash
# Check CUDA version
nvcc --version

# Check GPU compute capability
nvidia-smi

# Check driver version
nvidia-smi --query-gpu=driver_version --format=csv
```

### Install DALI

**For CUDA 11.x:**
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    nvidia-dali-cuda110
```

**For CUDA 12.x:**
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    nvidia-dali-cuda120
```

**CPU-only version (for testing):**
```bash
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    nvidia-dali
```

### Verify Installation

```bash
python3 << EOF
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
print('NVIDIA DALI installed successfully')
EOF
```

### DALI GPU Requirements

**Minimum Requirements:**
- GPU: GTX 1050 or better
- VRAM: 2GB+ recommended
- CUDA: 11.0+

**Optimal Performance:**
- GPU: RTX 3060 or better
- VRAM: 6GB+
- CUDA: 12.0+
- NVMe SSD for data storage

### Common DALI Issues

**Issue: "CUDA not available"**
```bash
# Check CUDA installation
which nvcc
nvcc --version

# Verify GPU is detected
nvidia-smi
```

**Issue: "Driver version mismatch"**
```bash
# Update NVIDIA drivers
# Ubuntu/Debian
sudo ubuntu-drivers autoinstall

# Or manually from NVIDIA website
# https://www.nvidia.com/Download/index.aspx
```

**Issue: "cuDNN library not found"**
- DALI packages include cuDNN, but if you get this error:
```bash
# Download cuDNN from NVIDIA Developer
# https://developer.nvidia.com/cudnn

# Install cuDNN (example for CUDA 11.8)
tar -xzvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

---

## Running Benchmarks

### Generate Test Dataset

```bash
# Generate 2000-image benchmark dataset
python3 scripts/generate_benchmark_dataset.py \
    --num-images 2000 \
    --output /tmp/benchmark_datasets/bench_2k \
    --image-size 256 256 \
    --quality 90
```

### Run Individual Benchmarks

```bash
# PIL Baseline
python3 benchmarks/01_pil_baseline.py \
    /tmp/benchmark_datasets/bench_2k/images \
    --epochs 3 --batch-size 32

# PyTorch Naive
python3 benchmarks/02_pytorch_naive.py \
    /tmp/benchmark_datasets/bench_2k/images \
    --epochs 3 --batch-size 32 --num-workers 4

# PyTorch Optimized
python3 benchmarks/03_pytorch_optimized.py \
    /tmp/benchmark_datasets/bench_2k/images \
    --epochs 3 --batch-size 32 --num-workers 8

# PyTorch with Caching
python3 benchmarks/04_pytorch_cached.py \
    /tmp/benchmark_datasets/bench_2k/dataset.tar \
    --epochs 3 --batch-size 32 --num-workers 8

# TurboLoader
python3 benchmarks/05_turboloader.py \
    /tmp/benchmark_datasets/bench_2k/dataset.tar \
    --epochs 3 --batch-size 32 --num-workers 8

# FFCV (requires .beton conversion first)
python3 benchmarks/06_ffcv.py \
    /tmp/benchmark_datasets/bench_2k/images \
    --epochs 3 --batch-size 32

# NVIDIA DALI (GPU required)
python3 benchmarks/07_dali.py \
    /tmp/benchmark_datasets/bench_2k/dataset.tar \
    --epochs 3 --batch-size 32 --device gpu

# TensorFlow
python3 benchmarks/08_tensorflow.py \
    /tmp/benchmark_datasets/bench_2k/dataset.tar \
    --epochs 3 --batch-size 32 --num-workers 8
```

### Run All Benchmarks

```bash
# Run comprehensive benchmark suite
./scripts/run_all_benchmarks.sh
```

---

## Performance Tuning

### System-Level Optimizations

**Increase File Descriptor Limit:**
```bash
# Temporary
ulimit -n 65536

# Permanent (add to ~/.bashrc or /etc/security/limits.conf)
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

**Disable CPU Frequency Scaling:**
```bash
# Linux
sudo cpupower frequency-set -g performance

# Verify
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Use Faster Storage:**
- NVMe SSD > SATA SSD > HDD
- RAMDisk for maximum performance:
```bash
# Create 4GB RAMDisk (Linux)
sudo mount -t tmpfs -o size=4g tmpfs /tmp/ramdisk
```

### Framework-Specific Tuning

**PyTorch:**
```bash
# Set optimal thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable TF32 on Ampere GPUs
export NVIDIA_TF32_OVERRIDE=1
```

**TensorFlow:**
```bash
# Control parallelism
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=8

# Enable XLA JIT compilation
export TF_XLA_FLAGS=--tf_xla_auto_jit=2
```

**DALI:**
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Enable async execution
# (controlled in pipeline code)
```

---

## Troubleshooting

### General Issues

**Import errors:**
```bash
# Verify Python can find packages
python3 -c "import sys; print('\n'.join(sys.path))"

# Reinstall package
pip install --force-reinstall <package>
```

**Permission errors:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install <packages>
```

**Out of memory:**
```bash
# Reduce batch size
--batch-size 16  # instead of 32

# Reduce number of workers
--num-workers 4  # instead of 8
```

### Framework-Specific Issues

**PyTorch "too many open files":**
```bash
# Increase ulimit
ulimit -n 65536

# Reduce num_workers
--num-workers 4
```

**TensorFlow GPU not detected:**
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**FFCV "compilation failed":**
```bash
# Make sure Rust is updated
rustup update

# Clear build cache
pip cache purge
pip install --no-cache-dir ffcv
```

**DALI "CUDA error":**
```bash
# Check CUDA compatibility
nvidia-smi
nvcc --version

# Match DALI CUDA version to system CUDA
pip install nvidia-dali-cuda120  # for CUDA 12.x
```

---

## Reference Documentation

- **TurboLoader**: [GitHub](https://github.com/yourusername/turboloader)
- **PyTorch DataLoader**: [Docs](https://pytorch.org/docs/stable/data.html)
- **TensorFlow tf.data**: [Guide](https://www.tensorflow.org/guide/data)
- **FFCV**: [Docs](https://docs.ffcv.io/)
- **NVIDIA DALI**: [Docs](https://docs.nvidia.com/deeplearning/dali/)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/turboloader/issues)
3. Create a new issue with:
   - System information (`uname -a`, `python --version`)
   - Error message and full traceback
   - Benchmark script being run
   - Dataset configuration
