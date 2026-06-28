# Benchmark Setup Guide

This guide explains how to set up the data loading frameworks used in TurboLoader's
honest, measured comparison: **PyTorch `DataLoader`** and **TensorFlow `tf.data`**.

> **Scope note:** FFCV and NVIDIA DALI head-to-head comparisons are future work and
> have **not** been measured yet. They are intentionally omitted here so the setup
> guide only covers frameworks whose numbers we actually report. There is no GPU /
> nvJPEG path in the shipped wheel.

## Table of Contents

1. [Core Requirements](#core-requirements)
2. [TurboLoader Setup](#turboloader-setup)
3. [PyTorch Setup](#pytorch-setup)
4. [TensorFlow Setup](#tensorflow-setup)
5. [Running Benchmarks](#running-benchmarks)

---

## Core Requirements

### System Requirements
- Python 3.9+
- Linux x86_64 / aarch64, or macOS (Apple Silicon)
- 8GB+ RAM recommended
- (Optional, source builds only) C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+) and CMake 3.15+

### Python Dependencies

```bash
pip install numpy pillow psutil
```

---

## TurboLoader Setup

### Install from PyPI (recommended)

```bash
# Prebuilt manylinux wheels are published for Linux x86_64 and aarch64 (plus an sdist).
# Portable macOS wheels built from source are being added.
pip install turboloader

# torch is OPTIONAL — only needed for the PyTorch output path / torch interop:
pip install turboloader[torch]
```

### Build from Source (optional)

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

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
python3 -c "import turboloader; print('TurboLoader installed successfully')"
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

## Running Benchmarks

### Prepare the Dataset

The published numbers use **Imagenette-160** — 9,469 real ImageNet JPEGs resized to
160 px. Download the fast.ai Imagenette-160 archive and use its `train` split:

```bash
# Download and extract Imagenette-160 (~98 MB)
curl -L -o imagenette2-160.tgz \
    https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf imagenette2-160.tgz
# The train split (imagenette2-160/train) contains 9,469 JPEGs.
```

### Run Individual Benchmarks

All loaders are run with **batch size 64**, `output_format='pytorch'`, one warmup epoch
and the median of 3 timed epochs under real consumption.

```bash
# PIL baseline
python3 benchmarks/01_pil_baseline.py \
    imagenette2-160/train \
    --epochs 3 --batch-size 64

# PyTorch optimized (8 persistent workers — the reported PyTorch number)
python3 benchmarks/03_pytorch_optimized.py \
    imagenette2-160/train \
    --epochs 3 --batch-size 64 --num-workers 8

# TensorFlow tf.data (AUTOTUNE)
python3 benchmarks/08_tensorflow.py \
    imagenette2-160/train \
    --epochs 3 --batch-size 64

# TurboLoader (single process-wide C++ thread pool; on-the-fly)
python3 benchmarks/05_turboloader.py \
    imagenette2-160/train \
    --epochs 3 --batch-size 64

# TurboLoader with decoded cache (cache_decoded=True)
python3 benchmarks/05_turboloader.py \
    imagenette2-160/train \
    --epochs 3 --batch-size 64 --cache-decoded
```

> Note: `--num-workers` is only meaningful for PyTorch (separate OS processes).
> TurboLoader's fast path is a single saturated C++ thread pool, and `tf.data` uses
> AUTOTUNE, so neither is swept over worker counts.

### Run All Benchmarks

```bash
# Run the comparison suite
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
```

**TensorFlow:**
```bash
# Control parallelism
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=8

# Enable XLA JIT compilation
export TF_XLA_FLAGS=--tf_xla_auto_jit=2
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
--batch-size 32  # instead of 64

# Reduce number of workers (PyTorch only)
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

---

## Reference Documentation

- **TurboLoader**: [GitHub](https://github.com/ALJainProjects/TurboLoader)
- **PyTorch DataLoader**: [Docs](https://pytorch.org/docs/stable/data.html)
- **TensorFlow tf.data**: [Guide](https://www.tensorflow.org/guide/data)

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
3. Create a new issue with:
   - System information (`uname -a`, `python --version`)
   - Error message and full traceback
   - Benchmark script being run
   - Dataset configuration
