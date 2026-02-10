# TurboLoader

**Production-Ready ML Data Loading Library**

[![PyPI version](https://img.shields.io/pypi/v/turboloader.svg)](https://pypi.org/project/turboloader/)
[![Tests](https://github.com/ALJainProjects/TurboLoader/actions/workflows/test.yml/badge.svg)](https://github.com/ALJainProjects/TurboLoader/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TurboLoader is a high-performance data loading library for machine learning workflows. Built with C++20 and featuring Python bindings, it provides efficient data loading with SIMD-accelerated transforms, custom binary formats, and distributed training support.

### Core Features

- **Decoded Tensor Caching (v2.7.0)** - `cache_decoded=True` for 100K+ img/s on subsequent epochs
- **Multiple Loader Types** - FastDataLoader (8-12% faster), MemoryEfficientDataLoader, standard DataLoader
- **Distributed Training Support** - Multi-node data loading with deterministic sharding
- **SIMD-Accelerated Transforms** - 19 vectorized transforms using AVX2/AVX-512/NEON
- **TBL v2 Binary Format** - Custom format with LZ4 compression for reduced storage
- **Framework Integration** - Seamless support for PyTorch, TensorFlow, and JAX
- **Memory-Mapped I/O** - Zero-copy file access for improved throughput
- **Lock-Free Queues** - Concurrent data structures for efficient multi-threading
- **GPU JPEG Decoding** - Optional NVIDIA nvJPEG support for accelerated decoding

---

## Installation

### From PyPI (Recommended)

```bash
pip install turboloader
```

### From Source

```bash
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader
pip install -e .
```

### System Requirements

- **Python:** 3.10 or higher
- **Compiler:** C++20 capable (GCC 10+, Clang 12+, MSVC 19.29+)
- **OS:** macOS, Linux, Windows

#### Optional Dependencies

Install for enhanced performance:

```bash
# macOS
brew install jpeg-turbo libpng libwebp lz4

# Ubuntu/Debian
sudo apt-get install libjpeg-turbo8-dev libpng-dev libwebp-dev liblz4-dev
```

---

## Quick Start

### Basic Usage

```python
import turboloader

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=128,
    num_workers=8
)

# Iterate over batches
for batch in loader:
    for sample in batch:
        image = sample['image']  # NumPy array (H, W, C)
        label = sample['label']
        # Train your model...
```

### With Transforms

```python
import turboloader

# Create transforms
resize = turboloader.Resize(224, 224)
normalize = turboloader.ImageNetNormalize()
flip = turboloader.RandomHorizontalFlip(p=0.5)

# Apply transforms
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

for batch in loader:
    for sample in batch:
        img = sample['image']
        img = resize.apply(img)
        img = flip.apply(img)
        img = normalize.apply(img)
        # Ready for training
```

### PyTorch Integration

```python
import turboloader
import torch

loader = turboloader.DataLoader('imagenet.tar', batch_size=64, num_workers=8)

# Convert to PyTorch tensors
to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW
)

for batch in loader:
    images = []
    for sample in batch:
        img = to_tensor.apply(sample['image'])
        images.append(torch.from_numpy(img))

    batch_tensor = torch.stack(images)
    # Train model...
```

### Distributed Training

```python
import turboloader
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Create loader with distributed support
loader = turboloader.DataLoader(
    data_path="/data/imagenet.tar",
    batch_size=64,
    num_workers=4,
    shuffle=True,
    enable_distributed=True,
    world_rank=dist.get_rank(),
    world_size=dist.get_world_size(),
    drop_last=True
)

# Each rank automatically gets its shard
for batch in loader:
    # Your training code
    pass
```

---

## Transform Library

TurboLoader includes 19 SIMD-accelerated transforms:

### Core Transforms
- **Resize** - Bilinear/Bicubic/Lanczos interpolation
- **Normalize** - Mean/std normalization with SIMD
- **CenterCrop** - Center region extraction
- **RandomCrop** - Random crop with padding

### Augmentation Transforms
- **RandomHorizontalFlip** - SIMD horizontal flip
- **RandomVerticalFlip** - SIMD vertical flip
- **ColorJitter** - Brightness/contrast/saturation/hue
- **RandomRotation** - Arbitrary angle rotation
- **GaussianBlur** - Separable convolution
- **RandomErasing** - Cutout augmentation
- **Pad** - Border padding (CONSTANT/EDGE/REFLECT)

### Advanced Transforms
- **RandomPosterize** - Bit-depth reduction
- **RandomSolarize** - Threshold inversion
- **RandomPerspective** - Perspective warp
- **AutoAugment** - Learned policies (ImageNet/CIFAR10/SVHN)

### Tensor Conversion
- **ToTensor** - PyTorch CHW or TensorFlow HWC format

---

## TBL v2 Binary Format

TurboLoader includes a custom binary format optimized for ML workloads:

### Features
- LZ4 compression for reduced storage
- Memory-mapped access for fast loading
- O(1) random access via indexed structure
- Data integrity validation with CRC checksums
- Cached image dimensions for filtered loading

### Convert TAR to TBL

```python
import turboloader

writer = turboloader.TblWriterV2(
    output_path="/data/imagenet.tbl",
    compression=True
)

reader = turboloader.TarReader("/data/imagenet.tar")
for sample in reader:
    writer.add_sample(
        data=sample.data,
        format=sample.format,
        metadata={"label": sample.label}
    )

writer.finalize()
```

---

## Documentation

### Getting Started
- **[Quick Start Notebook](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/quickstart.ipynb)** - Interactive tutorial for beginners
- **[Installation Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/installation.md)** - Detailed setup instructions
- **[Quick Start](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/quickstart.md)** - Getting started examples
- **[Troubleshooting Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions

### API Documentation
- **[API Reference](https://github.com/ALJainProjects/TurboLoader/tree/main/docs/api)** - Complete API documentation
- **[Transforms API](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/api/transforms.md)** - All 19 transforms with examples

### Framework Integration
- **[PyTorch Integration Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/guides/pytorch-integration.md)** - Complete PyTorch guide
- **[TensorFlow Integration Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/guides/tensorflow-integration.md)** - Complete TensorFlow/Keras guide
- **[PyTorch Lightning Example](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/pytorch_lightning_example.py)** - Production-ready Lightning integration
- **[Distributed Training (DDP)](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/distributed_ddp.py)** - Multi-GPU PyTorch DDP example

### Examples
- **[ImageNet ResNet50 Training](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/imagenet_resnet50.py)** - Complete training pipeline with AMP, checkpointing, TensorBoard
- **[Distributed Training](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/distributed.md)** - Multi-node setup guide

---

## Benchmarks

Head-to-head comparison with **optimized** PyTorch DataLoader (`persistent_workers=True`, `prefetch_factor=4`). Both loaders tested under identical conditions.

### vs PyTorch DataLoader (BS=32, NW=4)

| Configuration | TurboLoader | PyTorch | Speedup |
|---|---|---|---|
| uint8 CHW (resize only) | 8,027 img/s | 2,457 img/s | **3.3x** |
| float32 CHW (0-1 normalize) | 8,456 img/s | 2,040 img/s | **4.1x** |
| float32 CHW + ImageNet mean/std | 8,029 img/s | 2,039 img/s | **3.9x** |

### Decoded Tensor Caching (`cache_decoded=True`)

| Configuration | Epoch 2 Throughput |
|---|---|
| uint8 HWC (from cache) | 57,692,695 img/s |
| float32 CHW (from cache) | 42,933,573 img/s |
| float32 CHW + ImageNet (from cache) | 39,853,643 img/s |

### Worker Scaling (BS=32, float32 CHW + ImageNet)

| Workers | TurboLoader | PyTorch | Speedup |
|---|---|---|---|
| 1 worker | 1,585 img/s | 625 img/s | **2.5x** |
| 2 workers | 3,383 img/s | 1,184 img/s | **2.9x** |
| 4 workers | 7,744 img/s | 2,016 img/s | **3.8x** |
| 8 workers | 13,327 img/s | 3,047 img/s | **4.4x** |

### Batch Size Scaling (NW=4, float32 CHW + ImageNet)

| Batch Size | TurboLoader | PyTorch | Speedup |
|---|---|---|---|
| 8 | 7,997 img/s | 2,342 img/s | **3.4x** |
| 16 | 8,280 img/s | 2,261 img/s | **3.7x** |
| 32 | 7,418 img/s | 1,946 img/s | **3.8x** |
| 64 | 7,896 img/s | 1,765 img/s | **4.5x** |
| 128 | 7,841 img/s | 1,521 img/s | **5.2x** |

**Test conditions:** Apple M4 Pro, 5000 JPEG images (640x480), best of 3 trials, 100 batches per trial. PyTorch uses `persistent_workers=True`, `prefetch_factor=4`.

### Key Optimizations

- **OpenMP parallelism** for batch assembly (decode, resize, transpose, convert)
- **Fused SIMD deinterleave**: NEON `vld3q_u8` for HWC→CHW + u8→f32 + normalize in a single pass
- **Thread-local buffers** to eliminate per-sample heap allocation under OpenMP
- **Pipeline reset** reuses buffer pools, decoders, and memory maps across epochs
- **LTO (thin)** for cross-TU inlining of SIMD functions
- **GIL released** during all C++ processing

> **Note:** Actual throughput depends on your hardware, image sizes, and pipeline configuration.
> Run the benchmark on your setup for precise figures.

---

## Architecture

TurboLoader uses a multi-threaded pipeline architecture:

```
┌─────────────────────────────────────────────┐
│           Memory-Mapped Reader              │
│     (TAR/TBL v2 with zero-copy access)      │
└──────────────┬──────────────────────────────┘
               │
        ┌──────▼──────┐
        │Worker Pool  │
        │  (N threads)│
        ├─────────────┤
        │ Decode      │
        │ Transform   │
        │ Convert     │
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │ Lock-Free Queue     │
        └──────┬──────────────┘
               │
        ┌──────▼──────┐
        │Python API   │
        └─────────────┘
```

### Key Components

- **Memory-Mapped I/O** - Zero-copy file access
- **Worker Thread Pool** - Parallel processing with per-thread decoders
- **SIMD Transforms** - Vectorized operations (AVX2/AVX-512/NEON)
- **Lock-Free Queues** - High-performance concurrent data structures

---

## License

TurboLoader is released under the MIT License.

---

## Citation

If you use TurboLoader in your research:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: Production-Ready ML Data Loading},
  year = {2025},
  version = {2.7.0},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```

---

## Support

- **Documentation:** [https://github.com/ALJainProjects/TurboLoader/tree/main/docs](https://github.com/ALJainProjects/TurboLoader/tree/main/docs)
- **Troubleshooting:** [https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md)
- **Verification Script:** Run `python scripts/verify_installation.py` to check your setup
- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)

---

TurboLoader - Production-ready ML data loading. 2.5-5.2x faster than optimized PyTorch DataLoader.
