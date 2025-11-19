# TurboLoader

**Production-Ready ML Data Loading Library**

[![PyPI version](https://badge.fury.io/py/turboloader.svg)](https://pypi.org/project/turboloader/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TurboLoader is a high-performance data loading library for machine learning workflows. Built with C++20 and featuring Python bindings, it provides efficient data loading with SIMD-accelerated transforms, custom binary formats, and distributed training support.

### Core Features

- **Distributed Training Support** - Multi-node data loading with deterministic sharding (v1.7.1)
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

- **Python:** 3.8 or higher
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

- **[Installation Guide](docs/installation.md)** - Detailed setup instructions
- **[Quick Start](docs/quickstart.md)** - Getting started examples
- **[API Reference](docs/api/)** - Complete API documentation
- **[PyTorch Integration](docs/pytorch.md)** - PyTorch-specific examples
- **[Distributed Training](docs/distributed.md)** - Multi-node setup guide

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
  version = {1.7.1},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```

---

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)

---

TurboLoader v1.7.1 - Production-ready ML data loading. Fast. Efficient. Reliable.
