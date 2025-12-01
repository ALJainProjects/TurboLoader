# TurboLoader Documentation

Welcome to TurboLoader - the fastest ML data loading library with SIMD-accelerated transforms.

## Quick Links

- [Getting Started](getting-started.md)
- [API Reference](api/index.md)
- [Benchmarks](benchmarks/index.md)
- [Architecture](architecture.md)
- [GitHub Repository](https://github.com/ALJainProjects/TurboLoader)

## Overview

TurboLoader achieves **21,035 img/s peak throughput** (12x faster than PyTorch) through:

- **Native C++20 implementation** - No Python GIL overhead
- **TBL v2 Binary Format** - LZ4 compression (40-60% space savings), O(1) streaming writer, 4,875 img/s conversion
- **Data Integrity** - CRC32/CRC16 checksums for reliable data loading
- **Cached Dimensions** - Width/height in index for fast filtering without decoding
- **19 SIMD-accelerated transforms** - AVX-512/AVX2/NEON optimized operations
- **Smart Batching** - Size-based grouping reduces padding by 15-25% (~1.2x boost)
- **Distributed Training** - Multi-node support (PyTorch DDP, Horovod, DeepSpeed)
- **Linear Scalability** - 9.65x speedup with 16 workers
- **Zero-copy tensor conversion** - Direct memory mapping to PyTorch/TensorFlow
- **Lock-free concurrent queues** - Maximizes multi-core utilization
- **Memory-mapped I/O** - Efficient TAR/TBL parsing (52+ Gbps)
- **AutoAugment policies** - State-of-the-art learned augmentation

## Key Features

### Performance

**Latest Results (v2.0.0):**

| Workers | Throughput | Linear Scaling | Efficiency |
|---------|------------|----------------|------------|
| 1 | 2,180 img/s | 1.00x | 100% |
| 2 | 4,020 img/s | 1.84x | 92% |
| 4 | 6,755 img/s | 3.10x | 77% |
| 8 | 6,973 img/s | 3.20x | 40% |
| **16** | **21,036 img/s** | **9.65x** | **60%** |

**Test Config:** Apple M4 Max, 1000 images, batch_size=64

**Framework Comparison:** 12x faster than PyTorch Optimized (39 img/s baseline)

See [Benchmark Results](benchmarks/index.md) for detailed analysis.

### Transform Library

TurboLoader includes 19 SIMD-accelerated transforms:

**Core Transforms:**
- Resize (4 interpolation modes including Lanczos)
- Normalize (ImageNet presets)
- CenterCrop / RandomCrop
- RandomHorizontalFlip / RandomVerticalFlip
- Pad

**Augmentation Transforms:**
- ColorJitter
- RandomRotation
- RandomAffine
- GaussianBlur
- RandomErasing
- Grayscale

**Advanced Transforms (v0.7.0+):**
- RandomPosterize (336K+ img/s)
- RandomSolarize (21K+ img/s)
- RandomPerspective (9.9K+ img/s)
- AutoAugment (ImageNet, CIFAR10, SVHN policies)

**Tensor Conversion:**
- ToTensor (PyTorch CHW, TensorFlow HWC)

See [Transforms API](api/transforms.md) for complete reference.

## Installation

```bash
pip install turboloader
```

**Requirements:**
- Python 3.8+
- C++20 compiler
- libjpeg-turbo, libpng, libwebp (optional but recommended)

See [Installation Guide](guides/installation.md) for detailed instructions.

## Quick Start

### Basic Usage

```python
import turboloader

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=32,
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

# Create transform pipeline
resize = turboloader.Resize(224, 224, turboloader.InterpolationMode.BILINEAR)
normalize = turboloader.ImageNetNormalize(to_float=True)
flip = turboloader.RandomHorizontalFlip(p=0.5)

# Apply to images
for batch in loader:
    for sample in batch:
        img = sample['image']
        img = resize.apply(img)
        img = flip.apply(img)
        img = normalize.apply(img)
```

### PyTorch Integration

```python
import turboloader
import torch

# Create loader with tensor conversion
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW,
    normalize=True
)

for batch in loader:
    images = []
    for sample in batch:
        img = to_tensor.apply(sample['image'])
        images.append(torch.from_numpy(img))

    batch_tensor = torch.stack(images)
    # Train model...
```

See [PyTorch Integration Guide](guides/pytorch-integration.md) for more examples.

## Documentation Sections

### API Reference

- [Pipeline API](api/pipeline.md) - DataLoader and pipeline configuration
- [Transforms API](api/transforms.md) - Complete transform reference
- [Tensor Conversion](api/tensor-conversion.md) - PyTorch/TensorFlow integration

### Guides

- [Installation](guides/installation.md) - Detailed installation instructions
- [Basic Usage](guides/basic-usage.md) - Getting started examples
- [Advanced Usage](guides/advanced-usage.md) - Complex pipelines and patterns
- [PyTorch Integration](guides/pytorch-integration.md) - PyTorch-specific guide
- [TensorFlow Integration](guides/tensorflow-integration.md) - TensorFlow-specific guide

### Benchmarks

- [Overview](benchmarks/index.md) - Performance results
- [Methodology](benchmarks/methodology.md) - How we benchmark
- [Results](benchmarks/results.md) - Detailed performance data
- [Memory Profiling](benchmarks/memory-profiling.md) - Memory usage analysis

### Development

- [Contributing](development/contributing.md) - Contribution guidelines
- [Building from Source](development/building.md) - Build instructions
- [Testing](development/testing.md) - Running and writing tests

## Architecture

TurboLoader uses a high-performance multi-threaded pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ TAR Reader  │────▶│ Decode Pool  │────▶│  Transform    │
│ (mmap I/O)  │     │ (libjpeg-turbo)│   │  Pipeline     │
└─────────────┘     └──────────────┘     │ (SIMD ops)    │
                                         └───────┬───────┘
                                                 │
                                         ┌───────▼───────┐
                                         │  Tensor       │
                                         │  Conversion   │
                                         │ (PyTorch/TF)  │
                                         └───────────────┘
```

**Key Components:**
1. **Memory-mapped TAR reader** - Zero-copy file access
2. **Thread-local decoders** - Per-worker JPEG/PNG/WebP decoders
3. **SIMD transform pipeline** - Vectorized image operations
4. **Lock-free queues** - High-throughput sample passing

See [Architecture Guide](architecture.md) for detailed design.

## Version History

- **v2.0.0** (Current) - Tiered Caching (L1 memory + L2 disk), Smart Batching enabled by default, Pipeline tuning optimizations
- **v1.9.0** - Transform Pipe Operator, HDF5/TFRecord/Zarr support (headers), COCO/VOC annotations, Azure Blob Storage, GPU transforms, io_uring
- **v1.8.0** - Modern Augmentations (MixUp, CutMix, Mosaic, RandAugment, GridMask), Logging system
- **v1.7.7** - Developer experience improvements: Issue templates, Quick Start notebook, PyTorch Lightning example
- **v1.5.0** - TBL v2 Format with LZ4 compression (40-60% space savings)
- **v1.2.0** - Smart Batching + Distributed Training (21,035 img/s peak)
- **v1.1.0** - AVX-512 SIMD + Binary Format Improvements + Prefetching
- **v1.0.0** - Production/Stable Release (10,146 img/s)

See [CHANGELOG](../CHANGELOG.md) for complete history.

## License

TurboLoader is released under the [MIT License](../LICENSE).

## Support

- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)
- **Email:** arnavjain@example.com

## Citation

If you use TurboLoader in your research:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance ML Data Loading},
  year = {2025},
  version = {2.0.0},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```
