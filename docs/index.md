# TurboLoader Documentation

Welcome to TurboLoader - a high-performance, multi-modal ML data loading library with SIMD-accelerated transforms.

## Quick Links

- [Getting Started](getting-started.md)
- [API Reference](api/index.md)
- [Benchmarks](benchmarks/index.md)
- [Architecture](architecture.md)
- [GitHub Repository](https://github.com/ALJainProjects/TurboLoader)

## Overview

TurboLoader is a C++20 data-loading library for ML featuring:

- **Native C++20 implementation** - GIL released during processing
- **FFCV / tf.data-style direct-batch loader** - one parallel pass decodes, resizes, and normalizes straight into the output batch buffer, with automatic libjpeg-turbo DCT scaled decode for large images
- **GPU image loaders** - `CudaImageLoader` (NVIDIA nvImageCodec — **beats DALI** on a 3090, ~28.5k vs ~25.5k img/s) and `GpuImageLoader` (Apple Metal); end-to-end GPU decode + resize + normalize. See [GPU acceleration](GPU_ACCELERATION.md). CUDA is build-from-source
- **Multi-modality** - images in WebDataset TAR, LLM token streams via `TokenDataLoader`, and generic `(N, ...)` arrays via `ArrayDataLoader`
- **TBL v2 Binary Format** - LZ4 compression, streaming writer, CRC integrity checks
- **Cached Dimensions** - Width/height in index for fast filtering without decoding
- **24 transforms** - 19 per-image SIMD (AVX-512/AVX2/NEON) + 5 batch augmentations
- **PIL/PyTorch/TF-matched resize** - half-pixel sampling with optional antialiasing
- **Smart Batching** - Size-based grouping to reduce padding
- **Distributed sharding** - DDP-safe equal/disjoint sharding for PyTorch DDP
- **Decoded cache** - reuse decoded arrays across epochs (`cache_decoded=True`)
- **NumPy / PyTorch-CHW / TensorFlow-HWC output**
- **Lock-free concurrent queues** - SPSC worker pipeline
- **Memory-mapped I/O** - TAR/TBL parsing
- **AutoAugment policies** - learned augmentation

> Throughput is hardware- and pipeline-dependent, so run `benchmarks/` on your own
> setup. On a fair Apple-Silicon benchmark (Imagenette-160: 9,469 real ImageNet
> JPEGs decoded to 160px, batch 64, real consumption, median of 3 epochs)
> TurboLoader's on-the-fly path reached ~39,100 img/s versus ~30,154 img/s for
> TensorFlow `tf.data` (AUTOTUNE) and ~18,991 img/s for a PyTorch DataLoader with
> 8 persistent workers.

## Key Features

### Performance

Throughput depends on hardware, image size, and pipeline configuration, so
**run `benchmarks/` on your own machine**. The numbers below were measured on
Apple Silicon over Imagenette-160 (9,469 real ImageNet JPEGs decoded to 160px,
`output_format='pytorch'`, batch 64, real consumption forcing materialization,
warmup + median of 3 epochs):

| Loader | Throughput | Speedup |
|--------|-----------:|:-------:|
| TurboLoader cached (`cache_decoded=True`) | ~65,499 img/s | - |
| TurboLoader on-the-fly | ~39,100 img/s | - |
| TensorFlow `tf.data` (AUTOTUNE) | ~30,154 img/s | TurboLoader 1.3x |
| PyTorch DataLoader (8 persistent workers) | ~18,991 img/s | TurboLoader 2.1x |

For LLM token streams, `TokenDataLoader` reached ~441M tokens/s versus ~163M
tokens/s for the NumPy memmap idiom (2.7x).

Worker-scaling nuance: only PyTorch scales with `num_workers` (separate processes).
TurboLoader's fast path is a single process-wide C++ thread pool already saturated
at one worker; `tf.data` uses AUTOTUNE.

See [Benchmark Results](benchmarks/index.md) for methodology.

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

**Advanced Transforms:**
- RandomPosterize
- RandomSolarize
- RandomPerspective
- AutoAugment (ImageNet, CIFAR10, SVHN policies)

**Tensor Conversion:**
- ToTensor (PyTorch CHW, TensorFlow HWC)

See [Transforms API](api/transforms.md) for complete reference.

## Installation

```bash
pip install turboloader
```

Prebuilt manylinux wheels are published for Linux x86_64 and aarch64 (plus an
sdist); portable macOS wheels built from source are being added. PyTorch is an
**optional** dependency - install it alongside TurboLoader with:

```bash
pip install turboloader[torch]
```

**Requirements:**
- Python 3.10+
- C++20 compiler (only needed when building from the sdist/source)
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

- [Overview](benchmarks/index.md) - Measured performance results
- [Benchmark Setup](benchmark_setup.md) - How to reproduce the benchmarks

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

- **v2.26.2** (Current; v2.26.1 is the latest published on PyPI) - FFCV-style direct-batch loader, multi-modality (`TokenDataLoader`, `ArrayDataLoader`), decoded cache (`cache_decoded=True`), DDP-safe distributed sharding
- **v2.7.0** - Decoded Tensor Caching (`cache_decoded=True`), FastDataLoader, MemoryEfficientDataLoader
- **v2.4.0** - Integrated transform pipeline support in DataLoader
- **v2.0.0** - Tiered Caching (L1 memory + L2 disk), Smart Batching enabled by default, Pipeline tuning optimizations
- **v1.9.0** - Transform Pipe Operator, COCO/VOC annotations, source-only/optional storage backends (HDF5/TFRecord/Zarr/cloud), io_uring
- **v1.8.0** - Modern Augmentations (MixUp, CutMix, Mosaic, RandAugment, GridMask), Logging system
- **v1.7.7** - Developer experience improvements: Issue templates, Quick Start notebook, PyTorch Lightning example
- **v1.5.0** - TBL v2 Format with LZ4 compression (40-60% space savings)
- **v1.2.0** - Smart Batching + Distributed Training
- **v1.1.0** - AVX-512 SIMD + Binary Format Improvements + Prefetching
- **v1.0.0** - Production/Stable Release

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
  version = {2.26.2},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```
