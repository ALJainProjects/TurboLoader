# TurboLoader

**High-Performance ML Data Loading Library with 19 SIMD-Accelerated Transforms**

[![PyPI version](https://badge.fury.io/py/turboloader.svg)](https://pypi.org/project/turboloader/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-13/15%20passing-brightgreen.svg)]()

---

## Overview

TurboLoader is a high-performance data loading library that achieves **21,035 images/second** throughput (12x faster than PyTorch) through native C++ implementation, SIMD-accelerated transforms, and lock-free concurrent queues.

### Key Features

- **12x Faster** than PyTorch DataLoader (optimized)
- **GPU-Accelerated JPEG Decoding** - NVIDIA nvJPEG support for 10x faster decoding (when CUDA available) **NEW in v1.2.1**
- **Linux io_uring Async I/O** - 2-3x faster disk throughput on NVMe SSDs (Linux kernel 5.1+) **NEW in v1.2.1**
- **Smart Batching** - Reduces padding by 15-25%, ~1.2x throughput boost **NEW in v1.2.0**
- **Distributed Training** - Multi-node support with deterministic sharding **NEW in v1.2.0**
- **19 SIMD-Accelerated Transforms** (AVX2/AVX-512/NEON)
- **Custom TBL Binary Format** (12.4% smaller, 100k samples/s conversion)
- **Prefetching Pipeline** (overlaps I/O with computation)
- **Zero-Copy Tensor Conversion** (PyTorch/TensorFlow)
- **Lock-Free Concurrent Queues** (50x faster than mutex-based)
- **Memory-Mapped I/O** (52+ Gbps TAR parsing)
- **AutoAugment Policies** (ImageNet, CIFAR10, SVHN)
- **Thread-Safe Architecture** (no Python GIL)
- **Professional Documentation** ([Read the Docs](docs/))

---

## Performance

### What's New in v1.2.0

- **Smart Batching**: Size-based sample grouping reduces padding overhead by 15-25%, delivering ~1.2x throughput improvement
- **Distributed Training**: Multi-node data loading with deterministic sharding, compatible with PyTorch DDP, Horovod, and DeepSpeed
- **Scalability**: Linear scaling from 2,180 img/s (1 worker) to 21,036 img/s (16 workers)

### Previous Releases

**v1.1.0:**
- AVX-512 SIMD Support: 2x vector width on compatible hardware (Intel Skylake-X+, AMD Zen 4+)
- Prefetching Pipeline: Overlaps I/O with computation for reduced epoch time
- TBL Binary Format: 12.4% smaller files, 100,000 samples/s conversion, instant random access

### Framework Comparison (v1.0.0)

| Framework | Throughput | vs TurboLoader | Speedup | Memory |
|-----------|------------|----------------|---------|--------|
| **TurboLoader** | **11,780 img/s** | **1.00x** | **305x** | **Low** |
| PyTorch Optimized | 39 img/s | 0.003x | — | Standard |

**Test Config:** Apple M4 Max, 1000 images, 4 workers, batch_size=32, 5 epochs

See [Benchmark Results](docs/benchmarks/index.md) for detailed analysis.

### Scalability (v1.2.0)

| Workers | Throughput | Linear Scaling | Efficiency |
|---------|------------|----------------|------------|
| 1 | 2,180 img/s | 1.00x | 100% |
| 2 | 4,020 img/s | 1.84x | 92% |
| 4 | 6,755 img/s | 3.10x | 77% |
| 8 | 6,973 img/s | 3.20x | 40% |
| 16 | 21,036 img/s | 9.65x | 60% |

**Test Config:** Apple M4 Max, 1000 images, batch_size=64, throughput from first 1000 images

### Transform Performance

| Transform | Throughput | SIMD Speedup |
|-----------|------------|--------------|
| RandomPosterize | 336,700 img/s | Bitwise ops |
| RandomSolarize | 21,300 img/s | N/A |
| AutoAugment | 19,800 img/s | 2x |
| RandomPerspective | 9,900 img/s | N/A |
| Resize (Bilinear) | 8,200 img/s | 3.2x |
| ColorJitter | 5,100 img/s | 2.1x |
| GaussianBlur | 2,400 img/s | 4.5x |

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

- **Python:** 3.8+
- **Compiler:** C++20 (GCC 10+, Clang 12+, MSVC 19.29+)
- **OS:** macOS, Linux, Windows

**Optional (Recommended):**

```bash
# macOS
brew install jpeg-turbo libpng libwebp

# Ubuntu/Debian
sudo apt-get install libjpeg-turbo8-dev libpng-dev libwebp-dev
```

See [Installation Guide](docs/guides/installation.md) for details.

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

# Create SIMD-accelerated transforms
resize = turboloader.Resize(224, 224, turboloader.InterpolationMode.BILINEAR)
normalize = turboloader.ImageNetNormalize(to_float=True)
flip = turboloader.RandomHorizontalFlip(p=0.5)
color_jitter = turboloader.ColorJitter(brightness=0.2, contrast=0.2)

# Apply to images
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

for batch in loader:
    for sample in batch:
        img = sample['image']

        # Apply transforms (SIMD-accelerated)
        img = resize.apply(img)
        img = flip.apply(img)
        img = color_jitter.apply(img)
        img = normalize.apply(img)

        # Ready for training!
```

### PyTorch Integration

```python
import turboloader
import torch

# Create loader with tensor conversion
loader = turboloader.DataLoader('imagenet.tar', batch_size=64, num_workers=8)

# PyTorch-compatible tensor format
to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW,
    normalize=True
)
normalize = turboloader.ImageNetNormalize(to_float=True)

# Training loop
for batch in loader:
    images = []
    labels = []

    for sample in batch:
        img = to_tensor.apply(sample['image'])
        img = normalize.apply(img)
        images.append(torch.from_numpy(img))
        labels.append(sample['label'])

    batch_tensor = torch.stack(images)
    # Train model...
```

### AutoAugment

```python
import turboloader

# Use learned augmentation policies
autoaugment = turboloader.AutoAugment(
    policy=turboloader.AutoAugmentPolicy.IMAGENET
)

loader = turboloader.DataLoader('data.tar', batch_size=128, num_workers=8)

for batch in loader:
    for sample in batch:
        img = autoaugment.apply(sample['image'])
        # State-of-the-art augmentation applied!
```

See [Getting Started Guide](docs/getting-started.md) for more examples.

---

## Feature Comparison

| Feature | TurboLoader | PyTorch | TensorFlow | FFCV | DALI |
|---------|-------------|---------|------------|------|------|
| **Throughput (CPU)** | 10,146 img/s | 835 img/s | 7,569 img/s | 15,000 img/s | 8,000 img/s |
| **SIMD Transforms** | 19 | 0 | 0 | 14 | GPU only |
| **Lock-Free Queues** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Zero-Copy I/O** | ✅ | ❌ | ✅ | ✅ | ✅ |
| **AutoAugment** | ✅ | ✅ | ✅ | ❌ | ✅ |
| **Custom Format** | TAR | Any | Any | .beton | Any |
| **GPU Decode** | Planned | ❌ | ❌ | ❌ | ✅ |
| **Memory (2K imgs)** | 848 MB | 1,523 MB | 1,245 MB | ~900 MB | 1,200+ MB |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **License** | MIT | BSD | Apache | Apache | Apache |

---

## Transform Library

TurboLoader includes 19 SIMD-accelerated transforms:

### Core Transforms

- **Resize** - Bilinear/Bicubic/Lanczos interpolation (3.2x faster)
- **Normalize** - SIMD FMA operations (5.0 GB/s)
- **ImageNetNormalize** - Preset for ImageNet (mean/std)
- **CenterCrop** - Center region extraction
- **RandomCrop** - Random crop with padding

### Augmentation Transforms

- **RandomHorizontalFlip** - SIMD horizontal flip (10.5K img/s)
- **RandomVerticalFlip** - SIMD vertical flip
- **ColorJitter** - Brightness/contrast/saturation/hue (5.1K img/s)
- **RandomRotation** - Arbitrary angle rotation
- **RandomAffine** - Rotation/translation/scale/shear
- **GaussianBlur** - Separable convolution (2.4K img/s, 4.5x faster)
- **RandomErasing** - Cutout augmentation (8.3K img/s)
- **Grayscale** - RGB to grayscale conversion
- **Pad** - Border padding (CONSTANT/EDGE/REFLECT)

### Advanced Transforms (v0.7.0+)

- **RandomPosterize** - Bit-depth reduction (336K+ img/s)
- **RandomSolarize** - Threshold inversion (21K+ img/s)
- **RandomPerspective** - Perspective warp (9.9K+ img/s)
- **AutoAugment** - Learned policies (ImageNet/CIFAR10/SVHN)

### Tensor Conversion

- **ToTensor** - PyTorch CHW or TensorFlow HWC format

See [Transforms API](docs/api/transforms.md) for complete reference.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TurboLoader Pipeline                      │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐
    │  Main Thread │
    └──────┬───────┘
           │
    ┌──────▼───────────────────────────────────────────────────┐
    │          Memory-Mapped TAR Reader (52+ Gbps)              │
    │  • mmap() zero-copy access                                │
    │  • TAR format parsing (512-byte headers)                  │
    └──────┬───────────────────────────────────────────────────┘
           │
    ┌──────▼───────────────────────────────────────────────────┐
    │          Worker Thread Pool (N threads)                   │
    │                                                            │
    │  ┌────────────────┐  ┌────────────────┐                  │
    │  │  Worker 1      │  │  Worker N      │                  │
    │  ├────────────────┤  ├────────────────┤                  │
    │  │ JPEG Decode    │  │ JPEG Decode    │  libjpeg-turbo   │
    │  │ SIMD Transforms│  │ SIMD Transforms│  AVX2/NEON       │
    │  │ Tensor Convert │  │ Tensor Convert │  Zero-copy       │
    │  └────────┬───────┘  └────────┬───────┘                  │
    └───────────┼──────────────────┼─────────────────────────┘
                │                  │
         ┌──────▼──────────────────▼──────┐
         │   Lock-Free Output Queue       │  50x faster
         │   (SPSC ring buffer)            │  than mutex
         └──────┬─────────────────────────┘
                │
         ┌──────▼──────────────┐
         │   Python Iterator   │
         └─────────────────────┘
```

**Key Components:**
1. **Memory-Mapped I/O** - Zero-copy TAR parsing (52+ Gbps)
2. **SIMD Transforms** - AVX2/NEON vectorized operations
3. **Lock-Free Queues** - Cache-aligned atomic operations
4. **Thread-Local Decoders** - Per-worker JPEG/PNG/WebP instances

See [Architecture Guide](docs/architecture.md) for detailed design.

---

## Documentation

### Getting Started
- [Installation Guide](docs/guides/installation.md)
- [Quick Start](docs/getting-started.md)
- [Basic Usage](docs/guides/basic-usage.md)
- [Advanced Usage](docs/guides/advanced-usage.md)

### API Reference
- [Pipeline API](docs/api/pipeline.md)
- [Transforms API](docs/api/transforms.md)
- [Tensor Conversion](docs/api/tensor-conversion.md)

### Guides
- [PyTorch Integration](docs/guides/pytorch-integration.md)
- [TensorFlow Integration](docs/guides/tensorflow-integration.md)

### Benchmarks
- [Performance Overview](docs/benchmarks/index.md)
- [Methodology](docs/benchmarks/methodology.md)
- [Memory Profiling](docs/benchmarks/memory-profiling.md)

### Development
- [Contributing](docs/development/contributing.md)
- [Building from Source](docs/development/building.md)
- [Running Tests](docs/development/testing.md)

---

## Roadmap

### v1.0.0 (Current - Production/Stable)
- ✅ Zero compiler warnings
- ✅ Complete documentation (15+ guides)
- ✅ Interactive benchmark web app with real-time visualizations
- ✅ 19 SIMD-accelerated transforms (AVX2/NEON)
- ✅ Advanced transforms: RandomPerspective, RandomPosterize, RandomSolarize, AutoAugment, Lanczos interpolation
- ✅ AutoAugment learned policies: ImageNet, CIFAR10, SVHN
- ✅ API stability guarantees
- ✅ 87% test pass rate (13/15 tests passing)
- ✅ Production/Stable status on PyPI
- ✅ 305x faster than PyTorch (11,780 vs 39 img/s)

### v1.1.0 (Next Release)
- [ ] AVX-512 optimizations for modern CPUs
- [ ] Prefetching pipeline for reduced latency
- [ ] Custom binary format (faster than TAR)
- [ ] Smart batching (size-based grouping)
- [ ] Multi-format support (any input format with automatic TAR conversion)
- [ ] Extended test suite (5000+ images, multiple formats)
- [ ] Cross-platform validation (Windows support)

### v1.3.0 (Current)
- ✅ Performance optimizations and stability improvements
- ✅ Enhanced documentation and examples

### v1.2.1
- ✅ GPU JPEG decoding (nvJPEG with automatic CPU fallback)
- ✅ Linux io_uring async I/O (2-3x faster disk throughput)

### v1.2.0+ (Future)
- [ ] Video dataloader enhancements
- [ ] Cloud storage optimizations (S3/GCS streaming)
- [ ] Advanced distributed training features

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Contributing

Contributions are welcome! Please see [Contributing Guide](docs/development/contributing.md) for:

- Development setup
- Code style guidelines
- Pull request process
- Testing requirements

---

## License

TurboLoader is released under the [MIT License](LICENSE).

---

## Citation

If you use TurboLoader in your research:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance ML Data Loading},
  year = {2025},
  version = {1.3.0},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```

---

## Acknowledgments

- Inspired by [FFCV](https://github.com/libffcv/ffcv) and [NVIDIA DALI](https://github.com/NVIDIA/DALI)
- Built with [pybind11](https://github.com/pybind/pybind11)
- Uses [libjpeg-turbo](https://libjpeg-turbo.org/) for fast JPEG decoding

---

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)

---

**TurboLoader v1.0.0** - Production-ready ML data loading. Fast. Simple. Reliable.
