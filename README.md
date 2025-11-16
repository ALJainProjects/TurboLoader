# TurboLoader

**High-Performance ML Data Loading Library**

[![PyPI version](https://badge.fury.io/py/turboloader.svg)](https://badge.fury.io/py/turboloader)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TurboLoader is a high-performance data loading library designed to accelerate ML training by replacing Python's multiprocessing-based data loaders with efficient C++ native threads and lock-free data structures.

**Key Features**:
- 🚀 **Native C++ Implementation** with Python bindings via pybind11
- ⚡ **SIMD-Optimized Transforms** using AVX2/AVX-512/NEON
- 🔒 **Lock-Free Concurrent Queues** for zero-contention data passing
- 🧵 **C++ Native Threads** (no Python GIL, no multiprocessing overhead)
- 💾 **Zero-Copy Memory-Mapped I/O** for efficient file reading
- 📦 **WebDataset TAR Format** support for sharded datasets
- 🎯 **SIMD-Accelerated Image Decoders** (JPEG, PNG, WebP)
- 🎨 **7 Data Augmentation Transforms** with SIMD optimization
- 🐍 **PyTorch-Compatible API** drop-in replacement

---

## Performance

TurboLoader achieves high performance through several optimizations:

- **Lock-free queues** eliminate synchronization overhead between threads
- **SIMD-optimized transforms** (AVX2/NEON) accelerate image preprocessing
- **Native C++ threads** avoid Python GIL and multiprocessing overhead
- **Memory-mapped I/O** enables zero-copy file reading
- **Thread-local decoders** eliminate allocation overhead

### Benchmark Results

**SIMD Transform Performance** (Apple M1 Pro, NEON backend):

| Operation | Throughput | Time per Image |
|-----------|------------|----------------|
| Resize (256x256→224x224) | 6,718 img/s | 148.85 μs |
| Normalize (RGB) | 47,438 img/s | 21.08 μs |

**Test Configuration**:
- Hardware: Apple M1 Pro (8 cores, 16GB RAM)
- Input: 256x256 RGB images
- Backend: NEON SIMD instructions
- All tests run on synthetic datasets

> **Note**: These are micro-benchmarks of individual SIMD operations. End-to-end data loading performance depends on dataset size, hardware, I/O bandwidth, and pipeline configuration. We recommend benchmarking on your specific use case.

---

## Installation

```bash
pip install turboloader
```

**Requirements**:
- Python 3.8+
- C++20 compiler (GCC 10+, Clang 12+, MSVC 19.29+)
- CMake 3.15+

**Optional Dependencies**:
- libjpeg-turbo (JPEG decoding)
- libpng (PNG decoding)
- libwebp (WebP decoding)

---

## Quick Start

### Basic Usage

```python
import turboloader

# Create pipeline
pipeline = turboloader.Pipeline(
    tar_paths=['imagenet.tar'],
    num_workers=8,
    batch_size=32,
    decode_jpeg=True
)

pipeline.start()

# Get batches
for _ in range(100):
    batch = pipeline.next_batch(32)
    for sample in batch:
        img = sample.get_image()  # NumPy array (H, W, C)
        # Your training code here...

pipeline.stop()
```

### With SIMD Transforms

```python
import turboloader

# Configure SIMD-accelerated transforms
config = turboloader.TransformConfig()
config.enable_resize = True
config.resize_width = 224
config.resize_height = 224
config.enable_normalize = True
config.mean = [0.485, 0.456, 0.406]
config.std = [0.229, 0.224, 0.225]

pipeline = turboloader.Pipeline(
    tar_paths=['imagenet.tar'],
    num_workers=8,
    decode_jpeg=True,
    enable_simd_transforms=True,
    transform_config=config
)

pipeline.start()
batch = pipeline.next_batch(256)
pipeline.stop()
```

### With Data Augmentation

```python
import turboloader

# Create augmentation pipeline
aug_pipeline = turboloader.AugmentationPipeline()
aug_pipeline.add_transform(turboloader.RandomHorizontalFlip(0.5))
aug_pipeline.add_transform(turboloader.ColorJitter(brightness=0.2, contrast=0.2))
aug_pipeline.add_transform(turboloader.RandomCrop(224, 224))

# Use with data loader (planned feature)
# pipeline = turboloader.Pipeline(tar_paths=['data.tar'], augmentations=aug_pipeline)
```

---

## Architecture

TurboLoader is built on several high-performance components:

### Core Components

1. **Lock-Free Queues**
   - SPSC (Single Producer Single Consumer) queues
   - Atomic operations for thread-safe data passing
   - Zero-copy transfer of decoded images

2. **Memory-Mapped I/O**
   - `mmap()` for zero-copy file reading
   - Efficient TAR archive parsing
   - Minimizes memory allocations

3. **SIMD Transforms**
   - AVX2/AVX-512 on x86_64
   - NEON on ARM (Apple Silicon, ARM servers)
   - Vectorized resize, normalize, color conversion

4. **Thread-Local Decoders**
   - Per-thread JPEG/PNG/WebP decoders
   - Eliminates decoder allocation overhead
   - Maximizes cache locality

### Supported Transforms

TurboLoader v0.3.x includes 7 SIMD-accelerated augmentation transforms:

- **RandomHorizontalFlip**: SIMD-optimized horizontal flip
- **RandomVerticalFlip**: SIMD-optimized vertical flip
- **ColorJitter**: Brightness, contrast, saturation adjustments
- **RandomRotation**: Bilinear interpolation rotation
- **RandomCrop**: Random crop with padding
- **RandomErasing**: Cutout augmentation
- **GaussianBlur**: Separable Gaussian filter (SIMD)

---

## API Reference

### Pipeline

```python
class Pipeline:
    def __init__(
        self,
        tar_paths: List[str],
        num_workers: int = 4,
        queue_size: int = 256,
        shuffle: bool = False,
        decode_jpeg: bool = False,
        enable_simd_transforms: bool = False,
        transform_config: Optional[TransformConfig] = None
    )

    def start() -> None
    def stop() -> None
    def reset() -> None
    def next_batch(batch_size: int) -> List[Sample]
    def total_samples() -> int
```

### TransformConfig

```python
class TransformConfig:
    enable_resize: bool = False
    resize_width: int = 224
    resize_height: int = 224
    resize_method: ResizeMethod = ResizeMethod.BILINEAR

    enable_normalize: bool = False
    mean: List[float] = [0.0, 0.0, 0.0]
    std: List[float] = [1.0, 1.0, 1.0]

    enable_color_convert: bool = False
    src_color: ColorSpace = ColorSpace.RGB
    dst_color: ColorSpace = ColorSpace.RGB
    output_float: bool = False
```

### Augmentation Transforms

```python
class AugmentationPipeline:
    def __init__(seed: Optional[int] = None)
    def add_transform(transform: AugmentationTransform) -> None
    def clear() -> None
    def num_transforms() -> int

class RandomHorizontalFlip(AugmentationTransform):
    def __init__(probability: float = 0.5)

class ColorJitter(AugmentationTransform):
    def __init__(
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0
    )
```

---

## Roadmap

### v0.4.0 (Q2 2025)
- [ ] Full ImageNet benchmark suite
- [ ] TensorFlow/JAX integration
- [ ] Additional image formats (TIFF, BMP)
- [ ] Video decoding support

### v0.5.0 (Q3 2025)
- [ ] GPU-accelerated JPEG decoding (nvJPEG)
- [ ] Distributed training support
- [ ] S3/GCS remote dataset loading

### v1.0.0 (Q4 2025)
- [ ] Production-ready API stability
- [ ] Comprehensive documentation
- [ ] Full test coverage
- [ ] Performance optimization

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

# Install dependencies
brew install cmake libjpeg-turbo libpng libwebp  # macOS
# or
apt-get install cmake libjpeg-turbo8-dev libpng-dev libwebp-dev  # Ubuntu

# Build from source
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8

# Run tests
./tests/turboloader_tests
./tests/test_simd_transforms
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Citation

If you use TurboLoader in your research, please cite:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance ML Data Loading},
  year = {2025},
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

- **Issues**: [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Documentation**: [docs/](docs/)
- **PyPI**: [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)
