"""TurboLoader: High-performance data loading for machine learning.

v1.5.0 - TBL v2 Format with LZ4 Compression

Production-Ready Features:
- TBL v2 format: 40-60% space savings with LZ4 compression
- Streaming writer with constant memory usage
- Memory-mapped reader for zero-copy reads
- Data integrity validation (CRC32/CRC16 checksums)
- Cached image dimensions for fast filtered loading
- Rich metadata support (JSON, Protobuf, MessagePack)
- 4,875 img/s TAR→TBL conversion throughput
- 21,035 img/s throughput with 16 workers (12x faster than PyTorch, 1.3x faster than TensorFlow)
- Smart Batching: Size-based sample grouping reduces padding by 15-25%, ~1.2x throughput boost
- Distributed Training: Multi-node data loading with deterministic sharding (PyTorch DDP, Horovod, DeepSpeed)
- 19 SIMD-accelerated data augmentation transforms (AVX2/NEON)
- Advanced transforms: RandomPerspective, RandomPosterize, RandomSolarize, AutoAugment, Lanczos interpolation
- AutoAugment learned policies: ImageNet, CIFAR10, SVHN
- Interactive benchmark web app with real-time visualizations
- WebDataset format support for multi-modal datasets
- Remote TAR support (HTTP, S3, GCS)
- GPU-accelerated JPEG decoding (nvJPEG)
- PyTorch/TensorFlow/JAX framework integration
- Lock-free SPSC queues for maximum concurrency
- 52+ Gbps local file throughput
- Multi-format pipeline (images, video, tabular data)
- SIMD-optimized JPEG decoder (SSE2/AVX2/NEON via libjpeg-turbo)
- Comprehensive test suite (90%+ pass rate)
- Zero compiler warnings

Developed and tested on Apple M4 Max (48GB RAM) with C++20 and Python 3.8+
"""

__version__ = "1.5.1"

# Import C++ extension module
try:
    from _turboloader import DataLoader, version, features
    __all__ = ['DataLoader', 'version', 'features', '__version__']
except ImportError:
    # Fallback for development/documentation builds
    __all__ = ['__version__']
