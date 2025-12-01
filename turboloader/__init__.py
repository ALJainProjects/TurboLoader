"""TurboLoader: High-performance data loading for machine learning.

v2.3.18 - Fix macOS cibuildwheel builds with sysconfig patching

New in v2.0.0:
- Tiered Caching: L1 memory (LRU) + L2 disk cache for 5-10x faster subsequent epochs
- Smart Batching enabled by default: 1.2x throughput, 15-25% memory savings
- Pipeline tuning: Increased prefetch (4 batches), larger buffer pool (256)
- xxHash64 content hashing for fast cache key generation
- Cache-aside pattern: L1 → L2 → decode on miss
- Async disk writes via background thread
- DataLoader parameters: enable_cache, cache_l1_mb, cache_l2_gb, cache_dir

Previous features (v1.9.0):
- Transform Pipe Operator: pipeline = Resize(224) | Normalize() | ToTensor()
- HDF5/TFRecord/Zarr format support
- COCO/Pascal VOC annotation format support
- Azure Blob Storage, GPU transforms, io_uring

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
- 24 SIMD-accelerated data augmentation transforms (AVX2/NEON)
- Advanced transforms: RandomPerspective, RandomPosterize, RandomSolarize, AutoAugment, Lanczos interpolation
- AutoAugment learned policies: ImageNet, CIFAR10, SVHN
- Interactive benchmark web app with real-time visualizations
- WebDataset format support for multi-modal datasets
- Remote TAR support (HTTP, S3, GCS, Azure)
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

__version__ = "2.3.18"

# Import C++ extension module
try:
    from _turboloader import (
        # Core DataLoader
        DataLoader,
        version,
        features,
        # TBL v2 Format
        TblReaderV2,
        TblWriterV2,
        SampleFormat,
        MetadataType,
        # Smart Batching
        SmartBatchConfig,
        # Transform Composition
        Compose,
        ComposedTransforms,
        # Transforms (all SIMD-accelerated transforms)
        Resize,
        CenterCrop,
        RandomCrop,
        RandomHorizontalFlip,
        RandomVerticalFlip,
        ColorJitter,
        GaussianBlur,
        Grayscale,
        Normalize,
        ImageNetNormalize,
        ToTensor,
        Pad,
        RandomRotation,
        RandomAffine,
        RandomPerspective,
        RandomPosterize,
        RandomSolarize,
        RandomErasing,
        AutoAugment,
        AutoAugmentPolicy,
        # Modern Augmentations (v1.8.0)
        MixUp,
        CutMix,
        Mosaic,
        RandAugment,
        GridMask,
        # Logging (v1.8.0)
        LogLevel,
        enable_logging,
        disable_logging,
        set_log_level,
        set_log_output,
        # Enums
        InterpolationMode,
        PaddingMode,
        TensorFormat,
    )

    __all__ = [
        "DataLoader",
        "version",
        "features",
        "__version__",
        # TBL v2
        "TblReaderV2",
        "TblWriterV2",
        "SampleFormat",
        "MetadataType",
        # Smart Batching
        "SmartBatchConfig",
        # Transform Composition
        "Compose",
        "ComposedTransforms",
        # Transforms
        "Resize",
        "CenterCrop",
        "RandomCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "ColorJitter",
        "GaussianBlur",
        "Grayscale",
        "Normalize",
        "ImageNetNormalize",
        "ToTensor",
        "Pad",
        "RandomRotation",
        "RandomAffine",
        "RandomPerspective",
        "RandomPosterize",
        "RandomSolarize",
        "RandomErasing",
        "AutoAugment",
        "AutoAugmentPolicy",
        # Modern Augmentations (v1.8.0)
        "MixUp",
        "CutMix",
        "Mosaic",
        "RandAugment",
        "GridMask",
        # Logging (v1.8.0)
        "LogLevel",
        "enable_logging",
        "disable_logging",
        "set_log_level",
        "set_log_output",
        # Enums
        "InterpolationMode",
        "PaddingMode",
        "TensorFormat",
    ]
except ImportError:
    # Fallback for development/documentation builds
    __all__ = ["__version__"]
