"""TurboLoader: High-performance data loading for machine learning.

v2.4.0 - Add transform parameter to DataLoader for integrated pipeline transforms

New in v2.4.0:
- DataLoader now accepts a `transform` parameter for integrated transforms
- Transforms are applied after decoding using SIMD-accelerated C++ code
- Example: DataLoader('data.tar', transform=Resize(224, 224) | ImageNetNormalize())

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

__version__ = "2.4.0"

# Import C++ extension module
try:
    from _turboloader import (
        # Core DataLoader (internal - we wrap this)
        DataLoader as _DataLoaderBase,
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

    # Create DataLoader wrapper with transform support
    class DataLoader:
        """High-performance DataLoader with integrated transform support.

        Drop-in replacement for PyTorch DataLoader with TurboLoader performance
        and SIMD-accelerated transforms.

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet).
                            Supports: local files, http://, https://, s3://, gs://
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads (default: 4)
            shuffle (bool): Shuffle samples (future feature, default: False)
            transform: Transform or composed transforms to apply to images.
                      Use pipe operator: Resize(224, 224) | ImageNetNormalize()
                      Or Compose([Resize(224, 224), ImageNetNormalize()])
            enable_distributed (bool): Enable distributed training (default: False)
            world_rank (int): Rank of this process (default: 0)
            world_size (int): Total number of processes (default: 1)
            drop_last (bool): Drop incomplete batches (default: False)
            distributed_seed (int): Seed for shuffling (default: 42)
            enable_cache (bool): Enable tiered caching (default: False)
            cache_l1_mb (int): L1 memory cache size in MB (default: 512)
            cache_l2_gb (int): L2 disk cache size in GB (default: 0)
            cache_dir (str): L2 cache directory (default: /tmp/turboloader_cache)
            auto_smart_batching (bool): Auto-detect smart batching (default: True)
            enable_smart_batching (bool): Manual smart batching override (default: False)
            prefetch_batches (int): Batches to prefetch (default: 4)

        Example:
            >>> # With transforms
            >>> loader = turboloader.DataLoader(
            ...     'imagenet.tar',
            ...     batch_size=128,
            ...     num_workers=8,
            ...     transform=turboloader.Resize(224, 224) | turboloader.ImageNetNormalize()
            ... )
            >>> for batch in loader:
            ...     images = [sample['image'] for sample in batch]
        """

        def __init__(
            self,
            data_path,
            batch_size=32,
            num_workers=4,
            shuffle=False,
            transform=None,
            enable_distributed=False,
            world_rank=0,
            world_size=1,
            drop_last=False,
            distributed_seed=42,
            enable_cache=False,
            cache_l1_mb=512,
            cache_l2_gb=0,
            cache_dir="/tmp/turboloader_cache",
            auto_smart_batching=True,
            enable_smart_batching=False,
            prefetch_batches=4,
        ):
            self._transform = transform
            self._loader = _DataLoaderBase(
                data_path,
                batch_size,
                num_workers,
                shuffle,
                enable_distributed,
                world_rank,
                world_size,
                drop_last,
                distributed_seed,
                enable_cache,
                cache_l1_mb,
                cache_l2_gb,
                cache_dir,
                auto_smart_batching,
                enable_smart_batching,
                prefetch_batches,
            )

        def _apply_transform(self, sample):
            """Apply transform to a sample's image if transform is set."""
            if self._transform is not None and "image" in sample:
                img = sample["image"]
                if img is not None:
                    # Apply the SIMD-accelerated C++ transform
                    sample["image"] = self._transform.apply(img)
            return sample

        def next_batch(self):
            """Get next batch with transforms applied."""
            batch = self._loader.next_batch()
            if self._transform is not None:
                batch = [self._apply_transform(s) for s in batch]
            return batch

        def is_finished(self):
            """Check if all data has been processed."""
            return self._loader.is_finished()

        def smart_batching_enabled(self):
            """Check if smart batching is active."""
            return self._loader.smart_batching_enabled()

        def stop(self):
            """Stop the pipeline and clean up resources."""
            self._loader.stop()

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            """Context manager exit."""
            self.stop()

        def __iter__(self):
            """Make DataLoader iterable."""
            return self

        def __next__(self):
            """Get next batch (iterator protocol) with transforms applied."""
            batch = self._loader.__next__()
            if self._transform is not None:
                batch = [self._apply_transform(s) for s in batch]
            return batch

        @property
        def transform(self):
            """Get the current transform."""
            return self._transform

        @transform.setter
        def transform(self, value):
            """Set the transform."""
            self._transform = value

except ImportError:
    # Fallback for development/documentation builds
    __all__ = ["__version__"]
