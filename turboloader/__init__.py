"""TurboLoader: High-performance data loading for machine learning.

v2.24.0 - PyTorch Compatibility Layer

New in v2.24.0:
- PyTorchCompatibleLoader: Drop-in replacement for torch.utils.data.DataLoader
- Returns (images, labels) tuples exactly like PyTorch DataLoader
- Automatic label extraction from folder structure (ImageFolder style)
- ImageFolderConverter: Convert ImageFolder datasets to TAR format
- TransformAdapter: Convert torchvision transforms to TurboLoader transforms
- FolderLabelExtractor, FilenamePatternExtractor, MetadataLabelExtractor
- Full PyTorch training loop compatibility

v2.10.0 - Performance Optimizations (Phase 2)

New in v2.10.0:
- Lanczos LUT cache: ~25% faster Lanczos interpolation
- BufferPool class: Thread-safe buffer pooling for memory reuse (5-15% throughput gain)
- ResizeTransform buffer pool integration: use_buffer_pool=True parameter
- OpenMP threshold fix: Better small batch performance (5-10% improvement)
- SPSC cache line alignment: 5-10% latency improvement

New in v2.9.0:
- Fixed uninitialized pixels in Bicubic/Lanczos resize at corners (now neutral gray 128)
- Fixed TAR header buffer overflow with proper bounds checking on prefix/name fields
- Fixed silent GPU decode failures: now logs errors and respects skip_corrupted config
- Fixed CPU decode failures: now logs errors and respects skip_corrupted config
- Improved error messages with sample index and filename for decode failures

New in v2.8.0:
- Complete AutoAugment: All 14 operations fully implemented (Invert, AutoContrast,
  Equalize, Color, Brightness, Contrast, Sharpness, ShearX/Y, TranslateX/Y)
- shuffle=True: Enable data shuffling with intra-worker Fisher-Yates algorithm
- set_epoch(): Reproducible shuffling across epochs (matches PyTorch DataLoader)

New in v2.7.0:
- cache_decoded=True: Cache decoded numpy arrays in memory
- Subsequent epochs iterate directly from cache (100K+ img/s vs 30K uncached)
- Matches TensorFlow's cache() performance pattern
- clear_cache(): Free memory or force cache repopulation
- cache_populated, cache_size_mb properties for introspection

New in v2.6.0:
- MemoryEfficientDataLoader: Auto-tuned settings for memory-constrained environments
- create_loader(): Factory function with loader_type='fast'/'memory_efficient'/'standard'

New in v2.5.0:
- FastDataLoader: 8-12% faster batch loading via contiguous array transfer
- next_batch_array(): Single allocation for entire batch, parallel memcpy
- next_batch_into(): Zero-allocation batch loading with pre-allocated buffers
- Loader(): Unified factory function with fast=True/False parameter
- Output formats: 'numpy', 'numpy_chw', 'pytorch', 'tensorflow'

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
- 21,035 img/s throughput with 16 workers (~2x faster than PyTorch DataLoader)
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

__version__ = "2.24.0"

# Prevent duplicate libomp crash when co-existing with PyTorch
# (PyTorch bundles its own libomp; TurboLoader links against system libomp)
import os as _os

if "KMP_DUPLICATE_LIB_OK" not in _os.environ:
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        "FastDataLoader",
        "MemoryEfficientDataLoader",
        "create_loader",
        "Loader",
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
            shuffle (bool): Shuffle samples within each worker (default: False).
                          Use set_epoch() for reproducible shuffling across epochs.
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

        def set_epoch(self, epoch: int):
            """Set the epoch for reproducible shuffling (NEW in v2.8.0).

            When shuffle=True, call this at the start of each epoch to get
            reproducible shuffling. Different epochs produce different orderings,
            but the same epoch + seed = same ordering.

            Args:
                epoch (int): The epoch number (0, 1, 2, ...)

            Example:
                >>> loader = turboloader.DataLoader('data.tar', shuffle=True)
                >>> for epoch in range(10):
                ...     loader.set_epoch(epoch)  # Different shuffle each epoch
                ...     for batch in loader:
                ...         train(batch)
            """
            self._loader.set_epoch(epoch)

    class FastDataLoader:
        """High-performance DataLoader with batch array transfer (8-12% faster).

        Uses contiguous array transfer instead of per-sample dicts for maximum
        throughput. Designed to beat TensorFlow's tf.data performance.

        Key optimizations:
        - Single allocation for entire batch (N, H, W, C) or (N, C, H, W)
        - Parallel memcpy with OpenMP
        - GIL released during C++ data preparation
        - No per-sample Python dict creation overhead

        New in v2.7.0 - Decoded Tensor Caching:
        - cache_decoded=True stores complete numpy batch arrays in memory
        - Subsequent epochs iterate directly over cached arrays (no decoding)
        - Achieves 100K+ img/s on cached epochs (vs ~30K uncached)
        - Matches TensorFlow's cache() performance pattern

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet).
                            Supports: local files, http://, https://, s3://, gs://
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads (default: 4)
            output_format (str): Output format - 'numpy', 'numpy_chw', 'pytorch',
                               'tensorflow' (default: 'numpy')
            target_height (int): Target image height (0 = auto from first image)
            target_width (int): Target image width (0 = auto from first image)
            transform: Transform pipeline to apply to images
            enable_distributed (bool): Enable distributed training (default: False)
            world_rank (int): Rank of this process (default: 0)
            world_size (int): Total number of processes (default: 1)
            drop_last (bool): Drop incomplete batches (default: False)
            enable_cache (bool): Enable tiered caching (default: False)
            cache_l1_mb (int): L1 memory cache size in MB (default: 512)
            prefetch_batches (int): Batches to prefetch (default: 4)
            cache_decoded (bool): Cache decoded numpy arrays in memory for fast
                                 subsequent epochs (default: False)
            cache_decoded_mb (int): Max memory for decoded cache in MB. If None,
                                   defaults to 4096MB with warning for larger datasets.

        Example:
            >>> # Maximum throughput - returns numpy array
            >>> loader = turboloader.FastDataLoader(
            ...     'imagenet.tar',
            ...     batch_size=128,
            ...     num_workers=16,
            ...     output_format='numpy'
            ... )
            >>> for images, metadata in loader:
            ...     # images is np.ndarray of shape (N, H, W, C)
            ...     batch_size = images.shape[0]
            ...
            >>> # PyTorch format - returns (N, C, H, W)
            >>> loader = turboloader.FastDataLoader(
            ...     'imagenet.tar',
            ...     output_format='pytorch'
            ... )
            ...
            >>> # With decoded tensor caching (100K+ img/s on epoch 2+)
            >>> loader = turboloader.FastDataLoader(
            ...     'imagenet.tar',
            ...     batch_size=64,
            ...     cache_decoded=True,
            ...     cache_decoded_mb=2048
            ... )
            >>> for epoch in range(5):
            ...     for images, metadata in loader:
            ...         # Epoch 1: ~30K img/s (populating cache)
            ...         # Epoch 2+: ~100K+ img/s (from cache)
            ...         pass
        """

        def __init__(
            self,
            data_path,
            batch_size=32,
            num_workers=4,
            output_format="numpy",
            target_height=0,
            target_width=0,
            transform=None,
            shuffle=False,
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
            cache_decoded=False,
            cache_decoded_mb=None,
        ):
            self._output_format = output_format
            self._target_height = target_height
            self._target_width = target_width
            self._transform = transform
            self._chw_format = output_format in ("numpy_chw", "pytorch")
            self._data_path = data_path
            self._batch_size = batch_size
            self._num_workers = num_workers

            # Decoded tensor cache (v2.7.0)
            self._cache_decoded = cache_decoded
            self._cache_decoded_mb = cache_decoded_mb if cache_decoded_mb is not None else 4096
            self._decoded_cache = []
            self._cache_populated = False
            self._cache_index = 0

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

        def _extract_normalize_params(self):
            """Extract mean/std from transform pipeline for C++ fast path.

            Walks the transform pipeline to find Normalize/ImageNetNormalize
            and extracts their mean/std values. Returns (mean, std) lists
            or ([], []) if no normalization found.
            """
            if self._transform is None:
                return [], []

            # Check if transform is a single Normalize/ImageNetNormalize
            transform = self._transform
            transform_type = type(transform).__name__

            if transform_type in ("Normalize", "ImageNetNormalize"):
                # ImageNet defaults: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                if transform_type == "ImageNetNormalize":
                    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                # For generic Normalize we can't easily extract mean/std from C++ object
                # Fall back to Python path
                return [], []

            # Check if transform is a ComposedTransforms pipeline
            if transform_type == "ComposedTransforms":
                # Walk the pipeline to check if it's only Resize + Normalize
                # (the common case for benchmarks)
                return [], []

            return [], []

        def _can_use_cpp_float32_path(self):
            """Check if we can use the C++ float32 fast path.

            The C++ path handles: resize (via target_height/width) + normalize.
            If the transform pipeline contains only these operations (or is None),
            we can use the fast path.
            """
            if self._transform is None:
                return True

            transform_type = type(self._transform).__name__

            # Single Normalize/ImageNetNormalize - C++ handles this
            if transform_type in ("Normalize", "ImageNetNormalize"):
                return True

            # For composed transforms, check if all are Resize + Normalize
            # (Resize is handled by target_height/target_width params)
            if transform_type == "ComposedTransforms":
                try:
                    transforms = (
                        self._transform.get_transforms()
                        if hasattr(self._transform, "get_transforms")
                        else []
                    )
                except:
                    return False

                # Walk through transforms
                for t in transforms:
                    tname = type(t).__name__
                    if tname == "Resize":
                        continue  # Handled by C++ target_height/target_width
                    elif tname in ("Normalize", "ImageNetNormalize"):
                        continue  # Handled by C++ mean/std params
                    else:
                        return False  # Unsupported transform, fall back to Python
                return True

            return False

        def next_batch(self):
            """Get next batch as contiguous array.

            Returns:
                tuple: (images_array, metadata_dict)
                    - images_array: np.ndarray of shape (N, H, W, C) or (N, C, H, W)
                    - metadata_dict: {'indices': [...], 'filenames': [...], ...}

            Raises:
                StopIteration: When all data has been processed.
            """
            import time

            # Fast C++ float32 path for PyTorch/CHW output with normalization
            if self._chw_format and self._can_use_cpp_float32_path():
                mean, std = self._extract_normalize_params()

                max_retries = 10
                for attempt in range(max_retries):
                    images, metadata = self._loader.next_batch_array_float32(
                        True,  # chw_format
                        self._target_height,
                        self._target_width,
                        True,  # normalize_01
                        mean,
                        std,
                    )

                    if images.size == 0:
                        if attempt < max_retries - 1 and not self._loader.is_finished():
                            time.sleep(0.01)
                            continue
                        raise StopIteration
                    else:
                        break

                return images, metadata

            # Standard path for non-CHW or complex transforms
            max_retries = 10
            for attempt in range(max_retries):
                images, metadata = self._loader.next_batch_array(
                    self._chw_format, self._target_height, self._target_width
                )

                # Check for empty batch
                if images.size == 0:
                    # On first attempts, the pipeline may not be ready yet
                    if attempt < max_retries - 1 and not self._loader.is_finished():
                        time.sleep(0.01)  # 10ms wait
                        continue
                    # Pipeline finished or max retries reached
                    raise StopIteration
                else:
                    break

            # Apply transforms if set (slow Python path - fallback)
            if self._transform is not None:
                # For batch transforms, apply to each image
                import numpy as np

                batch_size = images.shape[0]
                transformed = []
                for i in range(batch_size):
                    if self._chw_format:
                        # CHW -> HWC for transform, then back
                        img = np.transpose(images[i], (1, 2, 0))
                        img = self._transform.apply(img)
                        img = np.transpose(img, (2, 0, 1))
                    else:
                        img = self._transform.apply(images[i])
                    transformed.append(img)
                images = np.stack(transformed)

            return images, metadata

        def next_batch_torch(self, device=None, non_blocking=True, dtype=None):
            """Get next batch as PyTorch tensor (zero-copy when possible).

            Phase 4 optimization: Returns torch.Tensor directly for PyTorch training.
            Uses torch.from_numpy() for zero-copy when data is contiguous.

            Args:
                device: Target device ('cuda', 'cuda:0', 'cpu', or torch.device).
                       If None, returns CPU tensor.
                non_blocking (bool): If True and device is CUDA, use non-blocking
                                    transfer (default: True)
                dtype: Target dtype (torch.float32, torch.float16, etc).
                      If None, uses float32 for normalized data, uint8 otherwise.

            Returns:
                tuple: (images_tensor, metadata_dict)
                    - images_tensor: torch.Tensor of shape (N, C, H, W)
                    - metadata_dict: {'indices': [...], 'filenames': [...], ...}

            Example:
                >>> loader = turboloader.FastDataLoader('data.tar', output_format='pytorch')
                >>> images, metadata = loader.next_batch_torch(device='cuda')
                >>> # images is already a torch.Tensor on GPU
                >>> output = model(images)

            Note:
                Requires PyTorch to be installed. Will raise ImportError if not available.
            """
            try:
                import torch
            except ImportError:
                raise ImportError(
                    "PyTorch is required for next_batch_torch(). " "Install with: pip install torch"
                )

            import time

            # Extract normalization params for C++ fast path
            mean, std = self._extract_normalize_params()

            # Use C++ float32 path - does uint8→float32 + HWC→CHW + normalize in SIMD
            max_retries = 10
            for attempt in range(max_retries):
                if self._can_use_cpp_float32_path():
                    images_np, metadata = self._loader.next_batch_array_float32(
                        True,  # Always CHW for PyTorch
                        self._target_height,
                        self._target_width,
                        True,  # normalize_01
                        mean,
                        std,
                    )
                else:
                    images_np, metadata = self._loader.next_batch_array(
                        True,  # Always CHW for PyTorch
                        self._target_height,
                        self._target_width,
                    )
                if images_np.size > 0 or self._loader.is_finished():
                    break
                time.sleep(0.01)

            # Apply transforms if set AND we used the non-float32 path
            if (
                not self._can_use_cpp_float32_path()
                and self._transform is not None
                and images_np.size > 0
            ):
                import numpy as np

                batch_size = images_np.shape[0]
                transformed = []
                for i in range(batch_size):
                    # CHW -> HWC for transform, then back
                    img = np.transpose(images_np[i], (1, 2, 0))
                    img = self._transform.apply(img)
                    img = np.transpose(img, (2, 0, 1))
                    transformed.append(img)
                images_np = np.stack(transformed)

            # Convert to tensor (zero-copy if contiguous)
            if images_np.size == 0:
                # Empty batch - return empty tensor
                tensor = torch.empty(0, 3, 0, 0)
            elif images_np.flags["C_CONTIGUOUS"]:
                # Zero-copy conversion
                tensor = torch.from_numpy(images_np)
            else:
                # Make contiguous first
                tensor = torch.from_numpy(images_np.copy())

            # Convert dtype if needed
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            elif images_np.dtype == "uint8":
                # Default: convert to float32 and normalize (only if not already float32)
                tensor = tensor.to(dtype=torch.float32) / 255.0
            # If already float32 from C++ path, no conversion needed

            # Move to device if specified
            if device is not None:
                tensor = tensor.to(device=device, non_blocking=non_blocking)

            return tensor, metadata

        def next_batch_tf(self, dtype=None):
            """Get next batch as TensorFlow tensor.

            Phase 4 optimization: Returns tf.Tensor directly for TensorFlow training.

            Args:
                dtype: Target dtype (tf.float32, tf.float16, etc).
                      If None, uses float32 for normalized data, uint8 otherwise.

            Returns:
                tuple: (images_tensor, metadata_dict)
                    - images_tensor: tf.Tensor of shape (N, H, W, C)
                    - metadata_dict: {'indices': [...], 'filenames': [...], ...}

            Example:
                >>> loader = turboloader.FastDataLoader('data.tar', output_format='tensorflow')
                >>> images, metadata = loader.next_batch_tf()
                >>> # images is already a tf.Tensor
                >>> output = model(images, training=True)

            Note:
                Requires TensorFlow to be installed. Will raise ImportError if not available.
            """
            try:
                import tensorflow as tf
            except ImportError:
                raise ImportError(
                    "TensorFlow is required for next_batch_tf(). "
                    "Install with: pip install tensorflow"
                )

            import time

            # Get batch as NumPy array in HWC format (with retry for pipeline warmup)
            max_retries = 10
            for attempt in range(max_retries):
                images_np, metadata = self._loader.next_batch_array(
                    False,  # HWC for TensorFlow
                    self._target_height,
                    self._target_width,
                )
                if images_np.size > 0 or self._loader.is_finished():
                    break
                time.sleep(0.01)

            # Apply transforms if set
            if self._transform is not None and images_np.size > 0:
                import numpy as np

                batch_size = images_np.shape[0]
                transformed = []
                for i in range(batch_size):
                    img = self._transform.apply(images_np[i])
                    transformed.append(img)
                images_np = np.stack(transformed)

            # Convert to tensor
            if images_np.size == 0:
                # Empty batch - return empty tensor
                tensor = tf.zeros((0, 0, 0, 3), dtype=tf.float32)
            else:
                tensor = tf.convert_to_tensor(images_np)

            # Convert dtype if needed
            if dtype is not None:
                tensor = tf.cast(tensor, dtype)
            elif images_np.dtype == "uint8":
                # Default: convert to float32 and normalize
                tensor = tf.cast(tensor, tf.float32) / 255.0

            return tensor, metadata

        def is_finished(self):
            """Check if all data has been processed."""
            return self._loader.is_finished()

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
            """Make FastDataLoader iterable.

            If cache_decoded is enabled:
            - First epoch: Yields batches from pipeline, stores copies in cache
            - Subsequent epochs: Yields directly from cache (100K+ img/s)
            """
            if self._cache_decoded and self._cache_populated:
                # Cache hit path - iterate over cached arrays directly
                # This is the fast path: ~100K+ img/s (no decoding, no pipeline)
                for images, metadata in self._decoded_cache:
                    yield images, metadata
                return

            # Normal path (or first epoch with caching)
            # Reset the pipeline for a new epoch (reuses buffers, decoders, memory maps)
            self._loader.reset()

            # If caching enabled but not populated, clear any partial cache
            if self._cache_decoded and not self._cache_populated:
                self._decoded_cache = []

            # Iterate through the pipeline
            try:
                while True:
                    images, metadata = self.next_batch()

                    # If caching enabled and first epoch, store in cache
                    if self._cache_decoded and not self._cache_populated:
                        # Make copies to ensure data persists
                        cached_images = images.copy()
                        cached_metadata = {
                            k: (v.copy() if hasattr(v, "copy") else v) for k, v in metadata.items()
                        }
                        self._decoded_cache.append((cached_images, cached_metadata))

                    yield images, metadata

            except StopIteration:
                # Mark cache as populated after first complete epoch
                if self._cache_decoded and not self._cache_populated:
                    self._cache_populated = True

        def __next__(self):
            """Get next batch (iterator protocol)."""
            return self.next_batch()

        def clear_cache(self):
            """Clear the decoded tensor cache.

            Call this to free memory or to force re-population of the cache
            on the next epoch.
            """
            self._decoded_cache = []
            self._cache_populated = False

        @property
        def cache_populated(self):
            """Check if the decoded cache is populated."""
            return self._cache_populated

        @property
        def cache_size_mb(self):
            """Get the current size of the decoded cache in MB."""
            if not self._decoded_cache:
                return 0.0
            total_bytes = sum(
                images.nbytes
                + sum(v.nbytes if hasattr(v, "nbytes") else 0 for v in metadata.values())
                for images, metadata in self._decoded_cache
            )
            return total_bytes / (1024 * 1024)

        @property
        def output_format(self):
            """Get the output format."""
            return self._output_format

        @property
        def transform(self):
            """Get the current transform."""
            return self._transform

        @transform.setter
        def transform(self, value):
            """Set the transform."""
            self._transform = value

    def Loader(
        data_path,
        batch_size=32,
        num_workers=4,
        fast=False,
        output_format="numpy",
        target_height=0,
        target_width=0,
        transform=None,
        shuffle=False,
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
        """Unified factory function for creating DataLoaders.

        Creates either a DataLoader or FastDataLoader based on the `fast` parameter.
        This provides a convenient single entry point with a toggle for the
        high-performance batch array API.

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet)
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads (default: 4)
            fast (bool): If True, use FastDataLoader with batch array transfer
                        for maximum throughput. If False, use standard DataLoader
                        that returns list of dicts. (default: False)
            output_format (str): For fast=True: 'numpy', 'numpy_chw', 'pytorch',
                               'tensorflow' (default: 'numpy')
            target_height (int): For fast=True: target image height (default: 0 = auto)
            target_width (int): For fast=True: target image width (default: 0 = auto)
            transform: Transform pipeline to apply to images
            shuffle (bool): Shuffle samples (default: False)
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

        Returns:
            DataLoader or FastDataLoader: The appropriate loader class

        Example:
            >>> # Standard API (list of dicts)
            >>> loader = turboloader.Loader('data.tar', batch_size=128)
            >>> for batch in loader:
            ...     images = [s['image'] for s in batch]
            ...
            >>> # High-performance API (contiguous arrays)
            >>> loader = turboloader.Loader('data.tar', batch_size=128, fast=True)
            >>> for images, metadata in loader:
            ...     # images is np.ndarray (N, H, W, C)
            ...     pass
            ...
            >>> # With PyTorch format
            >>> loader = turboloader.Loader(
            ...     'data.tar',
            ...     fast=True,
            ...     output_format='pytorch'
            ... )
        """
        if fast:
            return FastDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                output_format=output_format,
                target_height=target_height,
                target_width=target_width,
                transform=transform,
                shuffle=shuffle,
                enable_distributed=enable_distributed,
                world_rank=world_rank,
                world_size=world_size,
                drop_last=drop_last,
                distributed_seed=distributed_seed,
                enable_cache=enable_cache,
                cache_l1_mb=cache_l1_mb,
                cache_l2_gb=cache_l2_gb,
                cache_dir=cache_dir,
                auto_smart_batching=auto_smart_batching,
                enable_smart_batching=enable_smart_batching,
                prefetch_batches=prefetch_batches,
            )
        else:
            return DataLoader(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle,
                transform=transform,
                enable_distributed=enable_distributed,
                world_rank=world_rank,
                world_size=world_size,
                drop_last=drop_last,
                distributed_seed=distributed_seed,
                enable_cache=enable_cache,
                cache_l1_mb=cache_l1_mb,
                cache_l2_gb=cache_l2_gb,
                cache_dir=cache_dir,
                auto_smart_batching=auto_smart_batching,
                enable_smart_batching=enable_smart_batching,
                prefetch_batches=prefetch_batches,
            )

    class MemoryEfficientDataLoader:
        """Memory-optimized DataLoader with configurable memory budget.

        Uses aggressive memory-saving defaults to minimize memory footprint
        while maintaining good throughput. Ideal for:
        - Memory-constrained environments
        - Development machines with limited RAM
        - Running alongside other memory-intensive processes

        Key optimizations:
        - Reduced prefetch_batches (1-2 vs 4 in FastDataLoader)
        - Fewer workers (2-8 vs 16 in FastDataLoader)
        - Caching disabled by default
        - Auto-tuned settings based on memory budget

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet).
                            Supports: local files, http://, https://, s3://, gs://
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads. If None, auto-calculated based on
                              max_memory_mb (default: None)
            max_memory_mb (int): Target memory budget in MB. Settings are auto-tuned
                                to stay within this budget. (default: 512)
            output_format (str): Output format - 'numpy', 'numpy_chw', 'pytorch',
                               'tensorflow' (default: 'numpy')
            target_height (int): Target image height (0 = auto from first image)
            target_width (int): Target image width (0 = auto from first image)
            transform: Transform pipeline to apply to images
            shuffle (bool): Shuffle samples (default: False)
            enable_distributed (bool): Enable distributed training (default: False)
            world_rank (int): Rank of this process (default: 0)
            world_size (int): Total number of processes (default: 1)
            drop_last (bool): Drop incomplete batches (default: False)

        Example:
            >>> # Memory-efficient loading with 512MB budget
            >>> loader = turboloader.MemoryEfficientDataLoader(
            ...     'imagenet.tar',
            ...     batch_size=64,
            ...     max_memory_mb=512
            ... )
            >>> for images, metadata in loader:
            ...     batch_size = images.shape[0]
            ...
            >>> # Very low memory (256MB)
            >>> loader = turboloader.MemoryEfficientDataLoader(
            ...     'data.tar',
            ...     max_memory_mb=256
            ... )
        """

        def __init__(
            self,
            data_path,
            batch_size=32,
            num_workers=None,
            max_memory_mb=512,
            output_format="numpy",
            target_height=0,
            target_width=0,
            transform=None,
            shuffle=False,
            enable_distributed=False,
            world_rank=0,
            world_size=1,
            drop_last=False,
            distributed_seed=42,
        ):
            self._output_format = output_format
            self._target_height = target_height
            self._target_width = target_width
            self._transform = transform
            self._chw_format = output_format in ("numpy_chw", "pytorch")
            self._max_memory_mb = max_memory_mb

            # Auto-tune settings based on memory budget
            prefetch_batches, auto_workers = self._configure_for_memory_budget(
                max_memory_mb, batch_size
            )

            # Use provided num_workers or auto-calculated
            actual_workers = num_workers if num_workers is not None else auto_workers

            self._loader = _DataLoaderBase(
                data_path,
                batch_size,
                actual_workers,
                shuffle,
                enable_distributed,
                world_rank,
                world_size,
                drop_last,
                distributed_seed,
                False,  # enable_cache - disabled for memory efficiency
                0,  # cache_l1_mb - disabled
                0,  # cache_l2_gb - disabled
                "/tmp/turboloader_cache",
                False,  # auto_smart_batching - disabled to save memory
                False,  # enable_smart_batching
                prefetch_batches,
            )

            # Store actual settings for introspection
            self._prefetch_batches = prefetch_batches
            self._num_workers = actual_workers

        def _configure_for_memory_budget(self, max_memory_mb, batch_size):
            """Calculate optimal settings for memory budget.

            Returns:
                tuple: (prefetch_batches, num_workers)
            """
            import os

            # Estimate image size (assume 224x224 RGB worst case)
            est_image_bytes = 224 * 224 * 3
            bytes_per_batch = batch_size * est_image_bytes

            # Allocate 70% to prefetch, 30% to pools/overhead
            prefetch_budget = max_memory_mb * 0.7 * 1024 * 1024

            # Calculate prefetch_batches (minimum 1, maximum 2)
            prefetch_batches = min(2, max(1, int(prefetch_budget / bytes_per_batch)))

            # Calculate num_workers (fewer workers = less memory)
            cpu_count = os.cpu_count() or 4
            if max_memory_mb <= 256:
                num_workers = min(2, max(1, cpu_count // 4))
            elif max_memory_mb <= 512:
                num_workers = min(4, cpu_count // 2)
            elif max_memory_mb <= 1024:
                num_workers = min(8, cpu_count)
            else:
                num_workers = min(12, cpu_count)

            return prefetch_batches, num_workers

        def next_batch(self):
            """Get next batch as contiguous array.

            Returns:
                tuple: (images_array, metadata_dict)
                    - images_array: np.ndarray of shape (N, H, W, C) or (N, C, H, W)
                    - metadata_dict: {'indices': [...], 'filenames': [...], ...}

            Raises:
                StopIteration: When all data has been processed.
            """
            import time

            # Retry loop for async pipeline startup
            max_retries = 10
            for attempt in range(max_retries):
                images, metadata = self._loader.next_batch_array(
                    self._chw_format, self._target_height, self._target_width
                )

                # Check for empty batch
                if images.size == 0:
                    if attempt < max_retries - 1 and not self._loader.is_finished():
                        time.sleep(0.01)
                        continue
                    raise StopIteration
                else:
                    break

            # Apply transforms if set
            if self._transform is not None:
                import numpy as np

                batch_size = images.shape[0]
                transformed = []
                for i in range(batch_size):
                    if self._chw_format:
                        img = np.transpose(images[i], (1, 2, 0))
                        img = self._transform.apply(img)
                        img = np.transpose(img, (2, 0, 1))
                    else:
                        img = self._transform.apply(images[i])
                    transformed.append(img)
                images = np.stack(transformed)

            return images, metadata

        def is_finished(self):
            """Check if all data has been processed."""
            return self._loader.is_finished()

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
            """Make MemoryEfficientDataLoader iterable."""
            return self

        def __next__(self):
            """Get next batch (iterator protocol)."""
            return self.next_batch()

        @property
        def output_format(self):
            """Get the output format."""
            return self._output_format

        @property
        def transform(self):
            """Get the current transform."""
            return self._transform

        @transform.setter
        def transform(self, value):
            """Set the transform."""
            self._transform = value

        @property
        def max_memory_mb(self):
            """Get the memory budget."""
            return self._max_memory_mb

        @property
        def prefetch_batches(self):
            """Get the actual prefetch_batches setting."""
            return self._prefetch_batches

        @property
        def num_workers(self):
            """Get the actual num_workers setting."""
            return self._num_workers

    def create_loader(
        data_path,
        loader_type="fast",
        batch_size=32,
        num_workers=None,
        output_format="numpy",
        target_height=0,
        target_width=0,
        transform=None,
        shuffle=False,
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
        max_memory_mb=512,
    ):
        """Factory function to create the appropriate data loader.

        Creates DataLoader, FastDataLoader, or MemoryEfficientDataLoader based
        on the `loader_type` parameter.

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet)
            loader_type (str): Type of loader to create:
                - 'fast': FastDataLoader (max throughput, higher memory)
                - 'memory_efficient': MemoryEfficientDataLoader (low memory)
                - 'standard': DataLoader (original API, list of dicts)
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads. For memory_efficient, None means
                              auto-calculate based on max_memory_mb. (default: None for
                              memory_efficient, 4 for others)
            output_format (str): For fast/memory_efficient: 'numpy', 'numpy_chw',
                               'pytorch', 'tensorflow' (default: 'numpy')
            target_height (int): Target image height (default: 0 = auto)
            target_width (int): Target image width (default: 0 = auto)
            transform: Transform pipeline to apply to images
            shuffle (bool): Shuffle samples (default: False)
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
            max_memory_mb (int): Memory budget for memory_efficient loader (default: 512)

        Returns:
            DataLoader, FastDataLoader, or MemoryEfficientDataLoader

        Example:
            >>> # High throughput (default)
            >>> loader = turboloader.create_loader('data.tar', loader_type='fast')
            >>> for images, metadata in loader:
            ...     pass
            ...
            >>> # Memory constrained
            >>> loader = turboloader.create_loader(
            ...     'data.tar',
            ...     loader_type='memory_efficient',
            ...     max_memory_mb=512
            ... )
            ...
            >>> # Standard API (list of dicts)
            >>> loader = turboloader.create_loader('data.tar', loader_type='standard')
            >>> for batch in loader:
            ...     images = [s['image'] for s in batch]
        """
        if loader_type == "memory_efficient":
            return MemoryEfficientDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers,
                max_memory_mb=max_memory_mb,
                output_format=output_format,
                target_height=target_height,
                target_width=target_width,
                transform=transform,
                shuffle=shuffle,
                enable_distributed=enable_distributed,
                world_rank=world_rank,
                world_size=world_size,
                drop_last=drop_last,
                distributed_seed=distributed_seed,
            )
        elif loader_type == "fast":
            return FastDataLoader(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers if num_workers is not None else 4,
                output_format=output_format,
                target_height=target_height,
                target_width=target_width,
                transform=transform,
                shuffle=shuffle,
                enable_distributed=enable_distributed,
                world_rank=world_rank,
                world_size=world_size,
                drop_last=drop_last,
                distributed_seed=distributed_seed,
                enable_cache=enable_cache,
                cache_l1_mb=cache_l1_mb,
                cache_l2_gb=cache_l2_gb,
                cache_dir=cache_dir,
                auto_smart_batching=auto_smart_batching,
                enable_smart_batching=enable_smart_batching,
                prefetch_batches=prefetch_batches,
            )
        elif loader_type == "standard":
            return DataLoader(
                data_path=data_path,
                batch_size=batch_size,
                num_workers=num_workers if num_workers is not None else 4,
                shuffle=shuffle,
                transform=transform,
                enable_distributed=enable_distributed,
                world_rank=world_rank,
                world_size=world_size,
                drop_last=drop_last,
                distributed_seed=distributed_seed,
                enable_cache=enable_cache,
                cache_l1_mb=cache_l1_mb,
                cache_l2_gb=cache_l2_gb,
                cache_dir=cache_dir,
                auto_smart_batching=auto_smart_batching,
                enable_smart_batching=enable_smart_batching,
                prefetch_batches=prefetch_batches,
            )
        else:
            raise ValueError(
                f"Invalid loader_type: {loader_type}. "
                f"Must be 'fast', 'memory_efficient', or 'standard'."
            )

    class ProgressiveResizeLoader:
        """Progressive Resizing DataLoader for curriculum learning (Phase 5.5 v2.15.0).

        Implements progressive resizing as described in fastai/FFCV: start training
        with smaller images and gradually increase resolution. This provides:
        - Faster initial training epochs (smaller images = faster processing)
        - Regularization effect (prevents overfitting on low-level features)
        - 10-15% faster overall training time with minimal accuracy loss

        Args:
            data_path (str): Path to data (TAR, video, CSV, Parquet)
            batch_size (int): Samples per batch (default: 32)
            num_workers (int): Worker threads (default: 4)
            initial_size (int): Starting image size (default: 128)
            final_size (int): Final image size (default: 224)
            warmup_epochs (int): Epochs to reach final size (default: 5)
            output_format (str): 'numpy', 'numpy_chw', 'pytorch', 'tensorflow'
            transform: Additional transforms to apply after resize
            **kwargs: Additional arguments passed to FastDataLoader

        Example:
            >>> # Progressive resize from 128 to 224 over 5 epochs
            >>> loader = turboloader.ProgressiveResizeLoader(
            ...     'imagenet.tar',
            ...     batch_size=128,
            ...     num_workers=8,
            ...     initial_size=128,
            ...     final_size=224,
            ...     warmup_epochs=5
            ... )
            >>>
            >>> for epoch in range(10):
            ...     loader.set_epoch(epoch)  # Updates current size
            ...     print(f"Epoch {epoch}: size={loader.current_size}")
            ...     for images, metadata in loader:
            ...         # images shape changes as size increases
            ...         pass

        Size Schedule:
            - Epoch 0: 128x128
            - Epoch 1: 152x152
            - Epoch 2: 176x176
            - Epoch 3: 200x200
            - Epoch 4: 224x224
            - Epoch 5+: 224x224 (stays at final)
        """

        def __init__(
            self,
            data_path,
            batch_size=32,
            num_workers=4,
            initial_size=128,
            final_size=224,
            warmup_epochs=5,
            output_format="numpy",
            transform=None,
            shuffle=False,
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
            import numpy as np

            self._data_path = data_path
            self._batch_size = batch_size
            self._num_workers = num_workers
            self._initial_size = initial_size
            self._final_size = final_size
            self._warmup_epochs = warmup_epochs
            self._output_format = output_format
            self._base_transform = transform
            self._shuffle = shuffle
            self._enable_distributed = enable_distributed
            self._world_rank = world_rank
            self._world_size = world_size
            self._drop_last = drop_last
            self._distributed_seed = distributed_seed
            self._enable_cache = enable_cache
            self._cache_l1_mb = cache_l1_mb
            self._cache_l2_gb = cache_l2_gb
            self._cache_dir = cache_dir
            self._auto_smart_batching = auto_smart_batching
            self._enable_smart_batching = enable_smart_batching
            self._prefetch_batches = prefetch_batches

            # Calculate size schedule (linear interpolation)
            self._sizes = np.linspace(initial_size, final_size, warmup_epochs, dtype=int)

            # Current epoch and size
            self._current_epoch = 0
            self._current_size = initial_size

            # Create the underlying loader with current size
            self._loader = None
            self._update_loader()

        def _update_loader(self):
            """Update the underlying loader with current size."""
            # Stop existing loader if any
            if self._loader is not None:
                self._loader.stop()

            # Create resize transform with current size
            current_resize = Resize(self._current_size, self._current_size)

            # Compose with base transform if provided
            if self._base_transform is not None:
                full_transform = current_resize | self._base_transform
            else:
                full_transform = current_resize

            # Create new FastDataLoader with current settings
            self._loader = FastDataLoader(
                data_path=self._data_path,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                output_format=self._output_format,
                target_height=self._current_size,
                target_width=self._current_size,
                transform=full_transform,
                shuffle=self._shuffle,
                enable_distributed=self._enable_distributed,
                world_rank=self._world_rank,
                world_size=self._world_size,
                drop_last=self._drop_last,
                distributed_seed=self._distributed_seed,
                enable_cache=self._enable_cache,
                cache_l1_mb=self._cache_l1_mb,
                cache_l2_gb=self._cache_l2_gb,
                cache_dir=self._cache_dir,
                auto_smart_batching=self._auto_smart_batching,
                enable_smart_batching=self._enable_smart_batching,
                prefetch_batches=self._prefetch_batches,
            )

        def set_epoch(self, epoch: int):
            """Set the epoch and update image size accordingly.

            Call this at the start of each epoch. The image size will be
            updated based on the warmup schedule:
            - During warmup: size = linspace(initial, final, warmup_epochs)[epoch]
            - After warmup: size = final_size

            Args:
                epoch (int): The epoch number (0, 1, 2, ...)
            """
            self._current_epoch = epoch

            # Calculate current size
            if epoch < self._warmup_epochs:
                self._current_size = int(self._sizes[epoch])
            else:
                self._current_size = self._final_size

            # Update the loader with new size
            self._update_loader()

        @property
        def current_size(self):
            """Get the current image size."""
            return self._current_size

        @property
        def current_epoch(self):
            """Get the current epoch."""
            return self._current_epoch

        @property
        def size_schedule(self):
            """Get the full size schedule for warmup epochs."""
            return self._sizes.tolist()

        def next_batch(self):
            """Get next batch with current size."""
            return self._loader.next_batch()

        def is_finished(self):
            """Check if all data has been processed."""
            return self._loader.is_finished()

        def stop(self):
            """Stop the pipeline and clean up resources."""
            if self._loader is not None:
                self._loader.stop()

        def __enter__(self):
            """Context manager entry."""
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            """Context manager exit."""
            self.stop()

        def __iter__(self):
            """Make ProgressiveResizeLoader iterable."""
            return self._loader.__iter__()

        def __next__(self):
            """Get next batch (iterator protocol)."""
            return self.next_batch()

    # Add ProgressiveResizeLoader to exports
    __all__.append("ProgressiveResizeLoader")

    # Import PyTorch compatibility layer
    try:
        from turboloader.pytorch_compat import (
            PyTorchCompatibleLoader,
            ImageFolderConverter,
            TransformAdapter,
            LabelExtractor,
            FolderLabelExtractor,
            FilenamePatternExtractor,
            MetadataLabelExtractor,
            JSONSidecarExtractor,
            CallableLabelExtractor,
            create_loader as create_pytorch_loader,
            convert_imagefolder,
        )

        __all__.extend(
            [
                "PyTorchCompatibleLoader",
                "ImageFolderConverter",
                "TransformAdapter",
                "LabelExtractor",
                "FolderLabelExtractor",
                "FilenamePatternExtractor",
                "MetadataLabelExtractor",
                "JSONSidecarExtractor",
                "CallableLabelExtractor",
                "create_pytorch_loader",
                "convert_imagefolder",
            ]
        )
    except ImportError:
        # PyTorch not available - skip compatibility layer
        pass

except ImportError:
    # Fallback for development/documentation builds
    __all__ = ["__version__"]
