"""WebDataset-style iterator API for TurboLoader

This module provides a PyTorch-compatible DataLoader interface that wraps
TurboLoader's high-performance C++ pipeline with a Pythonic iterator API.

Example:
    >>> from turboloader import WebDatasetLoader
    >>> loader = WebDatasetLoader(
    ...     urls=['imagenet-{000000..001281}.tar'],
    ...     batch_size=256,
    ...     num_workers=8,
    ...     shuffle=True
    ... )
    >>> for batch in loader:
    ...     images, labels = batch['image'], batch['label']
    ...     # Training code...
"""

import sys
import os
from typing import List, Optional, Callable, Iterator, Dict, Any, Union
import warnings

# Import the C++ extension
try:
    # Try to import from installed package
    import turboloader as _tl
except ImportError:
    # Fallback to build directory during development
    sys.path.insert(0, 'build/python')
    import turboloader as _tl


def default_collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Default collation function that batches samples into tensors.

    Args:
        samples: List of sample dictionaries

    Returns:
        Dictionary with batched tensors
    """
    if len(samples) == 0:
        return {}

    # Get keys from first sample
    keys = samples[0].data.keys()

    batch = {}
    for key in keys:
        # Collect all values for this key
        values = [sample.data[key] for sample in samples]

        # For image data, stack into batch
        if key in ['jpg', 'png', 'webp', 'image']:
            # Convert to numpy arrays if needed
            import numpy as np
            arrays = []
            for sample in samples:
                if hasattr(sample, 'transformed_data') and len(sample.transformed_data) > 0:
                    # Use transformed data if available
                    arrays.append(np.array(sample.transformed_data).reshape(
                        sample.height, sample.width, sample.channels
                    ))
                else:
                    # Use raw image data
                    arrays.append(np.array(sample.data[key]))

            batch[key] = np.stack(arrays) if arrays else np.array([])
        else:
            # For other data (labels, metadata), just collect as list
            batch[key] = values

    # Add metadata
    batch['_indices'] = [sample.index for sample in samples]

    return batch


class WebDatasetLoader:
    """PyTorch-compatible DataLoader for WebDataset format TAR files.

    This class provides an iterator interface over TurboLoader's high-performance
    C++ pipeline, making it a drop-in replacement for PyTorch DataLoader.

    Args:
        urls: List of TAR file paths or glob patterns (e.g., 'data-{000..999}.tar')
        batch_size: Number of samples per batch
        num_workers: Number of worker threads for parallel loading
        shuffle: Whether to shuffle samples each epoch
        transforms: List of transformation functions to apply
        collate_fn: Function to collate samples into batches
        decode_jpeg: Whether to decode JPEG images automatically
        enable_simd_transforms: Whether to use SIMD-optimized transforms
        transform_config: Configuration for SIMD transforms
        drop_last: Whether to drop the last incomplete batch

    Example:
        >>> loader = WebDatasetLoader(
        ...     urls=['train-{000..127}.tar'],
        ...     batch_size=256,
        ...     num_workers=16,
        ...     shuffle=True
        ... )
        >>> for epoch in range(100):
        ...     for batch in loader:
        ...         images = batch['image']
        ...         # Training code...
    """

    def __init__(
        self,
        urls: Union[str, List[str]],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transforms: Optional[List[Callable]] = None,
        collate_fn: Optional[Callable] = None,
        decode_jpeg: bool = True,
        enable_simd_transforms: bool = False,
        transform_config: Optional[Any] = None,
        drop_last: bool = False,
        queue_size: int = 1024
    ):
        """Initialize the WebDataset loader."""

        # Expand URL patterns
        if isinstance(urls, str):
            urls = [urls]
        self.urls = self._expand_urls(urls)

        if not self.urls:
            raise ValueError("No TAR files found matching the provided URLs")

        # Store configuration
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transforms = transforms or []
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last

        # Create TurboLoader config
        self.config = _tl.Config()
        self.config.num_workers = num_workers
        self.config.batch_size = batch_size
        self.config.shuffle = shuffle
        self.config.decode_jpeg = decode_jpeg
        self.config.queue_size = queue_size
        self.config.enable_simd_transforms = enable_simd_transforms

        if transform_config is not None:
            self.config.transform_config = transform_config

        # Create the pipeline
        self.pipeline = _tl.Pipeline(self.urls, self.config)

        # Track state
        self.epoch = 0
        self._iterator = None

    def _expand_urls(self, urls: List[str]) -> List[str]:
        """Expand glob patterns in URLs.

        Supports patterns like:
        - 'data-{000..999}.tar' -> ['data-000.tar', 'data-001.tar', ...]
        - 'shard*.tar' -> all matching files
        """
        import glob
        import re

        expanded = []
        for url in urls:
            # Check for brace expansion pattern
            match = re.match(r'(.*)\{(\d+)\.\.(\d+)\}(.*)', url)
            if match:
                prefix, start, end, suffix = match.groups()
                width = len(start)  # Preserve leading zeros
                for i in range(int(start), int(end) + 1):
                    expanded.append(f"{prefix}{i:0{width}d}{suffix}")
            else:
                # Regular glob pattern
                matches = glob.glob(url)
                if matches:
                    expanded.extend(sorted(matches))
                else:
                    # If no matches, assume it's a literal path
                    if os.path.exists(url):
                        expanded.append(url)

        return expanded

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return iterator over batches.

        Resets the pipeline and creates a new iterator for the epoch.
        """
        # Reset pipeline for new epoch
        if self.epoch > 0:
            self.pipeline.reset()

        self.pipeline.start()
        self._iterator = self._iterate_batches()
        return self._iterator

    def _iterate_batches(self) -> Iterator[Dict[str, Any]]:
        """Internal generator for batches."""
        while True:
            # Get next batch from C++ pipeline
            batch_samples = self.pipeline.next_batch(self.batch_size)

            if len(batch_samples) == 0:
                break  # End of epoch

            # Drop last incomplete batch if requested
            if self.drop_last and len(batch_samples) < self.batch_size:
                break

            # Apply Python transforms if any
            if self.transforms:
                batch_samples = [
                    self._apply_transforms(sample) for sample in batch_samples
                ]

            # Collate into batch
            batch = self.collate_fn(batch_samples)

            yield batch

        # Increment epoch counter
        self.epoch += 1

        # Stop pipeline
        self.pipeline.stop()

    def _apply_transforms(self, sample: Any) -> Any:
        """Apply Python transforms to a single sample."""
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        total_samples = self.pipeline.total_samples()
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size

    def __repr__(self) -> str:
        """String representation of the loader."""
        return (
            f"WebDatasetLoader(\n"
            f"  num_files={len(self.urls)},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  shuffle={self.shuffle},\n"
            f"  epoch={self.epoch}\n"
            f")"
        )


class DistributedWebDatasetLoader(WebDatasetLoader):
    """Distributed version of WebDatasetLoader for multi-GPU training.

    Automatically shards data across multiple processes/GPUs and ensures
    each process sees different data.

    Args:
        rank: Current process rank (0 to world_size-1)
        world_size: Total number of processes
        **kwargs: All arguments from WebDatasetLoader

    Example:
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend='nccl')
        >>> loader = DistributedWebDatasetLoader(
        ...     urls=['train-{000..127}.tar'],
        ...     batch_size=256,
        ...     rank=dist.get_rank(),
        ...     world_size=dist.get_world_size()
        ... )
    """

    def __init__(
        self,
        urls: Union[str, List[str]],
        rank: int,
        world_size: int,
        **kwargs
    ):
        """Initialize distributed loader."""
        self.rank = rank
        self.world_size = world_size

        # Shard URLs across processes
        all_urls = self._expand_urls([urls] if isinstance(urls, str) else urls)
        sharded_urls = all_urls[rank::world_size]

        if not sharded_urls:
            warnings.warn(
                f"Process {rank} has no data shards! "
                f"Total shards: {len(all_urls)}, world_size: {world_size}"
            )

        # Initialize base class with sharded URLs
        super().__init__(urls=sharded_urls, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DistributedWebDatasetLoader(\n"
            f"  num_files={len(self.urls)},\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_workers={self.num_workers},\n"
            f"  shuffle={self.shuffle},\n"
            f"  rank={self.rank}/{self.world_size},\n"
            f"  epoch={self.epoch}\n"
            f")"
        )


# Convenience aliases
DataLoader = WebDatasetLoader
DistributedDataLoader = DistributedWebDatasetLoader
