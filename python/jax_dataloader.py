"""
JAX/Flax integration for TurboLoader

Provides JAX-compatible data loading with automatic device placement.
"""

import numpy as np
from typing import Optional, Callable, Iterator, Tuple, Dict, Any

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("JAX is required for JAX integration. "
                      "Install it with: pip install jax jaxlib")

try:
    from _turboloader import DataLoader as _CppDataLoader
except ImportError:
    raise ImportError("TurboLoader C++ extension not found. "
                      "Make sure TurboLoader is properly installed.")


class JAXDataLoader:
    """
    JAX-compatible DataLoader using TurboLoader backend.

    Automatically transfers data to JAX devices and provides
    iterator interface compatible with Flax training loops.

    Example:
        >>> loader = JAXDataLoader('imagenet.tar', batch_size=32, num_workers=8)
        >>> for batch in loader:
        >>>     loss = train_step(state, batch)
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
        device: Optional[jax.Device] = None,
        prefetch_size: int = 2,
        prefetch: Optional[int] = None,  # Alias for prefetch_size
    ):
        """
        Initialize JAX DataLoader.

        Args:
            data_path: Path to data (TAR, video, CSV, Parquet)
            batch_size: Samples per batch
            num_workers: Worker threads
            shuffle: Shuffle samples
            transform: Optional transform function (numpy -> numpy)
            device: JAX device to place data on (default: jax.devices()[0])
            prefetch_size: Number of batches to prefetch
            prefetch: Alias for prefetch_size (for compatibility)
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform
        self.device = device or jax.devices()[0]
        # Use prefetch if provided, otherwise use prefetch_size
        self.prefetch_size = prefetch if prefetch is not None else prefetch_size

        # Create C++ backend loader
        self._cpp_loader = _CppDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over batches"""
        while not self._cpp_loader.is_finished():
            batch = self._cpp_loader.next_batch()

            if not batch:
                break

            # Convert batch to JAX arrays
            images = []
            labels = []

            for sample in batch:
                img = sample['image']  # NumPy array (H, W, C)

                # Apply transform if provided
                if self.transform is not None:
                    img = self.transform(img)

                images.append(img)

                # Extract label from filename
                filename = sample.get('filename', '')
                label = 0  # Default label
                labels.append(label)

            if images:
                images_np = np.stack(images, axis=0)
                labels_np = np.array(labels, dtype=np.int64)

                # Convert to JAX arrays and place on device
                images_jax = jax.device_put(images_np, self.device)
                labels_jax = jax.device_put(labels_np, self.device)

                yield {
                    'image': images_jax,
                    'label': labels_jax,
                    'filename': [s['filename'] for s in batch],
                }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cpp_loader.stop()


class FlaxDataLoader:
    """
    Flax-optimized DataLoader with support for data parallelism.

    Automatically shards data across multiple devices for
    multi-GPU/TPU training.

    Example:
        >>> loader = FlaxDataLoader('data.tar', batch_size=32, num_workers=8)
        >>> for batch in loader:
        >>>     # batch['image'] has shape (num_devices, batch_per_device, H, W, C)
        >>>     state, metrics = train_step(state, batch)
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
        num_devices: Optional[int] = None,
    ):
        """
        Initialize Flax DataLoader.

        Args:
            data_path: Path to data
            batch_size: Total batch size (will be divided across devices)
            num_workers: Worker threads
            shuffle: Shuffle samples
            transform: Optional transform function
            num_devices: Number of devices (default: all available)
        """
        self.data_path = data_path
        self.num_devices = num_devices or jax.device_count()

        # Batch size must be divisible by number of devices
        if batch_size % self.num_devices != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_devices ({self.num_devices})"
            )

        self.batch_size = batch_size
        self.batch_size_per_device = batch_size // self.num_devices
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform

        # Create C++ backend loader
        self._cpp_loader = _CppDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterate over batches with device sharding"""
        while not self._cpp_loader.is_finished():
            batch = self._cpp_loader.next_batch()

            if not batch:
                break

            # Convert batch to arrays
            images = []
            labels = []

            for sample in batch:
                img = sample['image']
                if self.transform is not None:
                    img = self.transform(img)
                images.append(img)
                labels.append(0)  # Default label

            if images:
                images_np = np.stack(images, axis=0)
                labels_np = np.array(labels, dtype=np.int64)

                # Reshape for device parallelism: (num_devices, batch_per_device, ...)
                images_sharded = images_np.reshape(
                    self.num_devices,
                    self.batch_size_per_device,
                    *images_np.shape[1:]
                )
                labels_sharded = labels_np.reshape(
                    self.num_devices,
                    self.batch_size_per_device
                )

                # Convert to JAX and shard across devices
                images_jax = jax.device_put_sharded(
                    [images_sharded[i] for i in range(self.num_devices)],
                    jax.devices()
                )
                labels_jax = jax.device_put_sharded(
                    [labels_sharded[i] for i in range(self.num_devices)],
                    jax.devices()
                )

                yield {
                    'image': images_jax,
                    'label': labels_jax,
                }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cpp_loader.stop()


def prefetch_to_device(
    iterator: Iterator,
    size: int,
    device: Optional[jax.Device] = None
) -> Iterator:
    """
    Prefetch batches to device asynchronously.

    Args:
        iterator: Data iterator
        size: Number of batches to prefetch
        device: Target device (default: first device)

    Returns:
        Iterator with prefetched batches

    Example:
        >>> loader = JAXDataLoader('data.tar')
        >>> prefetched = prefetch_to_device(iter(loader), size=2)
        >>> for batch in prefetched:
        >>>     ...
    """
    device = device or jax.devices()[0]

    queue = []

    def enqueue(n):
        for _ in range(n):
            try:
                item = next(iterator)
                # Transfer to device asynchronously
                queue.append(jax.device_put(item, device))
            except StopIteration:
                return

    # Initial prefetch
    enqueue(size)

    while queue:
        yield queue.pop(0)
        enqueue(1)
