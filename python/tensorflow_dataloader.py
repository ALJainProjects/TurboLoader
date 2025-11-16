"""
TensorFlow/Keras integration for TurboLoader

Provides tf.data.Dataset compatible interface for high-performance data loading.
"""

import numpy as np
from typing import Optional, Tuple, Callable

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow is required for TensorFlow integration. "
                      "Install it with: pip install tensorflow")

try:
    from _turboloader import DataLoader as _CppDataLoader
except ImportError:
    raise ImportError("TurboLoader C++ extension not found. "
                      "Make sure TurboLoader is properly installed.")


class TensorFlowDataLoader:
    """
    TensorFlow-compatible DataLoader using TurboLoader backend.

    Drop-in replacement for tf.data.Dataset with TurboLoader performance.

    Example:
        >>> loader = TensorFlowDataLoader('imagenet.tar', batch_size=32, num_workers=8)
        >>> dataset = loader.as_dataset()
        >>> model.fit(dataset, epochs=10)
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
        prefetch: int = 2,
    ):
        """
        Initialize TensorFlow DataLoader.

        Args:
            data_path: Path to data (TAR, video, CSV, Parquet)
            batch_size: Samples per batch
            num_workers: Worker threads
            shuffle: Shuffle samples
            transform: Optional transform function (numpy -> numpy)
            prefetch: Number of batches to prefetch
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform
        self.prefetch = prefetch

        # Create C++ backend loader
        self._cpp_loader = _CppDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

    def _generator(self):
        """Generator function for tf.data.Dataset.from_generator"""
        while not self._cpp_loader.is_finished():
            batch = self._cpp_loader.next_batch()

            if not batch:
                break

            # Convert batch to tensors
            images = []
            labels = []  # Extract from filenames if available

            for sample in batch:
                img = sample['image']  # NumPy array (H, W, C)

                # Apply transform if provided
                if self.transform is not None:
                    img = self.transform(img)

                images.append(img)

                # Extract label from filename (e.g., "class_0001.jpg" -> 0)
                filename = sample.get('filename', '')
                # Simple label extraction - customize as needed
                label = 0  # Default label
                labels.append(label)

            if images:
                images_batch = np.stack(images, axis=0)
                labels_batch = np.array(labels, dtype=np.int64)

                yield images_batch, labels_batch

    def as_dataset(self) -> tf.data.Dataset:
        """
        Convert to tf.data.Dataset.

        Returns:
            tf.data.Dataset with (images, labels) tuples
        """
        # Determine output signature
        # Assumes RGB images - adjust as needed
        output_signature = (
            tf.TensorSpec(shape=(None, None, None, 3), dtype=tf.uint8),  # (B, H, W, C)
            tf.TensorSpec(shape=(None,), dtype=tf.int64),  # (B,)
        )

        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_signature=output_signature
        )

        # Add prefetching for performance
        if self.prefetch > 0:
            dataset = dataset.prefetch(self.prefetch)

        return dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cpp_loader.stop()


class KerasSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for training with model.fit().

    Example:
        >>> sequence = KerasSequence('data.tar', batch_size=32, num_workers=8)
        >>> model.fit(sequence, epochs=10)
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = False,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize Keras Sequence.

        Args:
            data_path: Path to data
            batch_size: Samples per batch
            num_workers: Worker threads
            shuffle: Shuffle samples
            transform: Optional transform function
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.transform = transform

        self._cpp_loader = _CppDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )

        # Give pipeline threads time to start processing
        import time
        time.sleep(0.2)

        # Pre-load and cache all batches to determine dataset size
        # This is necessary because Keras Sequence requires __len__()
        self._cached_batches = []
        while not self._cpp_loader.is_finished():
            batch = self._cpp_loader.next_batch()
            if batch:
                self._cached_batches.append(batch)
            else:
                break

        self._num_batches = len(self._cached_batches)
        self._current_index = 0

    def __len__(self):
        """Number of batches per epoch"""
        return self._num_batches

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch at index"""
        if idx >= len(self._cached_batches):
            raise IndexError(f"Index {idx} out of range for {len(self._cached_batches)} batches")

        batch = self._cached_batches[idx]

        # Convert to numpy arrays
        images = []
        labels = []

        for sample in batch:
            img = sample['image']
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(0)  # Default label

        images_batch = np.stack(images, axis=0).astype(np.float32) / 255.0
        labels_batch = np.array(labels, dtype=np.int64)

        return images_batch, labels_batch

    def on_epoch_end(self):
        """Called at the end of each epoch"""
        if self.shuffle:
            # Reload with new shuffle
            self._cpp_loader.stop()
            self._cpp_loader = _CppDataLoader(
                data_path=self.data_path,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True
            )

            # Re-cache batches with new shuffle
            self._cached_batches = []
            while not self._cpp_loader.is_finished():
                batch = self._cpp_loader.next_batch()
                if batch:
                    self._cached_batches.append(batch)
                else:
                    break
