# Pipeline API

DataLoader and pipeline configuration reference.

## DataLoader

PyTorch-compatible data loader with TurboLoader performance.

### Constructor

```python
turboloader.DataLoader(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False
)
```

**Parameters:**
- `data_path` (str): Path to TAR archive (local file, http://, s3://, gs://)
- `batch_size` (int): Samples per batch (default: 32)
- `num_workers` (int): Worker threads (default: 4)
- `shuffle` (bool): Shuffle samples within each worker's shard (default: False)

**Returns:**
- DataLoader instance

**Example:**
```python
import turboloader

loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=128,
    num_workers=8
)
```

### Methods

#### `next_batch() -> list`

Get next batch of samples.

**Returns:**
- `list[dict]`: Batch of samples with keys:
  - `index` (int): Sample index
  - `filename` (str): Original filename
  - `width` (int): Image width
  - `height` (int): Image height
  - `channels` (int): Number of channels
  - `image` (np.ndarray): Image data (H, W, C) uint8

#### `is_finished() -> bool`

Check if all data has been processed.

#### `stop() -> None`

Stop the pipeline and clean up resources.

#### `set_epoch(epoch: int) -> None` (v2.8.0)

Set epoch for reproducible shuffling.

When `shuffle=True`, call this at the start of each epoch to get reproducible shuffling. Different epochs produce different orderings, but the same epoch + seed = same ordering across runs.

**Parameters:**
- `epoch` (int): The epoch number (0, 1, 2, ...)

**Example:**
```python
loader = turboloader.DataLoader('data.tar', shuffle=True)

for epoch in range(10):
    loader.set_epoch(epoch)  # Ensure reproducible shuffling
    for batch in loader:
        train(batch)
```

**Note:** Shuffle uses intra-worker shuffling (each worker shuffles its own shard), matching PyTorch DataLoader's distributed behavior.

### Iterator Protocol

DataLoader supports Python iterator protocol:

```python
loader = turboloader.DataLoader('data.tar', batch_size=32, num_workers=8)

for batch in loader:
    for sample in batch:
        print(sample['filename'])
```

### Context Manager

DataLoader supports context manager protocol:

```python
with turboloader.DataLoader('data.tar', batch_size=32, num_workers=8) as loader:
    for batch in loader:
        # Process batch
        pass
# Automatically stopped and cleaned up
```

---

## FastDataLoader

High-performance loader returning contiguous numpy arrays. 8-12% faster than standard DataLoader due to reduced Python overhead.

### Constructor

```python
turboloader.FastDataLoader(
    source: str,
    batch_size: int = 32,
    num_workers: int = 4,
    output_format: str = 'numpy',
    cache_decoded: bool = False,
    cache_decoded_mb: int = None
)
```

**Parameters:**
- `source` (str): Path to TAR file or data directory
- `batch_size` (int): Images per batch (default: 32)
- `num_workers` (int): Parallel workers (default: 4)
- `output_format` (str): Output format - 'numpy', 'numpy_chw', or 'pytorch' (default: 'numpy')
- `cache_decoded` (bool): Cache decoded arrays for fast subsequent epochs (default: False)
- `cache_decoded_mb` (int): Max cache memory in MB (default: None = unlimited)

**Returns:**
- FastDataLoader instance

**Example:**
```python
import turboloader

loader = turboloader.FastDataLoader(
    'imagenet.tar',
    batch_size=64,
    num_workers=8,
    output_format='numpy'
)

for images, metadata in loader:
    # images is a contiguous numpy array (N, H, W, C)
    print(f"Batch shape: {images.shape}")
```

### Methods

#### `next_batch() -> tuple`

Get next batch as (images_array, metadata_dict).

**Returns:**
- `tuple`: (np.ndarray, dict) where:
  - `images`: Contiguous array (N, H, W, C) or (N, C, H, W) for CHW format
  - `metadata`: Dict with batch information

#### `next_batch_torch(device=None, non_blocking=False, dtype=None) -> tuple`

Get next batch as PyTorch tensors.

**Parameters:**
- `device`: Target device (e.g., 'cuda:0')
- `non_blocking` (bool): Use async transfer
- `dtype`: Target dtype (default: torch.float32)

**Returns:**
- `tuple`: (torch.Tensor, dict)

#### `next_batch_tf(dtype=None) -> tuple`

Get next batch as TensorFlow tensors.

**Parameters:**
- `dtype`: Target dtype (default: tf.float32)

**Returns:**
- `tuple`: (tf.Tensor, dict)

#### `clear_cache() -> None`

Clear the decoded tensor cache (if cache_decoded=True).

### Properties

- `cache_populated` (bool): Whether cache is filled
- `cache_size_mb` (float): Current cache memory usage in MB

### Iterator Protocol

FastDataLoader supports iteration:

```python
for images, metadata in loader:
    # images: np.ndarray (N, H, W, C)
    # metadata: dict with batch info
    pass
```

### Caching for Multi-Epoch Training (v2.7.0)

The `cache_decoded=True` parameter enables decoded tensor caching for fast subsequent epochs:

```python
import turboloader

loader = turboloader.FastDataLoader(
    'imagenet.tar',
    batch_size=64,
    num_workers=8,
    cache_decoded=True  # Enable caching
)

for epoch in range(10):
    for images, metadata in loader:
        # First epoch: ~25K img/s (decode from TAR)
        # Subsequent epochs: memory speed (cache hit)
        train_step(images)

    if loader.cache_populated:
        print(f"Cache size: {loader.cache_size_mb:.1f} MB")

# Clear cache when done
loader.clear_cache()
```

**Performance:**
- First epoch: Standard decode throughput (~25K img/s)
- Cached epochs: Memory iteration speed (100K+ img/s)
- Total time for 5 epochs: 2.6x faster than TensorFlow `.cache()`

---

## MemoryEfficientDataLoader

Memory-optimized loader for constrained environments. Auto-tunes workers and prefetch based on memory budget.

### Constructor

```python
turboloader.MemoryEfficientDataLoader(
    source: str,
    batch_size: int = 32,
    max_memory_mb: int = 512
)
```

**Parameters:**
- `source` (str): Path to data
- `batch_size` (int): Images per batch (default: 32)
- `max_memory_mb` (int): Memory budget - auto-tunes workers/prefetch (default: 512)

**Example:**
```python
import turboloader

loader = turboloader.MemoryEfficientDataLoader(
    'data.tar',
    batch_size=32,
    max_memory_mb=512  # Stay under 512MB
)

for batch in loader:
    for sample in batch:
        # Process sample
        pass
```

---

## create_loader()

Factory function to create the appropriate loader type.

### Signature

```python
turboloader.create_loader(
    source: str,
    loader_type: str = 'fast',
    **kwargs
) -> DataLoader | FastDataLoader | MemoryEfficientDataLoader
```

**Parameters:**
- `source` (str): Path to data
- `loader_type` (str): 'fast', 'memory_efficient', or 'standard' (default: 'fast')
- `**kwargs`: Additional parameters passed to underlying loader

**Returns:**
- Appropriate loader instance based on `loader_type`

**Example:**
```python
import turboloader

# For maximum throughput
loader = turboloader.create_loader('data.tar', loader_type='fast')

# For memory-constrained environments
loader = turboloader.create_loader(
    'data.tar',
    loader_type='memory_efficient',
    max_memory_mb=512
)

# Standard DataLoader
loader = turboloader.create_loader('data.tar', loader_type='standard')
```

---

## Loader()

Alias for `create_loader()` with simpler API.

```python
loader = turboloader.Loader('data.tar')  # Uses FastDataLoader by default
```

---

## See Also

- [Getting Started](../getting-started.md)
- [Transforms API](transforms.md)
