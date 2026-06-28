# Pipeline API

DataLoader and pipeline configuration reference.

## DataLoader

PyTorch-compatible data loader with TurboLoader performance. A single entry point for
every modality: images (TAR / WebDataset), LLM token streams, and generic `(N, ...)`
arrays via the `modality=` argument.

### Constructor

```python
turboloader.DataLoader(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    transform=None,
    image_size=None,
    output_format: str = 'dict',
    modality: str = 'image',
    drop_last: bool = False,
    cache_decoded: bool = False,
    antialias: bool = False,
    seed: int = 42,
    # token / array modalities
    seq_len: int = None,
    token_dtype: str = 'uint16',
    arrays=None,
)
```

**Parameters:**
- `data_path` (str): Path to the data source (TAR archive for images, raw token file for
  `modality='tokens'`). Remote schemes (`http://`, `s3://`, `gs://`) are only available
  when the build was compiled with those readers — check `turboloader.features()`.
- `batch_size` (int): Samples per batch (default: 32)
- `num_workers` (int): Worker threads (default: 4). The fast batch-assembly path uses one
  process-wide C++ thread pool; it does not require multiple Python worker processes.
- `shuffle` (bool): Shuffle samples within each worker's shard (default: False)
- `transform`: Transform or composed transforms to apply (e.g.
  `Resize(224, 224) | ImageNetNormalize()`)
- `image_size` (tuple|int): Output `(H, W)` (or square `N`). Required by the fast
  `output_format` paths when it cannot be inferred from the transform.
- `output_format` (str): `'dict'` (per-sample dicts, default), `'numpy'`, `'numpy_chw'`,
  `'pytorch'` (CHW), or `'tensorflow'` (HWC). The non-`'dict'` formats use the optimized
  one-pass batch-assembly path (decode → resize → normalize straight into the output batch
  buffer).
- `modality` (str): `'image'` (default), `'tokens'`, or `'array'`
- `drop_last` (bool): Drop the final incomplete batch (default: False)
- `cache_decoded` (bool): Cache decoded arrays after the first pass for fast subsequent
  epochs (default: False)
- `antialias` (bool): Apply antialiasing on downscale to match PIL/PyTorch/TF (default: False)
- `seed` (int): Seed for reproducible shuffling (default: 42)
- `seq_len` (int): Sequence length, required for `modality='tokens'`
- `token_dtype` (str): On-disk token dtype, e.g. `'uint16'` for GPT-2 BPE
- `arrays` (sequence): Aligned `(N, ...)` arrays/memmaps for `modality='array'`

**Returns:**
- DataLoader instance

**Example:**
```python
import turboloader

loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=64,
    num_workers=8,
    output_format='pytorch',   # one-pass decode/resize/normalize into a CHW batch
    image_size=(160, 160),
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

#### `set_epoch(epoch: int) -> None`

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

High-performance loader returning contiguous batch arrays. This is the FFCV / tf.data-style
direct-batch path: one parallel pass over the shard decodes, resizes, and normalizes each
image straight into the output batch buffer (no intermediate per-sample Python objects).
Large JPEGs use libjpeg-turbo's DCT scaled decode automatically, so downscaling work is
done during decode rather than after. The same fast path is reachable from `DataLoader` by
choosing a non-`'dict'` `output_format`.

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

### Caching for Multi-Epoch Training

The `cache_decoded=True` parameter enables decoded-array caching for fast subsequent epochs:

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
        # First epoch: decode from TAR
        # Subsequent epochs: served from the decoded cache
        train_step(images)

    if loader.cache_populated:
        print(f"Cache size: {loader.cache_size_mb:.1f} MB")

# Clear cache when done
loader.clear_cache()
```

**Performance** (measured on Apple Silicon over Imagenette-160 = 9,469 real ImageNet JPEGs
resized to 160px, `output_format='pytorch'`, batch 64, with real consumption forcing
materialization; warmup + median of 3 epochs):
- On-the-fly (first pass, decode from TAR): ~39,100 img/s
- Cached epochs (`cache_decoded=True`): ~65,499 img/s

Caching trades memory for speed: it keeps the decoded arrays resident, so subsequent epochs
skip JPEG decode entirely. Size the cache to your dataset and RAM.

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

## Multi-Modality

`DataLoader` is not image-only. The `modality=` argument routes to a dedicated loader for
each data type (the underlying `TokenDataLoader` / `ArrayDataLoader` classes live in
`turboloader.sequence` and can also be used directly).

### LLM token streams (`modality='tokens'`)

Streams next-token training batches from a flat token file. The file is memory-mapped, so
multi-GB corpora stream without loading into RAM. Yields `(inputs, targets)` int64 batches
of shape `(batch_size, seq_len)`, where `targets` is `inputs` shifted by one (standard
causal-LM objective).

```python
import turboloader

loader = turboloader.DataLoader(
    'train.bin',
    modality='tokens',
    seq_len=1024,
    batch_size=8,
    token_dtype='uint16',   # e.g. GPT-2 BPE
    shuffle=True,
)

for x, y in loader:          # x, y: (8, 1024) int64
    logits = model(x)
    loss = loss_fn(logits, y)

# Equivalent direct use:
from turboloader.sequence import TokenDataLoader
dl = TokenDataLoader('train.bin', seq_len=1024, batch_size=8)
```

**Throughput:** TurboLoader's token path sustains ~441M tokens/s versus ~163M tokens/s for
the plain NumPy memmap idiom — about 2.7x — measured on the same machine.

### Generic arrays (`modality='array'`)

Batches one or more aligned `(N, ...)` arrays or memmaps — embeddings, tabular features,
pre-tokenized sequences, labels, etc. Returns a single array for one input, otherwise a
tuple (like `TensorDataset` + `DataLoader`).

```python
import turboloader

loader = turboloader.DataLoader(
    None,
    modality='array',
    arrays=[features, labels],
    batch_size=256,
    shuffle=True,
    drop_last=True,
)

for xb, yb in loader:
    train_step(xb, yb)

# Equivalent direct use:
from turboloader.sequence import ArrayDataLoader
dl = ArrayDataLoader(features, labels, batch_size=256, shuffle=True)
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
