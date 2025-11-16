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
- `shuffle` (bool): Shuffle samples (future feature, default: False)

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

## See Also

- [Getting Started](../getting-started.md)
- [Transforms API](transforms.md)
