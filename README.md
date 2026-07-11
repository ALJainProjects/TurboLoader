# TurboLoader

**Production-Ready ML Data Loading Library**

[![PyPI version](https://img.shields.io/pypi/v/turboloader.svg)](https://pypi.org/project/turboloader/)
[![Tests](https://github.com/ALJainProjects/TurboLoader/actions/workflows/test.yml/badge.svg)](https://github.com/ALJainProjects/TurboLoader/actions/workflows/test.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

TurboLoader is a high-performance data loading library for machine learning workflows. Built with C++20 and featuring Python bindings, it provides efficient data loading with SIMD-accelerated transforms, custom binary formats, and distributed training support.

### Core Features

- **Fused train pipeline** - `train_aug=True`: RandomResizedCrop + flip + normalize inside the C++ pass (torchvision-parity, deterministic per epoch, ~3% over plain loading)
- **Trains end-to-end faster** - real ResNet-18/Imagenette: 1.17x vs PyTorch DataLoader (loader ~fully hidden behind the GPU); see `benchmarks/E2E_TRAINING_RESULTS.md`
- **Checkpointable** - `state_dict()/load_state_dict()`: exact, decode-free mid-epoch resumption
- **Pinned recycled buffers** - `pin_memory=True` yields torch tensors from a reused ring; async H2D with `non_blocking=True`
- **Decoded Tensor Caching** - `FastDataLoader(..., cache_decoded=True)` keeps decoded arrays in RAM so later epochs skip decoding
- **Multiple Loader Types** - FastDataLoader, MemoryEfficientDataLoader, standard DataLoader
- **Distributed Training Support** - Multi-node data loading with deterministic sharding
- **SIMD-Accelerated Transforms** - 19 vectorized transforms using AVX2/AVX-512/NEON
- **TBL v2 Binary Format** - Custom format with LZ4 compression for reduced storage
- **Framework-ready outputs** - `output_format='pytorch'/'numpy'/'tensorflow'` batch layouts, zero-copy torch adoption for the GPU loaders, and a shipped `WebDatasetLoader`
- **Memory-Mapped I/O** - Zero-copy file access for improved throughput
- **Lock-Free Queues** - Concurrent data structures for efficient multi-threading
- **GPU image loaders** - `CudaImageLoader` (NVIDIA nvImageCodec — **beats DALI** on an RTX 3090, see below) and `GpuImageLoader` (Apple Metal): end-to-end GPU decode + resize + normalize, GPU-resident output. See [GPU acceleration](docs/GPU_ACCELERATION.md)
- **Resident (pre-processed) epochs** - `CudaResidentLoader` (**~280k img/s**, beats FFCV 3.5×) and `MetalResidentLoader` (**433–757k img/s** on unified memory) + `MetalResidentArrays` for any-dtype rows. Decode once, serve every epoch with one fused gather+shuffle+normalize kernel launch per batch
- **Video loaders** - `MetalVideoLoader` (VideoToolbox **hardware** decode, **3.9× the best industry standard** on an M4 Max) and `CudaVideoLoader` (GPU-resident batches, dual CPU/NVDEC decode backends, novel fused clip-assembly kernel via `iter_clips`). See [video results](benchmarks/VIDEO_RESULTS.md)

---

## Which loader do I use?

One decision table for every entry point — pick by data type and hardware,
without reading the internals:

| You have | Use | Notes |
|---|---|---|
| A TAR of JPEGs, training on any hardware | **`DataLoader(..., output_format='pytorch', image_size=N)`** | The default fast path — auto-fused C++ decode+resize+normalize. Start here. |
| The same, need per-sample dicts (inspection, irregular data) | `DataLoader(...)` (default `output_format='dict'`) | Several times slower; not for training loops. |
| Labels | derive from `meta['indices']` / `sample['filename']` | Samples carry **no** `label` key; align an external label array by index. |
| A dataset that fits in GPU/unified memory, many epochs | `CudaResidentLoader` (NVIDIA) / `MetalResidentLoader` (Apple) | Decode once, ~280k / 433–757k img/s per epoch. `return_indices=True` for labels. |
| A pre-processed dataset larger than VRAM (NVIDIA) | `CudaStreamLoader` | Fully-C++ streaming, ~140k img/s. |
| On-the-fly GPU decode (NVIDIA) | `CudaImageLoader(decode='nvimgcodec', return_indices=True)` | Beats DALI; batches complete OUT of order — align labels via the returned indices. |
| On-the-fly GPU transforms (Apple) | `MetalImageLoader` (alias of `GpuImageLoader`) | Metal decode+transforms. |
| Video files | `MetalVideoLoader` (Apple) / `CudaVideoLoader` (NVIDIA) | Hardware decode → training batches; `iter_clips()` for augmented clips. |
| LLM token streams (memmap) | `TokenDataLoader` | CPU memmap is already optimal (measured). |
| Arrays / embeddings / tabular | `ArrayDataLoader`; `MetalResidentArrays` for GPU row gathers | |
| WebDataset-style TARs | `WebDatasetLoader` | |

Two lifetime rules to know: (1) loaders yielding **zero-copy views** (`pin_memory=True`
ring, Metal/CUDA resident + video loaders) reuse their buffers — consume or copy a
batch before advancing past the documented window; (2) GPU loaders yield
`__cuda_array_interface__` objects — adopt with `torch.as_tensor(x, device='cuda')`.

## Installation

### From PyPI (Recommended)

```bash
pip install turboloader
```

### From Source

```bash
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader
pip install -e .
```

### System Requirements

- **Python:** 3.10 or higher
- **Compiler:** C++20 capable (GCC 10+, Clang 12+, MSVC 19.29+)
- **OS:** Linux (x86_64/aarch64 wheels), macOS (arm64 wheel). Windows: not officially supported yet — use WSL2

#### Optional Dependencies

Install for enhanced performance:

```bash
# macOS
brew install jpeg-turbo libpng libwebp lz4

# Ubuntu/Debian
sudo apt-get install libjpeg-turbo8-dev libpng-dev libwebp-dev liblz4-dev
```

---

## Quick Start

### Training input (the fast path — start here)

```python
import turboloader

loader = turboloader.DataLoader(
    'imagenet.tar',                 # TAR archive of JPEGs
    batch_size=128,
    image_size=224,                 # fixed size => one contiguous tensor per batch
    output_format='pytorch',        # (N, 3, H, W) float32, normalized
    transform=turboloader.ImageNetNormalize(),
    shuffle=True,
    train_aug=True,                 # fused RandomResizedCrop + flip in C++
)
for images, meta in loader:
    # images: numpy (N,3,224,224); torch.from_numpy(images) is zero-copy.
    # meta['indices'] aligns external labels to this batch.
    ...
```

This is the path all the benchmark numbers refer to. The dict API below is the
flexible per-sample path — several times slower; use it for inspection, not epochs.

### Basic Usage (per-sample dicts)

```python
import turboloader

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet.tar',
    batch_size=128,
    num_workers=8
)

# Iterate over batches. Each sample is a dict:
#   {'image': np.ndarray (H, W, C), 'filename': str, 'index': int,
#    'width': int, 'height': int, 'channels': int}
for batch in loader:
    for sample in batch:
        image = sample['image']      # NumPy array (H, W, C)
        name = sample['filename']    # source path within the archive
        # Train your model...
```

> **Need (image, label) tuples like `torch.utils.data.DataLoader`?** Use
> `PyTorchCompatibleLoader`, which derives labels from the folder structure
> (ImageFolder-style). The base `DataLoader` does not attach labels.

### With Transforms

```python
import turboloader

# Create transforms
resize = turboloader.Resize(224, 224)
normalize = turboloader.ImageNetNormalize()
flip = turboloader.RandomHorizontalFlip(p=0.5)

# Apply transforms
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

for batch in loader:
    for sample in batch:
        img = sample['image']
        img = resize.apply(img)
        img = flip.apply(img)
        img = normalize.apply(img)
        # Ready for training
```

### PyTorch Integration

```python
import turboloader
import torch

loader = turboloader.DataLoader('imagenet.tar', batch_size=64, num_workers=8)

# Convert to PyTorch tensors
to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW
)

for batch in loader:
    images = []
    for sample in batch:
        img = to_tensor.apply(sample['image'])
        images.append(torch.from_numpy(img))

    batch_tensor = torch.stack(images)
    # Train model...
```

### Distributed Training

```python
import turboloader
import torch.distributed as dist

# Initialize distributed training
dist.init_process_group(backend='nccl')

# Create loader with distributed support
loader = turboloader.DataLoader(
    data_path="/data/imagenet.tar",
    batch_size=64,
    num_workers=4,
    shuffle=True,
    enable_distributed=True,
    world_rank=dist.get_rank(),
    world_size=dist.get_world_size(),
    drop_last=True
)

# Each rank automatically gets its shard
for batch in loader:
    # Your training code
    pass
```

---

## Transform Library

TurboLoader includes 24 transforms (19 per-image SIMD transforms + 5 batch
augmentations). The authoritative list is `turboloader.list_transforms()`.

### Core Transforms
- **Resize** - Bilinear/Bicubic/Lanczos interpolation
- **Normalize** - Mean/std normalization with SIMD
- **CenterCrop** - Center region extraction
- **RandomCrop** - Random crop with padding

### Augmentation Transforms
- **RandomHorizontalFlip** - SIMD horizontal flip
- **RandomVerticalFlip** - SIMD vertical flip
- **ColorJitter** - Brightness/contrast/saturation/hue
- **RandomRotation** - Arbitrary angle rotation
- **GaussianBlur** - Separable convolution
- **RandomErasing** - Cutout augmentation
- **Pad** - Border padding (CONSTANT/EDGE/REFLECT)

### Advanced Transforms
- **RandomPosterize** - Bit-depth reduction
- **RandomSolarize** - Threshold inversion
- **RandomPerspective** - Perspective warp
- **AutoAugment** - Learned policies (ImageNet/CIFAR10/SVHN)

### Batch Augmentations
- **MixUp**, **CutMix**, **Mosaic**, **RandAugment**, **GridMask**

### Tensor Conversion
- **ToTensor** - PyTorch CHW or TensorFlow HWC format

---

## TBL v2 Binary Format

TurboLoader includes a custom binary format optimized for ML workloads:

### Features
- LZ4 compression for reduced storage
- Memory-mapped access for fast loading
- O(1) random access via indexed structure
- Data integrity validation with CRC checksums
- Cached image dimensions for filtered loading

### Convert TAR to TBL

```python
import tarfile
import turboloader

writer = turboloader.TblWriterV2("/data/imagenet.tbl", enable_compression=True)

# The TAR archive is read with Python's stdlib (TurboLoader does not expose a
# standalone Python TarReader; the DataLoader reads TAR directly for training).
with tarfile.open("/data/imagenet.tar") as tar:
    for member in tar.getmembers():
        if not member.name.lower().endswith((".jpg", ".jpeg")):
            continue
        data = tar.extractfile(member).read()
        writer.add_sample(data=data, format=turboloader.SampleFormat.JPEG)

writer.finalize()
```

> For bulk conversion there is also a C++ CLI tool, `tools/tar_to_tbl_v2.cpp`.

---

## Documentation

### Getting Started
- **[Quick Start Notebook](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/quickstart.ipynb)** - Interactive tutorial for beginners
- **[Installation Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/installation.md)** - Detailed setup instructions
- **[Quick Start](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/quickstart.md)** - Getting started examples
- **[Troubleshooting Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions

### API Documentation
- **[API Reference](https://github.com/ALJainProjects/TurboLoader/tree/main/docs/api)** - Complete API documentation
- **[Transforms API](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/api/transforms.md)** - All 19 transforms with examples

### Framework Integration
- **[PyTorch Integration Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/guides/pytorch-integration.md)** - Complete PyTorch guide
- **[TensorFlow Integration Guide](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/guides/tensorflow-integration.md)** - Complete TensorFlow/Keras guide
- **[PyTorch Lightning Example](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/pytorch_lightning_example.py)** - Production-ready Lightning integration
- **[Distributed Training (DDP)](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/distributed_ddp.py)** - Multi-GPU PyTorch DDP example

### Examples
- **[ImageNet ResNet50 Training](https://github.com/ALJainProjects/TurboLoader/blob/main/examples/imagenet_resnet50.py)** - Complete training pipeline with AMP, checkpointing, TensorBoard
- **[Distributed Training](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/distributed.md)** - Multi-node setup guide

---

## Benchmarks

Measured on **Apple Silicon** over **Imagenette-160** (9,469 real ImageNet JPEGs →
resize 160×160 → ImageNet-normalize → batched CHW float32, batch 64). To control for
thermal throttling, every loader is built once, warmed up one epoch, then timed over
**5 interleaved rounds** (each loader runs once per round); the table reports the
median. Output is verified correct against torchvision (mean abs diff ≈ 0.04, bilinear
antialiasing only).

**Image — on-the-fly decode** (re-decode every epoch; for datasets too large to cache
or with per-epoch random augmentation):

| Loader | img/s (median) | vs tf.data |
|---|---:|---:|
| **TurboLoader `DataLoader`** (`output_format='pytorch'`, nw=6) | **~55,000** | **2.0×** |
| TensorFlow `tf.data` (AUTOTUNE) | ~27,300 | 1.00× |
| PyTorch `DataLoader` (PIL, 8 persistent workers) | ~20,500 | 0.75× |

**Image — cached** (decoded tensors held in RAM; both sides consume identically via
`np.sum`, i.e. delivered as numpy/torch-ready batches — the PyTorch use case):

| Loader | img/s (median) | vs tf.data.cache |
|---|---:|---:|
| **TurboLoader** (`cache_decoded=True`, prefetch) | **~67,000** | **1.9×** |
| TensorFlow `tf.data.cache()` (+ `.numpy()` materialize) | ~35,100 | 1.00× |

(For *TF-native* consumption that stays in tf tensors, `tf.data.cache()` is faster —
TurboLoader's cache win is for delivering numpy/torch batches.)

**LLM tokens** (real text, 55M-token memory-mapped corpus, `seq_len=1024`, next-token):

| Loader | sequences/s (median) |
|---|---:|
| **TurboLoader `TokenDataLoader`** | **~467,000** |
| numpy memmap idiom (nanoGPT `get_batch`) | ~251,000 |

**Transforms** (per-image throughput vs torchvision): Resize **2.7×**, ImageNetNormalize
**3.3×**, HFlip ~1.0×. For CenterCrop, torchvision returns a **lazy strided view** (moves
zero bytes); compared against TurboLoader's real contiguous crop that looks like 0.45×,
but when torchvision actually materializes the crop (`.contiguous()`, required before
batching/most ops) it drops to ~23k img/s and **TurboLoader's contiguous crop is ~6.8×
faster** (155k vs 23k). Like the cache, this is a lazy-vs-eager comparison; for the
realistic crop→batch path TurboLoader wins.

> Earlier drafts quoted single-run figures (~42k, "1.4×") and a "cached epoch" in the
> tens-of-millions img/s. Those were artifacts (thermal noise; a no-op loop over aliased
> cached arrays) and were replaced with the interleaved, identical-consumption medians
> above. Numbers are hardware-dependent — run `benchmarks/` yourself.

The fast path runs decode + resize + normalize + batch assembly in C++ across a thread
pool with zero Python per-sample work. Use it like this:

```python
loader = turboloader.DataLoader(
    'imagenet.tar', batch_size=64, num_workers=6,
    output_format='pytorch',          # (N, C, H, W) float32 array per batch
    image_size=160,                   # exact resize, done in C++
    transform=turboloader.ImageNetNormalize())
for epoch in range(epochs):           # re-iterable
    for images, meta in loader:       # images.shape == (64, 3, 160, 160)
        train_step(images)
```

Honest caveats:
- **Run it yourself** (`benchmarks/`) — results depend heavily on hardware, image size,
  and pipeline; Linux `fork`-based PyTorch workers shift the PyTorch numbers a lot.
- **Decode backend differs**: TurboLoader uses libjpeg-turbo; the PyTorch baseline uses PIL.
- The `output_format='dict'` path returns per-sample dicts and stacks in Python
  (GIL-bound), so it is much slower — use it only when you need per-sample metadata.

For **large source images**, the default path also wins: on 768×768 JPEGs resized to
160 it runs ~15,000 img/s — faster than even an expertly-tuned `tf.data` pipeline using
manual `decode_jpeg(ratio=...)` (~14,400) — because it picks the libjpeg-turbo DCT
scaled-decode factor automatically (you don't have to know to set `ratio`).

### GPU loaders (NVIDIA & Apple)

On **NVIDIA**, `CudaImageLoader(decode="nvimgcodec")` runs the whole decode + resize + normalize
+ batch in GIL-released C++ via **nvImageCodec** (the codec DALI uses), with K independent decode
slots overlapping batches (multi-batch-in-flight). Among **on-the-fly** loaders (read a JPEG
folder, decode+resize every epoch) on an **RTX 3090** (Imagenette-160, batch 64, real consumption,
interleaved rounds to control for ~40% host drift):

| On-the-fly loader | vs TurboLoader |
|---|---:|
| **TurboLoader** `decode="nvimgcodec"`, `nvimgcodec_slots=3` | **1.0× (fastest)** |
| NVIDIA **DALI** (`num_threads=8`, best-tuned) | ~0.9× (TurboLoader **+12%** cleanest run) |
| PyTorch `DataLoader` (PIL, CPU) | ~0.25× |

**TurboLoader beats DALI** (median above DALI's max in the cleanest run), output bijectively
verified correct. For **on-the-fly** loading FFCV is faster (~2.6–5.9×) — but it requires an
offline conversion to its `.beton` format.

**Pre-processed loaders** (decode+resize once, like FFCV's `.beton`) — here TurboLoader turns the
tables:

| Pre-processed loader | img/s | |
|---|---:|---|
| **TurboLoader `MetalResidentLoader`** (Apple M4 Max, unified memory: no H2D exists) | **~757,000 produced / ~433,000 consumed** | ships in the pip wheel |
| **TurboLoader `CudaResidentLoader`** (fits-in-VRAM: upload uint8 once, GPU-resident) | **~280,000** | **beats FFCV ~3.5×** |
| **TurboLoader `CudaStreamLoader`** (streaming, dataset > VRAM; fully-C++ loop) | **~140,000** | **beats FFCV ~1.6×** |
| FFCV, raw `.beton` (streams mmap→H2D each epoch, worker processes) | ~85,000 | |

On **Apple Silicon** the resident trick is even better than on NVIDIA: memory is unified, so
"upload" is one memcpy and every GPU-written batch is a **zero-copy numpy view**.
`MetalResidentLoader` serves each epoch as one fused gather+shuffle+normalize kernel launch per
batch; `MetalResidentArrays` does the same for any-dtype rows (embedding tables: ~5× numpy
fancy-indexing). Honest null result included: `MetalTokenGather` ties the CPU memmap path
(0.87–1.08×) — keep using `TokenDataLoader` for tokens.

**Video**: `MetalVideoLoader` (macOS arm64, in the pip wheel — no FFmpeg needed) drives
VideoToolbox **hardware** H.264/HEVC decode into a fused NV12→RGB+resize+normalize Metal
kernel: real 1080p → 224px training batches at **~2,550 frames/s** on an M4 Max —
**3.9× the best industry standard** (OpenCV 657, PyAV 535, torchcodec 173) and 97–99% of
the media engine's hardware decode ceiling. On NVIDIA, `CudaVideoLoader` (CUDA build)
lands GPU-resident batches via a dual decode backend (threaded CPU decode by default;
NVDEC opt-in — measured virtualization-throttled under WSL2) plus a novel **fused
clip-assembly kernel** (`iter_clips`: consistent RandomResizedCrop+flip across a whole
clip + YUV→RGB + resize + normalize in ONE launch). Honest scorecard incl. where decord
still wins on weak-CPU hosts: [benchmarks/VIDEO_RESULTS.md](benchmarks/VIDEO_RESULTS.md).

`CudaResidentLoader` uses a custom single-launch normalize kernel + fused gather (shuffles at
~257k) and **beats FFCV ~3.5×** when the pre-processed uint8 dataset fits in VRAM (very common:
fine-tuning, per-GPU shards, small/medium sets). For datasets **larger than VRAM**,
`CudaStreamLoader` runs the whole iteration GIL-free in C++ (`CudaStreamCore`: worker pool + async
H2D on non-blocking streams + prefetch) and **beats FFCV's streaming ~1.6×** (~140k vs ~85k, near
the PCIe transfer ceiling). So TurboLoader beats **DALI** on-the-fly and **FFCV** on pre-processed
data — both fits-in-VRAM and streaming. On **Apple Silicon**, `GpuImageLoader` offloads
resize+normalize (and a hybrid GPU JPEG decode) to Metal — where neither DALI nor FFCV runs at
all. CUDA is a build-from-source path (not
in the PyPI wheels); see [GPU acceleration](docs/GPU_ACCELERATION.md) for flags, usage, and the
full write-up (`experiments/cuda/RESULTS.md`).

### Implementation notes
- **Direct-batch path** (`src/pipeline/direct_batch_loader.hpp`): the default fast path
  is FFCV/`tf.data`-style — a persistent thread pool reads JPEG bytes by index and
  decodes → resizes → normalizes **directly into the output batch buffer** in one
  parallel pass (no worker queue, no per-sample heap copy, no serial collection).
  Verified memory-safe and race-free (disjoint slot writes, const mmap reads, atomic
  cursor, per-thread decoders).
- **Automatic DCT scaled decode**: large JPEGs are decoded at the nearest libjpeg-turbo
  scale ≥ target, then finely resized — much faster than full-decode + resize.
- **Resize convention**: half-pixel centers (`align_corners=False`), matching
  PIL/OpenCV/PyTorch/TF (agrees with torchvision plain bilinear to ~0.4/255; the only
  remaining difference vs torchvision's default is its antialiasing low-pass filter).
- SIMD transforms (AVX2/AVX-512/NEON), libjpeg-turbo decode, lock-free SPSC queues
  (legacy/dict + remote path), persistent `std::thread` pool (`src/core/parallel_for.hpp`).
- The GIL is released during C++ processing.
- **OpenMP is opt-in** (`TURBOLOADER_ENABLE_OPENMP=1`); off by default because linking a
  second OpenMP runtime crashes alongside PyTorch on macOS — the thread pool replaces it.

---

## Beyond Images: Tokens & Arrays

TurboLoader also ships loaders for non-image modalities with the same ergonomics
(re-iterable, `shuffle`, `set_epoch`, batched arrays):

```python
# LLM pretraining: memory-mapped token stream -> (B, seq_len) next-token batches
loader = turboloader.TokenDataLoader('train.bin', seq_len=1024, batch_size=8,
                                     dtype='uint16', shuffle=True)
for x, y in loader:          # x, y: (8, 1024) int64; y is x shifted by one
    loss = model(x, y)

# Generic arrays/memmaps (embeddings, tabular features, labels, pre-tokenized data)
loader = turboloader.ArrayDataLoader(features, labels, batch_size=256, shuffle=True)
for xb, yb in loader:
    ...
```

`TokenDataLoader` uses a vectorized fancy-index gather over a `np.memmap` (so multi-GB
corpora stream without loading into RAM) and benchmarks ~1.9× the standard nanoGPT
`get_batch` idiom. The image pipeline (decode/transform/TBL) remains C++; these
modality loaders are NumPy-based and modality-agnostic.

All three modalities are also reachable from the **single `DataLoader` entry point**:

```python
turboloader.DataLoader('train.bin', modality='tokens', seq_len=1024, batch_size=8)
turboloader.DataLoader(arrays=[feats, labels], data_path=None, modality='array', batch_size=256)
turboloader.DataLoader('data.tar', image_size=160, output_format='pytorch')   # modality='image' (default)
```

### Wrap *any* Python dataset (`MapDataLoader`)

When your data doesn't fit the native paths, `MapDataLoader` batches **any** map-style
dataset — anything with `__len__` and `__getitem__(i)`, i.e. exactly the
`torch.utils.data.Dataset` protocol — so your loading/decoding/business logic can be
arbitrary Python:

```python
class MyDataset:
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        x = decode_however_you_like(self.records[i])   # any Python logic
        return x, self.labels[i]                       # (features, label)

# directly, or via the unified entry point with dataset=...
for xb, yb in turboloader.MapDataLoader(MyDataset(), batch_size=64, shuffle=True, num_workers=8):
    train_step(xb, yb)
```

It parallelizes `__getitem__` on a bounded thread pool with read-ahead and collates
(tuples/dicts/arrays, or a custom `collate_fn`). **Honest tradeoff:** because the
per-sample work runs in Python, this path is roughly PyTorch-`DataLoader` speed (and
GIL-bound for pure-Python CPU work — threads help most when `__getitem__` releases the
GIL, e.g. NumPy/PIL/file/network I/O). It's about *flexibility*, not the C++ fast path —
use the image/token/array loaders above when you want maximum throughput.

---

## Architecture

TurboLoader uses a multi-threaded pipeline architecture:

```
┌─────────────────────────────────────────────┐
│           Memory-Mapped Reader              │
│     (TAR/TBL v2 with zero-copy access)      │
└──────────────┬──────────────────────────────┘
               │
        ┌──────▼──────┐
        │Worker Pool  │
        │  (N threads)│
        ├─────────────┤
        │ Decode      │
        │ Transform   │
        │ Convert     │
        └──────┬──────┘
               │
        ┌──────▼──────────────┐
        │ Lock-Free Queue     │
        └──────┬──────────────┘
               │
        ┌──────▼──────┐
        │Python API   │
        └─────────────┘
```

### Key Components

- **Memory-Mapped I/O** - Zero-copy file access
- **Worker Thread Pool** - Parallel processing with per-thread decoders
- **SIMD Transforms** - Vectorized operations (AVX2/AVX-512/NEON)
- **Lock-Free Queues** - High-performance concurrent data structures

---

## License

TurboLoader is released under the MIT License.

---

## Citation

If you use TurboLoader in your research:

```bibtex
@software{turboloader,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance ML Data Loading},
  year = {2026},
  version = {2.25.0},
  url = {https://github.com/ALJainProjects/TurboLoader}
}
```

---

## Support

- **Documentation:** [https://github.com/ALJainProjects/TurboLoader/tree/main/docs](https://github.com/ALJainProjects/TurboLoader/tree/main/docs)
- **Troubleshooting:** [https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md](https://github.com/ALJainProjects/TurboLoader/blob/main/docs/TROUBLESHOOTING.md)
- **Verification Script:** Run `python scripts/verify_installation.py` to check your setup
- **Issues:** [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **PyPI:** [https://pypi.org/project/turboloader/](https://pypi.org/project/turboloader/)

---

TurboLoader - High-performance ML data loading with a C++20 core and SIMD transforms.
