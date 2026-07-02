# TurboLoader Architecture

This document describes the internal architecture of TurboLoader v2.31.0.

## Overview

TurboLoader's default fast path is an FFCV / tf.data-style **direct-batch loader**: a single process-wide C++ thread pool makes one parallel pass that decodes, resizes, and normalizes images straight into the output batch buffer (with automatic libjpeg-turbo DCT scaled decode for large images). On Apple Silicon over Imagenette-160 (9,469 real ImageNet JPEGs → 160px, `output_format='pytorch'`, batch 64, real consumption, warmup + median of 3 epochs) it reaches **~39,100 img/s on the fly** and **~65,499 img/s with the decoded cache** (`cache_decoded=True`) — about **1.3x** TensorFlow tf.data (AUTOTUNE) and **2.1x** a PyTorch DataLoader with 8 persistent workers.

Beyond images, TurboLoader is **multi-modal**: images in WebDataset TAR, LLM token streams via `TokenDataLoader`, and generic `(N, ...)` arrays via `ArrayDataLoader`. It also provides SIMD-accelerated transforms (NEON / AVX2 / AVX-512), half-pixel resize matching PIL/PyTorch/TF (optional antialiasing), a decoded cache, DDP-safe equal/disjoint sharding, NumPy / PyTorch-CHW / TensorFlow-HWC output, and the TBL v2 binary format with LZ4 compression.

```
┌─────────────────────────────────────────────────────────────┐
│                    TurboLoader Pipeline                      │
└──────────┬──────────────────────────────────────────────────┘
           │
    ┌──────▼──────┐
    │  Main Thread │
    │  Coordinator │
    └──────┬───────┘
           │
    ┌──────▼───────────────────────────────────────────────┐
    │          Memory-Mapped TAR Reader                     │
    │  • mmap() for zero-copy access                        │
    │  • TAR format parsing (512-byte headers)              │
    │  • Sample metadata extraction                         │
    └──────┬───────────────────────────────────────────────┘
           │
    ┌──────▼───────────────────────────────────────────────┐
    │          Worker Thread Pool (N threads)               │
    │                                                        │
    │  ┌────────────────┐  ┌────────────────┐              │
    │  │  Worker 1      │  │  Worker N      │              │
    │  ├────────────────┤  ├────────────────┤              │
    │  │ 1. Decode      │  │ 1. Decode      │              │
    │  │    (libjpeg)   │  │    (libjpeg)   │              │
    │  │ 2. Transform   │  │ 2. Transform   │              │
    │  │    (SIMD ops)  │  │    (SIMD ops)  │              │
    │  │ 3. Convert     │  │ 3. Convert     │              │
    │  │    (to tensor) │  │    (to tensor) │              │
    │  └────────┬───────┘  └────────┬───────┘              │
    └───────────┼──────────────────┼─────────────────────┘
                │                  │
         ┌──────▼──────────────────▼──────┐
         │   Output Batch Buffer           │
         │   (workers write rows directly) │
         └──────┬─────────────────────────┘
                │
         ┌──────▼──────────────┐
         │   Python Iterator   │
         │   (yields filled     │
         │    batch tensor)     │
         └─────────────────────┘
```

> **Default fast path:** the worker pool writes decoded / resized / normalized rows
> straight into the output batch buffer (FFCV / tf.data-style direct-batch loading).
> The lock-free SPSC queue shown above is the **legacy / remote-streaming** path, not
> the default (see section 4).

## Core Components

### 1. Memory-Mapped Reader (TAR/TBL v2)

**Purpose:** Zero-copy file access for TAR archives and TBL v2 binary format

**TAR Reader:**
- Uses `mmap()` system call to map file into memory
- Parses TAR headers (512-byte chunks) to locate samples
- Builds index of file offsets for random access
- Throughput is bounded by SSD / page-cache bandwidth (zero-copy, no per-read syscall copy)

**TBL v2 Reader **
- Memory-mapped access to compressed binary format
- LZ4 decompression for space savings vs TAR (amount depends on the data)
- CRC32 checksum validation for data integrity
- CRC16 validation for index entries
- Cached image dimensions (width/height) in 16-bit index
- Zero-copy reads with on-demand LZ4 decompression (2.5-3.5 GB/s)
- 24-byte index entries with full metadata

**Performance characteristics:**
- Both readers are mmap-backed and bounded by SSD / page-cache bandwidth
- LZ4 decompression for TBL v2 happens on demand (~2.5–3.5 GB/s single-threaded)
- Random access: O(1) for TBL v2, O(n) for TAR
- Memory footprint: Minimal (mmap on-demand paging)

**Code Locations:**
- TAR: `src/tar/tar_reader.hpp`
- TBL v2: `src/readers/tbl_v2_reader.hpp`
- LZ4: `src/compression/lz4_compressor.hpp`

### 2. JPEG/PNG/WebP Decoders

**Purpose:** Fast image decoding using optimized libraries

**Libraries Used:**
- **libjpeg-turbo:** SIMD-accelerated JPEG decoding
- **libpng:** Optimized PNG decoding
- **libwebp:** WebP format support

**Thread Model:**
- **Thread-local decoder instances** - One decoder per worker thread
- Eliminates mutex contention
- Maximizes CPU cache locality

**Decode notes:**
- libjpeg-turbo SIMD decode, with automatic DCT scaled decode for large images (decode straight to a smaller size before resize)
- PNG/WebP decoded via their respective optimized libraries

**Code Location:** `src/decode/image_decoder.hpp`

### 3. SIMD Transform Pipeline

**Purpose:** Hardware-accelerated image transformations

**SIMD Instruction Sets:**
- **x86_64:** AVX2 (8-wide vectors), AVX-512 (16-wide vectors, Intel Skylake-X+, AMD Zen 4+)
- **ARM:** NEON (Apple Silicon, ARM servers) with graceful fallback from AVX-512

**Transform Categories:**

#### Geometric Transforms
- **Resize:** SIMD-accelerated bilinear/bicubic/Lanczos interpolation
- **Crop:** Fast memory copy operations
- **Rotation/Affine:** SIMD matrix transformations
- **Perspective:** Homography with SIMD interpolation

#### Color Transforms
- **Normalize:** SIMD floating-point operations (`_mm256_fmadd_ps`)
- **ColorJitter:** Vectorized brightness/contrast/saturation
- **Grayscale:** SIMD weighted average

#### Effect Transforms
- **GaussianBlur:** Separable convolution with SIMD
- **Posterize:** Bitwise operations (ultra-fast)
- **Solarize:** SIMD conditional operations
- **RandomErasing:** Fast memory fill

**Performance Example (Resize):**
```cpp
// AVX2 SIMD code for bilinear interpolation
__m256 pixels_0 = _mm256_loadu_ps(src_row0 + x);
__m256 pixels_1 = _mm256_loadu_ps(src_row1 + x);
__m256 interp_y = _mm256_fmadd_ps(pixels_1, fy_vec,
                   _mm256_mul_ps(pixels_0, one_minus_fy));
// 8 pixels interpolated per instruction
```

**Code Location:** `src/transforms/`

### 4. Lock-Free Concurrent Queues (legacy / remote-streaming path)

**Purpose:** Sample passing between threads on the legacy and remote-streaming code paths

> **Note:** Lock-free SPSC queues are **not** the default fast path. The default
> direct-batch loader has the worker pool write decoded / transformed rows straight
> into the output batch buffer (see Overview), so there is no per-sample enqueue or
> dequeue on the hot path. The SPSC ring buffer is retained for the legacy streaming
> pipeline and remote/streaming sources, where producer and consumer run at different
> rates.

**Implementation:**
- **SPSC (Single-Producer Single-Consumer) ring buffer**
- Cache-line aligned (`alignas(64)`)
- Atomic operations for indices
- No mutex locks in the hot path

**Memory Layout:**
```cpp
alignas(64) struct SPSCQueue {
    alignas(64) std::atomic<size_t> write_pos;  // Producer cache line
    alignas(64) std::atomic<size_t> read_pos;   // Consumer cache line
    alignas(64) Sample buffer[CAPACITY];        // Data
};
```

**Code Location:** `src/queue/concurrent_queue.hpp`

### 5. Worker Thread Pool

**Purpose:** Parallel processing of samples

**Thread Count:**
- The default fast path is **one process-wide C++ thread pool** that parallelizes
  internally across all cores — it is already saturated from a single Python "worker"
- Configurable pool size; rule of thumb: roughly one thread per CPU core
- Unlike a PyTorch DataLoader, raising `num_workers` does not spawn extra processes on
  the fast path, so it does not multiply throughput (see Scalability below)

**Worker Lifecycle:**
```
1. Receive sample metadata from main thread
2. Decode image from TAR (mmap'd memory)
3. Apply transform pipeline (SIMD operations)
4. Convert to tensor format (optional)
5. Push to output queue
6. Repeat
```

**Synchronization:**
- Main thread → Workers: work distribution across the pool
- Workers → output: each worker writes its rows directly into the shared output batch
  buffer (no result queue on the fast path)
- No inter-worker communication (embarrassingly parallel)

**Code Location:** `src/pipeline/pipeline.hpp`

### 6. Tensor Conversion

**Purpose:** Zero-copy conversion to PyTorch/TensorFlow formats

**Formats:**
- **PyTorch:** CHW (Channels, Height, Width) float32
- **TensorFlow:** HWC (Height, Width, Channels) float32
- **NumPy:** HWC uint8 (default)

**Memory Layout Transformation:**

```python
# Input: HWC uint8 (NumPy default)
# [H, W, C] → [C, H, W]

# PyTorch CHW:
for c in range(C):
    for h in range(H):
        for w in range(W):
            out[c, h, w] = in[h, w, c] / 255.0  # SIMD vectorized

# TensorFlow HWC:
# Direct memory copy + normalize (faster)
```

**Code Location:** `src/transforms/tensor_conversion.hpp`

## Data Flow

### Sample Processing Pipeline

```
1. TAR Reader Thread
   ├─ mmap TAR file
   ├─ Parse TAR headers
   ├─ Extract sample metadata (offset, size, filename)
   └─ Push to work queue
                ↓
2. Worker Threads (parallel)
   ├─ Pop from work queue
   ├─ Decode JPEG/PNG from mmap'd memory
   ├─ Apply transforms (SIMD)
   │  ├─ Resize (AVX2/NEON)
   │  ├─ RandomFlip
   │  ├─ ColorJitter
   │  ├─ Normalize
   │  └─ ToTensor
   └─ Write row directly into the output batch buffer
                ↓
3. Python Iterator
   ├─ Wait for the batch buffer to fill
   ├─ Wrap it as a NumPy / PyTorch / TensorFlow tensor (zero-copy)
   └─ Return to user
```

### Memory Management

**Zero-Copy Optimizations:**
1. **mmap I/O:** File data never copied to user space
2. **NumPy arrays:** Direct memory view (no copy)
3. **Transform pipeline:** In-place operations where possible

**Memory Pools:**
- **Decode buffers:** Reused per worker thread
- **Transform buffers:** Allocated once, reused
- **Output queue:** Pre-allocated ring buffer

**Peak Memory Usage:**
- Base: **~200 MB** (code + buffers)
- Per worker: **~50 MB** (decode + transform buffers)
- **Total:** ~200 + (50 × num_workers) MB

Example (8 workers): **~600 MB**

## Performance Characteristics

### Throughput Breakdown

Per-stage bottlenecks on the direct-batch fast path:

| Stage | Bottleneck |
|-------|------------|
| mmap TAR / TBL reader | SSD / page-cache bandwidth |
| JPEG decode (libjpeg-turbo) | CPU decode |
| SIMD resize / normalize | Memory bandwidth / SIMD FMA units |
| Direct-batch write | Memory bandwidth |

### Scalability

The default direct-batch fast path runs as **one process-wide C++ thread pool** that is
already saturated from a single Python "worker" — it parallelizes internally across
cores. Increasing `num_workers` on the fast path does **not** spin up additional
processes the way a PyTorch DataLoader does, so it does not multiply throughput. (By
contrast, a PyTorch DataLoader scales with `num_workers` because each worker is a
separate process; tf.data uses AUTOTUNE.)

End-to-end, on Apple Silicon over Imagenette-160 (9,469 real ImageNet JPEGs → 160px,
`output_format='pytorch'`, batch 64, real consumption forcing materialization, warmup +
median of 3 epochs):

| Loader | Throughput |
|--------|------------|
| TurboLoader (cached, `cache_decoded=True`) | ~65,499 img/s |
| TurboLoader (on the fly) | ~39,100 img/s |
| TensorFlow tf.data (AUTOTUNE) | ~30,154 img/s |
| PyTorch DataLoader (8 persistent workers) | ~18,991 img/s |

On the fly, TurboLoader is about **1.3x** tf.data and **2.1x** the PyTorch DataLoader;
with the decoded cache it is faster still.

### CPU Utilization

The direct-batch pool keeps all cores busy with minimal idle time, since decode,
resize, and normalize run in parallel and write straight into the output batch buffer
without a per-sample queue hop.

## Design Decisions

### Why C++20?

**Benefits:**
- Modern concurrency primitives (`std::atomic`, `std::thread`)
- SIMD intrinsics support
- Zero-overhead abstractions
- Better performance than Python multiprocessing

**Tradeoffs:**
- Longer compile times
- More complex build system
- Requires C++20 compiler

### Why a Direct-Batch Fast Path?

Earlier versions passed every sample through a queue between the worker threads and the
Python consumer. The default fast path now skips that hop entirely: worker threads in
the process-wide pool decode, resize, and normalize **directly into the output batch
buffer**, so there is no per-sample enqueue / dequeue on the hot path. Lock-free SPSC
queues are kept only for the legacy streaming and remote-source paths (see section 4),
where producer and consumer genuinely run at different rates.

### Why mmap Instead of read()?

**read() syscalls:**
- Kernel → user space copy on every call
- System-call overhead per read

**mmap():**
- Zero-copy (kernel manages pages)
- Page-cache reuse across epochs
- Random access without explicit seeks

mmap avoids the per-read copy and lets the OS page cache serve repeated reads, which
matters most when a dataset is read across many epochs.

## Code Organization

```
turboloader/
├── src/
│   ├── pipeline/
│   │   ├── pipeline.hpp          # Main pipeline
│   │   └── unified_pipeline.hpp  # Multi-format support
│   ├── tar/
│   │   ├── tar_reader.hpp        # TAR parsing
│   │   └── tar_index.hpp         # Index builder
│   ├── decode/
│   │   ├── image_decoder.hpp     # JPEG/PNG decoder
│   │   └── decoder_pool.hpp      # Thread-local pool
│   ├── transforms/
│   │   ├── transform_base.hpp    # Base class
│   │   ├── simd_utils.hpp        # SIMD helpers
│   │   ├── resize_transform.hpp  # Resize
│   │   ├── normalize_transform.hpp
│   │   ├── autoaugment_transform.hpp
│   │   └── [18 more transforms]
│   ├── queue/
│   │   └── concurrent_queue.hpp  # Lock-free SPSC
│   └── python/
│       └── turboloader_bindings.cpp  # pybind11
├── tests/
│   ├── test_transforms.cpp       # C++ unit tests
│   └── test_pytorch_transforms.py # Python integration
└── benchmarks/
    └── benchmark_comparison.py   # Performance tests
```

## Version History

### v2.31.0 Current Features

1. **Direct-batch fast path** ✅
   - FFCV / tf.data-style loader: one parallel pass decodes → resizes → normalizes
     straight into the output batch buffer
   - Automatic libjpeg-turbo DCT scaled decode for large images
   - Process-wide C++ thread pool (saturated from a single Python worker)

2. **Multi-modality** ✅
   - Images in WebDataset TAR
   - LLM token streams via `TokenDataLoader`
   - Generic `(N, ...)` arrays via `ArrayDataLoader`

3. **Decoded cache** ✅
   - `cache_decoded=True` skips re-decoding on subsequent epochs
   - Substantially faster repeat epochs once the cache is warm

4. **TBL v2 Binary Format** ✅
   - **LZ4 compression** - space savings vs TAR
   - **Streaming writer** - O(1) constant memory usage (not O(n))
   - **CRC32/CRC16 checksums** - Data integrity validation
   - **Cached image dimensions** - 16-bit width/height in index for fast filtering
   - **Rich metadata support** - JSON, Protobuf, MessagePack formats
   - **64-byte cache-aligned header**
   - **24-byte index entries** - Compressed size, format, dimensions, CRC16
   - **Code Locations:** `src/formats/tbl_v2_format.hpp`, `src/readers/tbl_v2_reader.hpp`, `src/writers/tbl_v2_writer.hpp`, `src/compression/lz4_compressor.hpp`

5. **DDP-safe distributed sharding** ✅
   - Equal / disjoint sharding across ranks for distributed training
   - Compatible with PyTorch DDP
   - **Code Location:** `src/distributed/`

6. **Packaging** ✅
   - `setuptools_scm` versioning; Trusted-Publishing PyPI releases on tags
   - Prebuilt manylinux wheels (Linux x86_64 + aarch64) plus an sdist; portable macOS
     wheels (built from source) are being added

### Earlier Features

1. **AVX-512 SIMD Support** ✅
   - 16-wide vector operations (2x throughput vs AVX2)
   - Compatible with Intel Skylake-X+, AMD Zen 4+
   - Graceful fallback to AVX2/NEON
   - **Code Location:** `src/transforms/simd_utils.hpp`

2. **Prefetching Pipeline** ✅
   - Double-buffering strategy for overlapped I/O
   - Reduces epoch time by eliminating wait states
   - **Code Location:** `src/pipeline/prefetch_pipeline.hpp`

3. **Binary Format Improvements** ✅
   - Optimized storage format for ML datasets
   - O(1) random access via index table
   - Foundation for TBL v2 format

### Future Improvements

1. **ZSTD Compression**
   - Higher compression ratios than LZ4
   - Tunable compression levels

2. **Cloud Storage Optimizations**
   - S3/GCS streaming with the TBL v2 format
   - Not built into the published wheel today (source-only / optional, planned)

## References

- [FFCV Paper](https://arxiv.org/abs/2306.12517) - Inspiration for design
- [Intel AVX2 Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide)
- [Lock-Free Programming](https://www.1024cores.net/home/lock-free-algorithms)

## Questions?

For architecture questions, see:
- [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- [Contributing Guide](development/contributing.md)
