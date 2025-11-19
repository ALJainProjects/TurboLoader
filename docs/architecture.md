# TurboLoader Architecture

This document describes the internal architecture of TurboLoader v1.5.0.

## Overview

TurboLoader achieves **10,146 img/s throughput** through a carefully designed multi-threaded pipeline architecture with SIMD-accelerated operations. Version 1.5.0 introduces the TBL v2 format with LZ4 compression, streaming writer, and enhanced data integrity features.

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
         │   Lock-Free Output Queue       │
         │   (SPSC ring buffer)            │
         └──────┬─────────────────────────┘
                │
         ┌──────▼──────────────┐
         │   Python Iterator   │
         │   (batch assembly)  │
         └─────────────────────┘
```

## Core Components

### 1. Memory-Mapped Reader (TAR/TBL v2)

**Purpose:** Zero-copy file access for TAR archives and TBL v2 binary format

**TAR Reader:**
- Uses `mmap()` system call to map file into memory
- Parses TAR headers (512-byte chunks) to locate samples
- Builds index of file offsets for random access
- Throughput: **52+ Gbps** on local SSD

**TBL v2 Reader (NEW in v1.5.0):**
- Memory-mapped access to compressed binary format
- LZ4 decompression for 40-60% space savings vs TAR
- CRC32 checksum validation for data integrity
- CRC16 validation for index entries
- Cached image dimensions (width/height) in 16-bit index
- Zero-copy reads with on-demand LZ4 decompression (2.5-3.5 GB/s)
- 24-byte index entries with full metadata

**Performance:**
- TAR: 52+ Gbps throughput
- TBL v2: 48+ Gbps (including LZ4 decompression)
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

**Performance:**
- JPEG: Up to **1.5 GB/s** decode bandwidth
- PNG: Up to **800 MB/s** decode bandwidth

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

### 4. Lock-Free Concurrent Queues

**Purpose:** High-throughput sample passing between threads

**Implementation:**
- **SPSC (Single-Producer Single-Consumer) ring buffer**
- Cache-line aligned (`alignas(64)`)
- Atomic operations for indices
- No mutex locks in hot path

**Performance:**
- Enqueue/dequeue: **~10-20 CPU cycles**
- Compared to mutex queues: **~50x faster**

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
- Default: **8 workers** (optimal for most systems)
- Configurable: 1-32 workers
- Rule of thumb: `2 * num_CPU_cores`

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
- Main thread → Workers: Work queue (lock-free)
- Workers → Main thread: Result queue (lock-free)
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
   └─ Push to output queue
                ↓
3. Python Iterator
   ├─ Pop batch_size samples from output queue
   ├─ Assemble into Python list
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

| Component | Throughput | Bottleneck |
|-----------|------------|------------|
| TAR Reader | 52+ Gbps | SSD bandwidth |
| JPEG Decode | 1.5 GB/s | CPU decode |
| SIMD Resize | 3.2x PyTorch | Memory bandwidth |
| Normalize | 5.0 GB/s | SIMD FMA units |
| Queue ops | 50M/s | Cache latency |

### Scalability

**Worker Count vs Throughput:**
```
1 worker:   1,500 img/s
2 workers:  3,000 img/s  (2.0x)
4 workers:  5,800 img/s  (3.9x)
8 workers: 10,146 img/s  (6.8x)
16 workers: 12,300 img/s  (8.2x)  ← diminishing returns
```

**Bottleneck Analysis:**
- 1-4 workers: CPU decode bound
- 4-8 workers: Balanced (decode + transform)
- 8-16 workers: Memory bandwidth bound
- 16+ workers: Queue contention + cache misses

### CPU Utilization

**Target:** 90-95% CPU utilization on all cores

**Achieved:**
- 8 workers on 10-core CPU: **92% average**
- Minimal idle time
- Low mutex contention (<1% overhead)

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

### Why Lock-Free Queues?

**Mutex-based queues (v0.5.0):**
- Lock contention under high load
- Throughput: **~500 img/s** (8 workers)

**Lock-free queues (v0.6.0+):**
- No lock contention
- Throughput: **10,146 img/s** (8 workers)
- **20x improvement**

### Why mmap Instead of read()?

**read() syscalls:**
- Kernel → user space copy
- System call overhead
- Throughput: **~2 GB/s**

**mmap():**
- Zero-copy (kernel manages pages)
- Page cache reuse
- Throughput: **52+ GB/s**
- **26x improvement**

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

### v1.5.0 New Features (Current)

1. **TBL v2 Binary Format** ✅
   - **LZ4 compression** - 40-60% space savings vs TAR
   - **Streaming writer** - O(1) constant memory usage (not O(n))
   - **CRC32/CRC16 checksums** - Data integrity validation
   - **Cached image dimensions** - 16-bit width/height in index for fast filtering
   - **Rich metadata support** - JSON, Protobuf, MessagePack formats
   - **64-byte cache-aligned header** - Optimal CPU cache performance
   - **24-byte index entries** - Compressed size, format, dimensions, CRC16
   - **4,875 img/s conversion** - TAR→TBL throughput with parallel processing
   - **Code Locations:** `src/formats/tbl_v2_format.hpp`, `src/readers/tbl_v2_reader.hpp`, `src/writers/tbl_v2_writer.hpp`, `src/compression/lz4_compressor.hpp`

### v1.2.0-1.2.1 Features

1. **GPU JPEG Decoding (nvJPEG)** ✅
   - NVIDIA nvJPEG support for 10x faster JPEG decoding
   - Automatic CPU fallback when GPU unavailable
   - **Code Location:** `src/decode/nvjpeg_decoder.hpp`

2. **Linux io_uring Async I/O** ✅
   - 2-3x faster disk throughput on NVMe SSDs
   - Zero-copy O_DIRECT support
   - **Code Location:** `src/io/io_uring_reader.hpp`

3. **Smart Batching** ✅
   - Size-based sample grouping reduces padding by 15-25%
   - ~1.2x throughput improvement
   - **Code Location:** `src/pipeline/smart_batching.hpp`

4. **Distributed Training** ✅
   - Multi-node data loading with deterministic sharding
   - Compatible with PyTorch DDP, Horovod, DeepSpeed
   - **Code Location:** `src/distributed/`

### v1.1.0 Features

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

### Future Improvements (v1.6+)

1. **ZSTD Compression**
   - Higher compression ratios than LZ4
   - Tunable compression levels

2. **Video Dataloader Enhancements**
   - Frame-level decoding
   - Temporal augmentation

3. **Cloud Storage Optimizations**
   - S3/GCS streaming with TBL v2 format
   - Intelligent prefetching for cloud data

## References

- [FFCV Paper](https://arxiv.org/abs/2306.12517) - Inspiration for design
- [Intel AVX2 Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide)
- [Lock-Free Programming](https://www.1024cores.net/home/lock-free-algorithms)

## Questions?

For architecture questions, see:
- [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- [Contributing Guide](development/contributing.md)
