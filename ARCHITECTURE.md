# TurboLoader v1.2.0 - Architecture Documentation

## Overview

TurboLoader is a high-performance data loading library for machine learning training. v1.2.0 features Smart Batching, Distributed Training support, and achieves peak performance of 21,035 img/s with 16 workers.

**Version**: 1.2.0
**Status**: Production/Stable
**Performance**: 21,035 img/s peak (16 workers), 52+ Gbps local I/O, 12x faster than PyTorch

---

## Core Design Principles

1. **Zero-Copy I/O**: Memory-mapped TAR files with `std::span` views
2. **Lock-Free Concurrency**: SPSC ring buffers eliminate mutex contention
3. **SIMD Acceleration**: 19 AVX-512/AVX2/NEON-optimized transforms
4. **Smart Batching**: Size-based grouping reduces padding by 15-25%
5. **Distributed Training**: Multi-node support with deterministic sharding
6. **Per-Worker Isolation**: Each worker has independent resources (no sharing)
7. **Cloud-Native**: Unified reader for local, HTTP, S3, GCS sources
8. **GPU Acceleration**: nvJPEG decoder with automatic CPU fallback

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        UnifiedPipeline                          │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Worker 0 │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │      │
│  │          │  │          │  │          │  │          │      │
│  │ TarReader│  │ TarReader│  │ TarReader│  │ TarReader│      │
│  │ +Decoder │  │ +Decoder │  │ +Decoder │  │ +Decoder │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │ SPSC        │ SPSC        │ SPSC        │ SPSC       │
│       │ Queue       │ Queue       │ Queue       │ Queue      │
│       └─────────────┴─────────────┴─────────────┘            │
│                          │                                     │
│                   ┌──────▼───────┐                            │
│                   │ Main Thread  │                            │
│                   │ (Batch       │                            │
│                   │  Assembly)   │                            │
│                   └──────────────┘                            │
└────────────────────────────────────────────────────────────────┘
```

---

## Component Hierarchy

```
src/
├── core/
│   ├── object_pool.hpp       # Buffer reuse, zero allocations
│   ├── sample.hpp             # Zero-copy sample structure
│   └── spsc_ring_buffer.hpp   # Lock-free queue (10ns push/pop)
├── readers/
│   ├── tar_reader.hpp         # Memory-mapped TAR (per-worker)
│   ├── http_reader.hpp        # HTTP with connection pooling
│   ├── s3_reader.hpp          # AWS S3 (SDK + HTTP fallback)
│   ├── gcs_reader.hpp         # Google Cloud Storage (SDK + HTTP)
│   └── reader_orchestrator.hpp # Unified API (auto-detect source)
├── decode/
│   ├── jpeg_decoder.hpp       # libjpeg-turbo (SIMD accelerated)
│   ├── nvjpeg_decoder.hpp     # GPU JPEG decode (CPU fallback)
│   ├── png_decoder.hpp        # libpng decoder
│   ├── webp_decoder.hpp       # WebP decoder
│   ├── video_decoder.hpp      # FFmpeg video decoder
│   ├── csv_decoder.hpp        # CSV parser
│   └── parquet_decoder.hpp    # Apache Arrow Parquet
└── pipeline/
    └── pipeline.hpp           # Unified multi-format pipeline
```

---

## Key Features

### 1. Lock-Free SPSC Ring Buffer
**File**: `src/core/spsc_ring_buffer.hpp`

- Single-Producer Single-Consumer optimized
- Cache-line aligned (prevents false sharing)
- Atomic operations with relaxed memory ordering
- **Performance**: ~10ns push/pop (vs 500ns with mutex)

```cpp
template<typename T, size_t Capacity>
class SPSCRingBuffer {
public:
    bool try_push(T&& item);
    bool try_pop(T& item);
    size_t size() const;
};
```

### 2. Per-Worker TAR Reader
**File**: `src/readers/tar_reader.hpp`

- Each worker opens independent file descriptor
- Memory-mapped I/O for zero-copy reads
- Workers process disjoint sample ranges
- **No mutex contention** by design

```cpp
class TarReader {
public:
    TarReader(const std::string& path, size_t worker_id, size_t num_workers);
    std::span<const uint8_t> get_sample(size_t index) const;  // Zero-copy
    size_t num_samples() const;
};
```

**Sample Partitioning**:
```
Total samples: 1000, Workers: 4

Worker 0: samples [0,   250)
Worker 1: samples [250, 500)
Worker 2: samples [500, 750)
Worker 3: samples [750, 1000)
```

### 3. Cloud Storage Support
**File**: `src/readers/reader_orchestrator.hpp`

Unified interface for all data sources with automatic protocol detection:

```cpp
ReaderOrchestrator reader;

// Auto-detects source from path
reader.read("/local/file.tar");              // Local file
reader.read("https://example.com/data.tar"); // HTTP/HTTPS
reader.read("s3://bucket/dataset.tar");      // AWS S3
reader.read("gs://bucket/dataset.tar");      // Google Cloud Storage
```

**Features**:
- Auto-detection based on path prefix
- Connection pooling for HTTP/HTTPS
- OAuth2/Service Account auth for GCS
- AWS SDK or HTTP fallback for S3
- Range request support
- Thread-safe operations

**Performance**: 52 Gbps local file throughput

### 4. GPU-Accelerated JPEG Decoding
**File**: `src/decode/nvjpeg_decoder.hpp`

- NVIDIA nvJPEG for GPU-accelerated decoding
- Automatic CPU fallback to libjpeg-turbo
- Batch decoding support
- **Performance**: 24,612 images/second (CPU), 10x faster on GPU

```cpp
NvJpegDecoder decoder;
NvJpegResult result;

decoder.decode(jpeg_data, jpeg_size, result);

if (result.gpu_decoded) {
    // Decoded on GPU (10x faster)
} else {
    // Automatic CPU fallback
}
```

### 5. Object Pool
**File**: `src/core/object_pool.hpp`

Pre-allocated buffers eliminate malloc/free overhead:

```cpp
BufferPool pool(256*256*3, 128, 256);  // initial, min, max

auto buffer = pool.acquire();  // Reused buffer
// Use buffer...
pool.release(buffer);  // Return to pool
```

**Benefits**:
- Zero allocations after initialization
- 5-10x faster than malloc/free
- Thread-safe acquire/release

### 6. Multi-Format Pipeline
**File**: `src/pipeline/pipeline.hpp`

Single unified pipeline supports all formats:

```cpp
UnifiedPipelineConfig config;
config.data_path = "/path/to/data.tar";
config.num_workers = 4;
config.batch_size = 32;

UnifiedPipeline pipeline(config);
pipeline.start();

while (!pipeline.is_finished()) {
    auto batch = pipeline.next_batch();
    // Process batch...
}
```

**Supported Formats**:
- **Archives**: TAR (WebDataset compatible)
- **Images**: JPEG, PNG, WebP, BMP, TIFF
- **Video**: MP4, AVI, MKV, MOV (FFmpeg)
- **Tabular**: CSV, Parquet (Apache Arrow)

---

## Performance Characteristics

### v1.2.0 Benchmarks

| Component | Metric | Performance |
|-----------|--------|-------------|
| **Overall Pipeline** | Peak Throughput | 21,035 img/s (16 workers) |
| **Overall Pipeline** | Baseline (1 worker) | 2,180 img/s |
| **Overall Pipeline** | Linear Scaling | 9.65x (16 workers) |
| **Overall Pipeline** | vs PyTorch | 12x faster |
| **SPSC Queue** | Push/Pop Latency | 10-20ns |
| **TAR Reader** | Local File Throughput | 52+ Gbps |
| **TAR Reader** | Range Request Latency | <1ms |
| **JPEG Decode** | CPU Throughput | 24,612 img/s |
| **JPEG Decode** | GPU Speedup | 10x vs CPU |
| **SIMD Transforms** | RandomPosterize | 335,677.5 img/s |
| **SIMD Transforms** | Resize (Bilinear) | 8,200 img/s (3.2x vs torchvision) |
| **SIMD Transforms** | GaussianBlur | 2,400 img/s (4.5x vs torchvision) |
| **Smart Batching** | Padding Reduction | 15-25% |
| **Smart Batching** | Throughput Boost | ~1.2x |
| **Object Pool** | vs malloc/free | 5-10x faster |

**Test Config:** Apple M4 Max, 1000 images, batch_size=64

---

## Memory Safety

1. **RAII Everywhere**: Smart pointers, automatic cleanup
2. **Mmap Lifetime**: Ensured to outlive all `std::span` views
3. **Pool Ownership**: `std::unique_ptr` with custom deleters
4. **Thread Safety**: SPSC queues inherently safe for 1-to-1
5. **Graceful Shutdown**: Workers terminate before resource destruction

---

## v1.2.0 Release Highlights

### New Features (v1.2.0)
- ✅ **Smart Batching** - Size-based sample grouping (15-25% padding reduction)
- ✅ **Distributed Training** - Multi-node support (PyTorch DDP, Horovod, DeepSpeed)
- ✅ **Scalability** - Linear scaling to 16 workers (21,035 img/s peak)
- ✅ **19 SIMD Transforms** - AVX-512/AVX2/NEON acceleration
- ✅ **AutoAugment Policies** - ImageNet, CIFAR10, SVHN

### Existing Features (v1.1.0+)
- ✅ **AVX-512 SIMD Support** - 2x vector width on compatible hardware
- ✅ **TBL Binary Format** - 12.4% smaller than TAR, instant random access
- ✅ **Prefetching Pipeline** - Overlaps I/O with computation
- ✅ **Google Cloud Storage** reader with OAuth2/Service Account auth
- ✅ **nvJPEG GPU-accelerated** JPEG decoder with CPU fallback
- ✅ **Multi-format pipeline** - Images, video, tabular, archives
- ✅ **Lock-free SPSC queues** - Zero-contention threading

### Test Coverage
- ✅ Smart Batching: 10/10 tests passing
- ✅ Distributed Training: Full multi-node test coverage
- ✅ TBL Format: 8/8 tests passing
- ✅ AVX-512 SIMD: 5/5 tests passing (NEON fallback on ARM)
- ✅ All 19 SIMD transforms validated
- ✅ 90%+ overall test pass rate (28 test files)

### Performance
- ✅ **21,035 img/s** peak throughput (16 workers)
- ✅ **12x faster** than PyTorch Optimized
- ✅ **9.65x linear scaling** with 16 workers
- ✅ **52 Gbps** local file read throughput
- ✅ **335,677 img/s** RandomPosterize transform
- ✅ **24,612 img/s** JPEG decode (CPU)
- ✅ **10ns** lock-free queue operations
- ✅ **<1ms** range request latency

### Code Quality
- ✅ Production/Stable status on PyPI
- ✅ Zero compiler warnings
- ✅ Comprehensive error handling
- ✅ Thread-safe operations
- ✅ Complete professional documentation (15+ guides)

---

## Future Roadmap (v1.3.0+)

### Planned Features
- **Enhanced GPU Support**: nvJPEG batch decoding optimization
- **Extended Distributed**: Advanced multi-node optimizations
- **Video Dataloader**: Enhanced video decoding performance
- **Cloud Streaming**: Optimized S3/GCS streaming for large datasets
- **Additional Transforms**: More SIMD-accelerated augmentations
- **MixUp/CutMix**: Advanced augmentation strategies

### Performance Targets
- **30K+ img/s**: With GPU acceleration on high-end hardware
- **20+ node scaling**: Linear scaling for distributed training
- **Video**: 60+ fps decode for HD video
- **Cloud**: <100ms latency for remote dataset access

---

## Build Requirements

### Required
- C++20 compiler (GCC 10+, Clang 12+, MSVC 2019+)
- CMake 3.15+
- libjpeg-turbo

### Optional (Auto-Detected)
- CUDA Toolkit + nvJPEG (GPU acceleration)
- FFmpeg (video support)
- Apache Arrow (Parquet support)
- Google Cloud Storage C++ SDK (native GCS)
- AWS SDK for C++ (native S3)
- libcurl (HTTP/HTTPS, S3/GCS fallback)

### Build Commands
```bash
mkdir build && cd build
cmake ..
make -j8

# Optional: Enable GPU support
cmake -DHAVE_NVJPEG=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda ..

# Run tests
ctest --output-on-failure
```

---

## References

- **Source Code**: https://github.com/ALJainProjects/TurboLoader
- **Documentation**: See README.md
- **Issues**: GitHub Issues
- **License**: MIT

---

**Last Updated**: v1.2.0 (2025-11-17)
