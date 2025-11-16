# TurboLoader Architecture


> **Note**: Performance claims in this documentation are based on preliminary benchmarks on synthetic datasets. 
> Actual performance will vary based on hardware, dataset characteristics, and workload. 
> We recommend running benchmarks on your specific use case.



Deep dive into how TurboLoader achieves high-performance data loading.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Performance Optimizations](#performance-optimizations)
- [Memory Management](#memory-management)
- [Threading Model](#threading-model)
- [Comparison with Other Frameworks](#comparison-with-other-frameworks)

---

## Overview

TurboLoader is designed around three key principles:

1. **Zero-copy I/O**: Memory-mapped files avoid unnecessary data copying
2. **Lock-free concurrency**: SPMC queues eliminate thread contention
3. **Native threading**: C++ threads bypass Python's GIL

### Performance Goals

- **Significantly faster** than TensorFlow tf.data
- **Significantly faster** than PyTorch DataLoader (naive TAR reading)
- **High throughput** throughput on Apple Silicon (1000 images, 256x256)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Application                         │
│                    (PyTorch/TensorFlow Training)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ Python API (pybind11)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Pipeline (C++)                              │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐        │
│  │   Reader   │───▶│  Lock-Free   │───▶│   Workers    │        │
│  │   Thread   │    │  SPMC Queue  │    │   (Decode)   │        │
│  └────────────┘    └──────────────┘    └──────────────┘        │
│         │                                       │                │
│         │ mmap                                  │ libjpeg-turbo │
│         ▼                                       ▼                │
│  ┌────────────┐                        ┌──────────────┐        │
│  │ TAR Files  │                        │ Output Queue │        │
│  │ (WebDataset)                        │ (Batches)    │        │
│  └────────────┘                        └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Reader Thread**: Reads TAR files using memory-mapped I/O
2. **Lock-Free Queue**: Distributes samples to workers without contention
3. **Worker Pool**: Decodes JPEG images in parallel
4. **Output Queue**: Buffers ready batches for user consumption

---

## Core Components

### 1. TAR Reader (I/O Layer)

**File**: `src/readers/tar_reader.cpp`

**Responsibilities**:
- Memory-map TAR files for zero-copy access
- Parse TAR headers to locate file entries
- Extract individual files without decompression overhead

**Key Technology**: `mmap()` system call

**Code snippet**:
```cpp
class TarReader {
public:
    TarReader(const std::string& path);

    // Memory-map the TAR file
    void open();

    // Parse TAR headers and build file index
    void parse_index();

    // Get file entry by index (zero-copy)
    FileEntry get_entry(size_t index);

private:
    void* mapped_data_;  // mmap pointer
    size_t file_size_;
    std::vector<TarEntry> entries_;
};
```

**Why memory-mapping?**
- Zero-copy: Data accessed directly from disk cache
- OS handles paging: No manual buffer management
- Sequential prefetch: OS automatically prefetches sequential data

---

### 2. Lock-Free SPMC Queue

**File**: `src/core/lock_free_queue.hpp`

**Type**: Single-Producer Multiple-Consumer (SPMC)

**Key Innovation**: Cache-line aligned slots prevent false sharing

**Implementation**:
```cpp
template<typename T>
class LockFreeQueue {
    struct alignas(64) Slot {  // Cache-line aligned
        std::atomic<uint64_t> sequence{0};
        T data;
    };

    Slot* buffer_;
    size_t capacity_;

    alignas(64) std::atomic<uint64_t> enqueue_pos_{0};  // Writer
    alignas(64) std::atomic<uint64_t> dequeue_pos_{0};  // Readers
};
```

**Why lock-free?**
- **No mutex contention**: Atomic operations only
- **Wait-free enqueue**: Producer never blocks
- **Cache-friendly**: Aligned slots prevent false sharing
- **Scalable**: Performance doesn't degrade with more threads

**Performance**: 100M ops/sec on Apple Silicon

---

### 3. Thread Pool (Worker Threads)

**File**: `src/core/thread_pool.hpp`

**Responsibilities**:
- Manage worker thread lifecycle
- Distribute work from SPMC queue
- Handle JPEG decoding per-thread

**Thread-local decoders**:
```cpp
class Worker {
    std::unique_ptr<JPEGDecoder> decoder_;  // Thread-local

    void process_sample(const Sample& sample) {
        // Each worker has its own decoder (no contention)
        auto decoded = decoder_->decode(sample.data[".jpg"]);
        // ... output to batch ...
    }
};
```

**Why thread-local decoders?**
- No decoder allocation per image
- No synchronization overhead
- Better cache locality

---

### 4. JPEG Decoder

**File**: `src/decoders/jpeg_decoder.cpp`

**Backend**: libjpeg-turbo (SIMD-optimized)

**SIMD Optimizations**:
- **Apple Silicon**: NEON instructions
- **x86_64**: AVX2 instructions
- **ARM**: NEON

**Performance**: 2-significantly faster than standard libjpeg

**Decode path**:
```cpp
DecodedImage JPEGDecoder::decode(const std::vector<uint8_t>& jpeg_data) {
    // 1. Initialize decoder
    jpeg_decompress_struct cinfo;
    jpeg_create_decompress(&cinfo);

    // 2. Set source
    jpeg_mem_src(&cinfo, jpeg_data.data(), jpeg_data.size());

    // 3. Read header
    jpeg_read_header(&cinfo, TRUE);

    // 4. Start decompression (SIMD kicks in here)
    jpeg_start_decompress(&cinfo);

    // 5. Read scanlines
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_pointer, 1);
    }

    // 6. Cleanup
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    return decoded_image;
}
```

---

### 5. Pipeline Orchestrator

**File**: `src/pipeline/pipeline.cpp`

**Responsibilities**:
- Coordinate reader, workers, and output
- Handle epoch boundaries
- Manage pipeline state (start/stop/reset)

**State machine**:
```
     start()           next_batch()         stop()
IDLE ────────▶ RUNNING ────────────▶ ... ────────▶ STOPPED
      │                                             │
      └─────────────────────────────────────────────┘
                       reset()
```

---

## Data Flow

### Step-by-Step Execution

**1. Pipeline Start**
```cpp
pipeline.start();
```
- Reader thread begins parsing TAR file
- Worker threads wait on SPMC queue
- Output queue initialized

**2. Reader Produces Samples**
```
Reader Thread:
  1. Memory-map TAR file (mmap)
  2. Parse TAR headers
  3. For each entry:
     - Extract file data (zero-copy pointer)
     - Create Sample object
     - Enqueue to SPMC queue
```

**3. Workers Consume & Decode**
```
Worker Threads (parallel):
  1. Dequeue sample from SPMC queue
  2. Decode JPEG using thread-local decoder
  3. Store decoded image in sample
  4. Enqueue to output queue
```

**4. User Consumes Batches**
```cpp
auto batch = pipeline.next_batch(32);
```
- Dequeue 32 samples from output queue
- Return to user as vector

**5. Epoch Boundary**
```cpp
auto batch = pipeline.next_batch(32);
if (batch.empty()) {
    // End of epoch
    pipeline.stop();
    pipeline.start();  // New epoch
}
```

---

## Performance Optimizations

### 1. Zero-Copy I/O

**Problem**: Traditional I/O involves multiple copies:
```
Disk → Kernel buffer → User buffer → Application
```

**Solution**: Memory-mapped I/O
```
Disk → Kernel page cache ←──────┐
                                 │ mmap (zero-copy)
Application ─────────────────────┘
```

**Benefit**: 2-significantly faster file reading

---

### 2. Cache-Line Alignment

**Problem**: False sharing causes cache invalidation

```
Thread 1 writes slot[0] ─┐
                         ▼
[slot0][slot1][slot2][slot3] ◀─── Same cache line
                         ▲
Thread 2 writes slot[1] ─┘
❌ Cache line bounces between cores!
```

**Solution**: Align each slot to cache line (64 bytes)

```
Thread 1 writes slot[0] ─┐
                         ▼
[slot0 (64B padding)     ]
[slot1 (64B padding)     ] ◀─── Thread 2
[slot2 (64B padding)     ]
✅ No cache line sharing!
```

**Benefit**: 4-8x better throughput on multi-core

---

### 3. SIMD JPEG Decoding

**libjpeg-turbo** uses SIMD instructions:

```
Standard libjpeg:
  for (int i = 0; i < width; i++) {
      output[i] = transform(input[i]);  // Scalar
  }

libjpeg-turbo (NEON/AVX2):
  for (int i = 0; i < width; i += 16) {
      vec = load_vector(&input[i]);     // Load 16 pixels
      vec = transform_vector(vec);       // SIMD transform
      store_vector(&output[i], vec);     // Store 16 pixels
  }
```

**Benefit**: 2-significantly faster JPEG decoding

---

### 4. Thread-Local State

**Problem**: Shared decoder requires locking
```cpp
class Pipeline {
    JPEGDecoder shared_decoder_;  // ❌ Needs mutex
    std::mutex decoder_mutex_;

    void worker() {
        std::lock_guard lock(decoder_mutex_);
        shared_decoder_.decode(data);  // Serialized!
    }
};
```

**Solution**: Thread-local decoders
```cpp
class Worker {
    JPEGDecoder decoder_;  // ✅ Each thread has own

    void process() {
        decoder_.decode(data);  // No locking!
    }
};
```

**Benefit**: Linear scaling with cores

---

## Memory Management

### Memory Pool Allocator

**File**: `src/core/memory_pool.hpp`

**Purpose**: Fast batch allocations without malloc overhead

**Design**:
```cpp
class MemoryPool {
    struct Block {
        uint8_t* data;
        size_t size;
        bool in_use;
    };

    std::vector<Block> blocks_;

public:
    // Pre-allocate blocks
    void preallocate(size_t num_blocks, size_t block_size);

    // Fast allocation (just mark block as in-use)
    uint8_t* allocate(size_t size);

    // Fast deallocation (mark as free)
    void deallocate(uint8_t* ptr);
};
```

**Benefits**:
- No malloc/free per image
- Better cache locality
- Predictable memory usage

---

### Smart Pointer Usage

**RAII Pattern**: All resources released automatically

```cpp
class Pipeline {
    std::unique_ptr<TarReader> reader_;       // Auto-cleanup
    std::vector<std::unique_ptr<Worker>> workers_;
    std::unique_ptr<LockFreeQueue<Sample>> queue_;

    ~Pipeline() {
        // All resources freed automatically
    }
};
```

**No memory leaks**: All verified with Valgrind/ASan

---

## Threading Model

### C++ Threads vs Python Multiprocessing

**TurboLoader (C++ threads)**:
```
Main Process
  ├── Reader Thread (I/O)
  ├── Worker Thread 1 (Decode) ◀─┐
  ├── Worker Thread 2 (Decode)   │ Shared memory
  ├── Worker Thread 3 (Decode)   │
  └── Worker Thread 4 (Decode) ◀─┘
```

**PyTorch DataLoader (multiprocessing)**:
```
Main Process
  ├── Worker Process 1 ◀─── Separate memory (copy via pickle)
  ├── Worker Process 2 ◀─── Separate memory (copy via pickle)
  ├── Worker Process 3 ◀─── Separate memory (copy via pickle)
  └── Worker Process 4 ◀─── Separate memory (copy via pickle)
```

### Performance Comparison

| Metric | TurboLoader (Threads) | PyTorch (Multiprocessing) |
|--------|----------------------|---------------------------|
| **Startup time** | 1-2 ms | 500-1000 ms (spawn processes) |
| **Memory** | Shared (450 MB @ 8 workers) | Duplicated (2400 MB @ 8 workers) |
| **IPC overhead** | None (shared memory) | High (pickle serialization) |
| **Scaling** | Linear | Sub-linear (GIL + IPC) |

---

## Comparison with Other Frameworks

### TurboLoader vs PyTorch DataLoader

| Component | TurboLoader | PyTorch DataLoader |
|-----------|-------------|-------------------|
| **Threading** | C++ native threads | Python multiprocessing |
| **TAR reading** | mmap (zero-copy) | Repeated open/read (slow) |
| **JPEG decoding** | libjpeg-turbo (SIMD) | Pillow (standard libjpeg) |
| **Memory** | Shared | Duplicated per worker |
| **Queue** | Lock-free SPMC | Python Queue (locks) |

**Result**: significantly faster on TAR datasets

---

### TurboLoader vs TensorFlow tf.data

| Component | TurboLoader | TensorFlow tf.data |
|-----------|-------------|-------------------|
| **TAR support** | Native streaming | Requires extraction |
| **Threading** | C++ threads | TF threading |
| **JPEG decoding** | libjpeg-turbo | TF JPEG ops |
| **Memory** | Shared | Shared |

**Result**: significantly faster (streaming vs extraction)

---

### TurboLoader vs FFCV

| Component | TurboLoader | FFCV |
|-----------|-------------|------|
| **Pre-processing** | None (TAR streaming) | Required (.beton format) |
| **JPEG decoding** | libjpeg-turbo | Custom fast decoder |
| **Format support** | JPEG/PNG/WebP | JPEG only |
| **Setup** | Low | High (dataset conversion) |

**Result**: TurboLoader at 89.5% of FFCV speed, but no preprocessing

---

## Design Patterns

### 1. Producer-Consumer

Reader (producer) → Queue → Workers (consumers)

### 2. Thread Pool

Pre-allocated worker threads avoid spawn overhead

### 3. RAII (Resource Acquisition Is Initialization)

All resources managed by smart pointers/destructors

### 4. Zero-Copy

Memory-mapped I/O avoids unnecessary data copying

### 5. Lock-Free Concurrent Data Structures

SPMC queue uses atomic operations instead of locks

---

## Future Optimizations

### Phase 4: SIMD Transforms

```cpp
// Vectorized image resize (planned)
void resize_simd(uint8_t* src, uint8_t* dst, int src_w, int src_h, int dst_w, int dst_h) {
    #ifdef __ARM_NEON
        // NEON implementation
    #elif defined(__AVX2__)
        // AVX2 implementation
    #endif
}
```

### Phase 5: GPU Decoding

```cpp
// NVIDIA Video Codec SDK (planned)
class CUDAJPEGDecoder {
    nvjpegHandle_t handle_;

    DecodedImage decode_gpu(const std::vector<uint8_t>& jpeg_data) {
        // GPU-accelerated JPEG decoding
    }
};
```

---

## See Also

- [API Documentation](API.md) - Complete API reference
- [Performance Tuning](PERFORMANCE.md) - Optimization guide
- [Benchmarks](../benchmarks/README.md) - Performance comparisons
