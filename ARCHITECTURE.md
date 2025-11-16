# TurboLoader TurboLoader - High-Performance Pipeline Architecture

## Executive Summary

Complete rewrite of the data loading pipeline to achieve >3x speedup over PyTorch DataLoader by eliminating mutex contention, implementing zero-copy I/O, and using lock-free queues.

**Target Performance**: 150+ img/s (vs PyTorch 48 img/s, current 17.56 img/s)

---

## Design Principles

1. **Zero Mutex Contention**: Each worker has independent TAR file handle
2. **Zero-Copy I/O**: Use `std::span` views into mmap regions
3. **Lock-Free Queues**: SPSC ring buffers for sample passing
4. **Object Pooling**: Reuse buffers to eliminate allocations
5. **SIMD Acceleration**: turbojpeg for 2-3x faster JPEG decode

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Pipeline                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Worker 1  │  │  Worker 2  │  │  Worker N  │            │
│  │  (Thread)  │  │  (Thread)  │  │  (Thread)  │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │ SPSC           │ SPSC           │ SPSC            │
│         │ RingBuf        │ RingBuf        │ RingBuf         │
│         └────────────────┴────────────────┘                 │
│                          │                                   │
│                    ┌─────▼──────┐                           │
│                    │  Sampler   │                           │
│                    │ (Main      │                           │
│                    │  Thread)   │                           │
│                    └────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. Lock-Free SPSC Ring Buffer

**File**: `src/core/lockfree_queue.hpp`

**Key Features**:
- Single-Producer Single-Consumer (SPSC) optimized
- Cache-line aligned to prevent false sharing
- Atomic head/tail pointers with relaxed ordering
- Zero allocations after construction

**API**:
```cpp
template<typename T, size_t Capacity>
class SPSCRingBuffer {
public:
    bool try_push(T&& item);
    bool try_pop(T& item);
    size_t size() const;
    bool empty() const;
};
```

**Performance**:
- Push/Pop: ~10-20ns (vs 500ns with mutex)
- Zero contention by design (SPSC)

---

### 2. Per-Worker TAR Reader

**File**: `src/io/tar_reader_v2.cpp`

**Design**:
- Each worker opens independent file descriptor
- Workers read disjoint sample ranges
- Memory-mapped I/O for zero-copy reads
- No mutexes - complete isolation

**Worker Assignment**:
```
Total samples: 1000
Workers: 4

Worker 0: samples [0, 250)
Worker 1: samples [250, 500)
Worker 2: samples [500, 750)
Worker 3: samples [750, 1000)
```

**API**:
```cpp
class TarReader {
public:
    TarReader(const std::string& path);
    std::span<const uint8_t> read_sample_zero_copy(size_t index);
    size_t total_samples() const;
};
```

---

### 3. Zero-Copy Sample

**File**: `src/core/sample.hpp`

**Design**:
- Samples hold `std::span<const uint8_t>` views into mmap
- No copies until absolutely necessary (JPEG decode)
- Decoded data goes into pooled buffers

**Structure**:
```cpp
struct Sample {
    size_t index;
    std::span<const uint8_t> jpeg_data;  // View into mmap (zero-copy)
    std::vector<uint8_t> decoded_rgb;     // From pool (reused)
    int width, height, channels;
};
```

---

### 4. Object Pool for Decoded Buffers

**File**: `src/core/object_pool.hpp`

**Design**:
- Pre-allocate buffers for decoded images
- Workers acquire/release from thread-local pools
- Eliminates malloc/free overhead

**API**:
```cpp
template<typename T>
class ObjectPool {
public:
    ObjectPool(size_t initial_size, size_t max_size);
    std::unique_ptr<T, Deleter> acquire();
    void release(T* obj);
};
```

---

### 5. TurboJPEG Decoder

**File**: `src/decoders/turbojpeg_decoder.cpp`

**Dependencies**: `brew install jpeg-turbo`

**Advantages**:
- 2-3x faster than libjpeg (SIMD AVX2/NEON)
- Batch decompression support
- Zero-copy decoding into provided buffer

**API**:
```cpp
class TurboJPEGDecoder {
public:
    void decode(std::span<const uint8_t> jpeg_data,
                std::span<uint8_t> output_rgb);
};
```

---

### 6. Worker Thread

**File**: `src/pipeline/worker_v2.cpp`

**Responsibilities**:
1. Read assigned sample range from TAR (zero-copy)
2. Decode JPEG using turbojpeg (into pooled buffer)
3. Push to SPSC queue (lock-free)

**Pseudocode**:
```cpp
void Worker::run() {
    while (running) {
        size_t index = get_next_sample_index();

        // Zero-copy read from TAR
        auto jpeg_span = tar_reader->read_sample_zero_copy(index);

        // Acquire buffer from pool
        auto buffer = pool->acquire();

        // Decode JPEG (SIMD accelerated)
        decoder->decode(jpeg_span, buffer->data);

        // Push to queue (lock-free, ~10ns)
        while (!queue->try_push(std::move(buffer))) {
            std::this_thread::yield();
        }
    }
}
```

---

## Performance Analysis

### Bottleneck Elimination

| Bottleneck | Old Design | New Design | Improvement |
|------------|-----------|------------|-------------|
| TAR Read | Mutex (1 at a time) | Per-worker FD (parallel) | 4x |
| Memory Copy | memcpy JPEG data | Zero-copy span | 2x |
| Queue Push | Mutex (~500ns) | Lock-free (~10ns) | 50x |
| JPEG Decode | libjpeg | turbojpeg SIMD | 2-3x |
| Buffer Alloc | malloc/free | Object pool | 5-10x |

**Combined**: ~4x × 1.5x × 1.5x × 2.5x × 2x = **45x speedup**

**Realistic (accounting for I/O bound)**: **3-5x speedup**

### Expected Throughput

- PyTorch baseline: 48 img/s
- TurboLoader old: 17.56 img/s
- **TurboLoader**: 150-200 img/s

---

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)
- [x] Lock-free SPSC ring buffer
- [ ] Object pool
- [ ] Zero-copy sample struct

### Phase 2: I/O Layer (2-3 hours)
- [ ] TarReader with per-worker handles
- [ ] Memory-mapped I/O
- [ ] Sample index partitioning

### Phase 3: Decoding (1-2 hours)
- [ ] TurboJPEG integration
- [ ] Decoder with pooled buffers

### Phase 4: Pipeline (3-4 hours)
- [ ] Worker thread implementation
- [ ] Main pipeline orchestration
- [ ] Graceful shutdown

### Phase 5: Testing & Benchmarking (2-3 hours)
- [ ] Unit tests for each component
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Memory leak checks (valgrind)

### Phase 6: Integration (1-2 hours)
- [ ] Python bindings update
- [ ] Documentation update
- [ ] Example scripts update

**Total Estimated Time**: 11-17 hours

---

## Memory Safety Considerations

1. **Mmap Lifetime**: Ensure mmap outlives all `std::span` views
2. **Pool Ownership**: Use `std::unique_ptr` with custom deleter
3. **Thread Safety**: SPSC queues are inherently thread-safe for 1-1
4. **Shutdown**: Graceful worker termination before destroying queues

---

## Fallback Compatibility

For systems without turbojpeg, provide:
- CMake option: `USE_TURBOJPEG` (default: ON)
- Fallback to libjpeg if not available
- Performance warning at runtime

---

## Success Criteria

- ✅ Build without warnings (CMake + make)
- ✅ All tests pass (C++ unit + Python integration)
- ✅ No memory leaks (valgrind clean)
- ✅ No data races (ThreadSanitizer clean)
- ✅ **Performance: >100 img/s (2x PyTorch)**
- ✅ Stable under load (1000+ batches)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| turbojpeg not available | Fallback to libjpeg + warning |
| Memory leaks | RAII + smart pointers everywhere |
| Data races | ThreadSanitizer in CI |
| API breakage | Versioned symbols, deprecation warnings |
| Performance regression | Automated benchmarks in CI |

---

## Next Steps

1. Implement lock-free SPSC ring buffer
2. Write comprehensive unit tests
3. Proceed with per-worker TAR reader
4. Continue incrementally with testing at each stage
