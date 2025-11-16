# TurboLoader v2.0: High-Performance Pipeline Rewrite

## Problem Statement

TurboLoader v0.3.8 is currently **2.7x slower** than PyTorch DataLoader baseline, the opposite of its intended performance improvement.

### Benchmark Results (1000 images, 4 workers, batch_size=32)
- **PyTorch DataLoader**: 48.07 img/s
- **TurboLoader v0.3.8**: 17.56 img/s
- **Performance gap**: -63% slower than baseline

## Root Cause Analysis

Comprehensive C++ code analysis identified 5 critical bottlenecks:

### 1. TAR Mutex Contention (75% impact)
**Location**: `src/pipeline/pipeline.cpp:228`

All workers serialize on a single mutex when reading from the TAR file:
```cpp
std::lock_guard<std::mutex> lock(*reader_mutexes_[reader_idx]);
const auto& tar_sample = reader->get_sample(local_index);
```

**Impact**: With 4 workers, only 1 can read at a time. This eliminates the benefit of parallelism.

### 2. Memory Allocation/Copy Overhead (20% impact)
**Location**: `src/pipeline/pipeline.cpp:240-243`

Data is copied from mmap regions while holding the mutex:
```cpp
std::vector<uint8_t> data(data_span.size());
std::memcpy(data.data(), data_span.data(), data_span.size());
```

### 3. Busy-Wait Spinning (10% impact)
**Location**: `src/pipeline/pipeline.cpp:198-200`

Queue push uses CPU-intensive spinning:
```cpp
while (running_ && !output_queue_->try_push(std::move(sample))) {
    std::this_thread::yield();
}
```

### 4. Thread Pool Overhead (10% impact)
Current queue implementation has contention under high load.

### 5. JPEG Decoder Inefficiency (5% impact)
Using libjpeg instead of SIMD-optimized turbojpeg.

## Proposed Solution: v2.0 Architecture

Complete rewrite based on **zero-copy, lock-free** principles. Full design document: [ARCHITECTURE_V2.md](https://github.com/ALJainProjects/TurboLoader/blob/v2.0-rewrite/ARCHITECTURE_V2.md)

### Core Design Principles

1. **Zero Mutex Contention**: Each worker has independent TAR file handle
2. **Zero-Copy I/O**: Use `std::span` views into mmap regions
3. **Lock-Free Queues**: SPSC ring buffers for sample passing
4. **Object Pooling**: Reuse buffers to eliminate allocations
5. **SIMD Acceleration**: turbojpeg for 2-3x faster JPEG decode

### Architecture Overview

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

### Expected Performance Improvements

| Component | Old Design | New Design | Improvement |
|-----------|-----------|------------|-------------|
| TAR Read | Mutex (serial) | Per-worker FD (parallel) | 4x |
| Memory Copy | memcpy JPEG data | Zero-copy span | 2x |
| Queue Push | Mutex (~500ns) | Lock-free (~10ns) | 50x |
| JPEG Decode | libjpeg | turbojpeg SIMD | 2-3x |
| Buffer Alloc | malloc/free | Object pool | 5-10x |

**Combined theoretical speedup**: ~45x
**Realistic (I/O bound)**: **3-5x faster than PyTorch (150-200 img/s target)**

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)
- [ ] Lock-free SPSC ring buffer (`src/core/lockfree_queue.hpp`)
- [ ] Object pool (`src/core/object_pool.hpp`)
- [ ] Zero-copy sample struct (`src/core/sample_v2.hpp`)
- [ ] Unit tests for each component

### Phase 2: I/O Layer (2-3 hours)
- [ ] TarReaderV2 with per-worker handles (`src/io/tar_reader_v2.cpp`)
- [ ] Memory-mapped I/O implementation
- [ ] Sample index partitioning logic
- [ ] Integration tests

### Phase 3: Decoding (1-2 hours)
- [ ] TurboJPEG integration (`src/decoders/turbojpeg_decoder.cpp`)
- [ ] Decoder with pooled buffers
- [ ] CMake dependency detection
- [ ] Fallback to libjpeg if unavailable

### Phase 4: Pipeline (3-4 hours)
- [ ] Worker thread implementation (`src/pipeline/worker_v2.cpp`)
- [ ] Main pipeline orchestration
- [ ] Graceful shutdown mechanism
- [ ] Error handling

### Phase 5: Testing & Benchmarking (2-3 hours)
- [ ] Unit tests for all components
- [ ] Integration tests
- [ ] Performance benchmarks vs PyTorch
- [ ] Memory leak checks (valgrind)
- [ ] Thread safety verification (ThreadSanitizer)

### Phase 6: Integration (1-2 hours)
- [ ] Python bindings update
- [ ] Documentation update
- [ ] Example scripts update
- [ ] Migration guide for users

**Total Estimated Time**: 11-17 hours

## Success Criteria

- [ ] Build without warnings (CMake + make)
- [ ] All tests pass (C++ unit + Python integration)
- [ ] No memory leaks (valgrind clean)
- [ ] No data races (ThreadSanitizer clean)
- [ ] **Performance: >100 img/s (2x PyTorch minimum)**
- [ ] Stable under load (1000+ batches without crashes)

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| turbojpeg not available | Fallback to libjpeg + warning |
| Memory leaks | RAII + smart pointers everywhere |
| Data races | ThreadSanitizer in CI |
| API breakage | Versioned symbols, deprecation warnings |
| Performance regression | Automated benchmarks in CI |

## Development Branch

All v2.0 work will be done on the `v2.0-rewrite` branch to avoid disrupting the stable v0.3.x releases.

```bash
git checkout v2.0-rewrite
```

## References

- [ARCHITECTURE_V2.md](https://github.com/ALJainProjects/TurboLoader/blob/v2.0-rewrite/ARCHITECTURE_V2.md) - Complete design document
- Benchmark results: See issue description above
- Related benchmarks: `benchmark_turboloader_tar.py`, `benchmark_pytorch_tar.py`

## Labels

- `enhancement`
- `performance`
- `breaking-change`
- `v2.0`

---

**Priority**: High
**Complexity**: High
**Impact**: Critical - fixes fundamental performance regression
