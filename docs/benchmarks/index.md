# Benchmark Overview

Comprehensive performance analysis of TurboLoader v0.8.0.

## Executive Summary

TurboLoader achieves **10,146 images/second throughput**, making it:
- **12x faster** than PyTorch DataLoader (optimized)
- **3.2x faster** than PyTorch with local file caching
- **1.3x faster** than TensorFlow tf.data

## Latest Results (v0.7.0)

### Framework Comparison

| Rank | Framework | Throughput | vs TurboLoader | Avg Epoch Time | Memory |
|------|-----------|------------|----------------|----------------|--------|
| 1 | **TurboLoader** | **10,146 img/s** | **1.00x** | **0.18s** | **848 MB** |
| 2 | TensorFlow tf.data | 7,569 img/s | 0.75x | 0.26s | 1,245 MB |
| 3 | PyTorch Cached | 3,123 img/s | 0.31x | 0.64s | 2,104 MB |
| 4 | PyTorch Optimized | 835 img/s | 0.08x | 2.40s | 1,523 MB |
| 5 | PIL Baseline | 277 img/s | 0.03x | 7.22s | 645 MB |
| 6 | PyTorch Naive | 85 img/s | 0.01x | 23.67s | 1,834 MB |

### Test Configuration

- **Hardware:** Apple M4 Max (16 cores, 48 GB RAM)
- **Dataset:** 2000 synthetic images, 256x256 JPEG (117 MB TAR)
- **Workers:** 8
- **Batch Size:** 32
- **Epochs:** 3
- **Transforms:** Resize(224x224) + Normalize + RandomHorizontalFlip

## Transform Performance

### Individual Transform Benchmarks (v0.7.0)

| Transform | Throughput | vs torchvision | Notes |
|-----------|------------|----------------|-------|
| **RandomPosterize** | **336,700 img/s** | **N/A** | Bitwise ops (ultra-fast) |
| **RandomSolarize** | **21,300 img/s** | **N/A** | SIMD threshold compare |
| **AutoAugment (ImageNet)** | **19,800 img/s** | **0.5x** | Composite policy |
| **RandomPerspective** | **9,900 img/s** | **N/A** | SIMD interpolation |
| **RandomHorizontalFlip** | **10,500 img/s** | **3.2x** | SIMD memory ops |
| **Resize (Bilinear)** | **8,200 img/s** | **3.2x** | SIMD interpolation |
| **ColorJitter** | **5,100 img/s** | **2.1x** | SIMD color ops |
| **Resize (Lanczos)** | **2,900 img/s** | **1.8x** | High-quality downsample |
| **GaussianBlur (k=5)** | **2,400 img/s** | **4.5x** | Separable convolution |
| **RandomErasing** | **8,300 img/s** | **2.8x** | Fast memory fill |

## Throughput vs Worker Count

```
Workers | Throughput | Speedup | CPU Usage
--------|------------|---------|----------
   1    |  1,500     |  1.0x   |  100%
   2    |  3,000     |  2.0x   |  200%
   4    |  5,800     |  3.9x   |  390%
   8    | 10,146     |  6.8x   |  750%
  16    | 12,300     |  8.2x   |  920%
  32    | 13,100     |  8.7x   |  980%
```

**Analysis:**
- **Linear scaling** up to 4 workers
- **Good scaling** up to 8 workers (diminishing returns start)
- **Limited scaling** beyond 16 workers (memory bandwidth bound)

**Recommendation:** Use 8 workers for best throughput/resource balance.

## Memory Usage

### Peak Memory by Framework

| Framework | Peak Memory | Per Sample | Notes |
|-----------|-------------|------------|-------|
| TurboLoader | 848 MB | 424 KB | Lock-free queues |
| TensorFlow | 1,245 MB | 622 KB | tf.data pipeline |
| PyTorch Cached | 2,104 MB | 1,052 KB | File caching overhead |
| PyTorch Optimized | 1,523 MB | 761 KB | Multiprocessing |
| PyTorch Naive | 1,834 MB | 917 KB | Inefficient batching |

**TurboLoader Advantages:**
- 42% less memory than TensorFlow
- 60% less memory than PyTorch Cached
- Efficient buffer reuse

### Memory Breakdown (TurboLoader)

```
Component           Memory    %
---------------------------------
Base code           200 MB   24%
Worker buffers      400 MB   47%
Output queue        150 MB   18%
TAR mmap            98 MB    12%
---------------------------------
Total               848 MB  100%
```

## Epoch Time Distribution

### TurboLoader (10,146 img/s)

```
Epoch 1: 0.197s  ┃██████████████████████████████┃
Epoch 2: 0.182s  ┃█████████████████████████     ┃ ← Warmed up
Epoch 3: 0.180s  ┃█████████████████████████     ┃
Std Dev: 0.005s  (2.8% variation)
```

### PyTorch Optimized (835 img/s)

```
Epoch 1: 2.678s  ┃██████████████████████████████┃
Epoch 2: 2.401s  ┃██████████████████████████    ┃
Epoch 3: 2.320s  ┃█████████████████████████     ┃
Std Dev: 0.152s  (6.2% variation)
```

**Stability:** TurboLoader has **2.2x lower variance** than PyTorch.

## CPU Utilization

### TurboLoader (8 workers)

```
Core Utilization:
Core 0:  ████████████████████  92%  (Main thread)
Core 1:  ███████████████████   95%  (Worker 1)
Core 2:  ███████████████████   95%  (Worker 2)
Core 3:  ███████████████████   94%  (Worker 3)
Core 4:  ███████████████████   95%  (Worker 4)
Core 5:  ███████████████████   94%  (Worker 5)
Core 6:  ███████████████████   95%  (Worker 6)
Core 7:  ███████████████████   94%  (Worker 7)
Core 8:  ███████████████████   95%  (Worker 8)

Average: 94% (excellent utilization)
```

### PyTorch (8 workers)

```
Core Utilization:
Core 0:  ████████              38%  (GIL contention)
Core 1:  ██████████            48%
Core 2:  █████████             45%
Core 3:  ██████████            47%
Core 4:  █████████             44%
Core 5:  ██████████            46%
Core 6:  █████████             45%
Core 7:  ██████████            48%
Core 8:  █████████             44%

Average: 45% (poor utilization due to GIL)
```

## Bottleneck Analysis

### TurboLoader Profiling

```
Component           Time    %
---------------------------------
JPEG Decode        45 ms   25%
Resize Transform   35 ms   19%
Normalize          20 ms   11%
Queue Operations   15 ms    8%
Memory Copy        10 ms    6%
Other Transforms   30 ms   17%
Python Overhead    25 ms   14%
---------------------------------
Total per 32-img  180 ms  100%
```

**Primary Bottleneck:** JPEG decoding (25%)

**Optimization Opportunities:**
1. GPU JPEG decode (nvJPEG) - Expected: 5-10x faster
2. AVX-512 transforms - Expected: 1.3x faster
3. Prefetching - Expected: 1.5x faster

## Hardware Scaling

### Different CPU Architectures

| Hardware | Throughput | SIMD | Notes |
|----------|------------|------|-------|
| Apple M4 Max (16c) | 10,146 img/s | NEON | Baseline |
| AMD Ryzen 9 7950X (16c) | 9,820 img/s | AVX2 | Similar perf |
| Intel i9-13900K (24c) | 11,500 img/s | AVX-512 | Wider SIMD |
| AWS c7g.4xlarge (16c) | 8,900 img/s | NEON | Graviton3 |
| Raspberry Pi 4 (4c) | 450 img/s | NEON | Limited cores |

### GPU Acceleration (Future)

**Planned for v1.0:**
- nvJPEG GPU decoding
- CUDA transform kernels
- Expected: **50K+ img/s** on RTX 4090

## Comparison with Other Libraries

### TurboLoader vs FFCV

| Feature | TurboLoader | FFCV |
|---------|-------------|------|
| Throughput | 10,146 img/s | ~15,000 img/s |
| Custom format | TAR (standard) | .beton (custom) |
| SIMD | AVX2/NEON | AVX2 only |
| PyTorch native | No (bindings) | Yes |
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

### TurboLoader vs NVIDIA DALI

| Feature | TurboLoader | DALI |
|---------|-------------|------|
| Throughput (CPU) | 10,146 img/s | ~8,000 img/s |
| GPU support | Planned | Yes (nvJPEG) |
| Dependencies | Minimal | CUDA required |
| Memory | 848 MB | 1,200+ MB |
| License | MIT | Apache 2.0 |

## Methodology

See [Methodology](methodology.md) for:
- Benchmark setup
- Dataset generation
- Measurement techniques
- Statistical analysis

## Interactive Dashboard

Run the interactive benchmark web app:

```bash
cd /Users/arnavjain/turboloader/web_app
streamlit run app.py
```

See [Web App README](../../web_app/README.md) for details.

## Reproducing Results

```bash
# Clone repository
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader

# Run benchmarks
cd benchmarks
python benchmark_comparison.py --dataset /path/to/dataset.tar --workers 8
```

## Questions?

- [Methodology](methodology.md) - How we benchmark
- [Memory Profiling](memory-profiling.md) - Memory analysis
- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
