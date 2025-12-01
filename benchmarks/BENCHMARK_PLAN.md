# TurboLoader Benchmarking Plan

## Overview

This document outlines how to benchmark TurboLoader against other popular data loading libraries to demonstrate performance advantages.

## Competitors to Benchmark Against

### 1. PyTorch DataLoader (torch.utils.data.DataLoader)
- The standard PyTorch data loading solution
- Most widely used, baseline for comparison

### 2. NVIDIA DALI (nvidia.dali)
- GPU-accelerated data loading pipeline
- Industry standard for high-performance training

### 3. FFCV (ffcv)
- Fast data loading library from MosaicML
- Uses memory-mapped files and custom format

### 4. WebDataset (webdataset)
- Streaming data loading from TAR archives
- Good for distributed training

### 5. tf.data (TensorFlow)
- TensorFlow's data loading API
- Optimized for TensorFlow workflows

### 6. Albumentations
- Fast image augmentation library
- For transform-specific benchmarks

---

## Benchmark Categories

### Category 1: Raw Throughput (Images/Second)

**What we measure:**
- Images loaded and decoded per second
- With varying batch sizes (32, 64, 128, 256)
- With varying worker counts (1, 2, 4, 8, 16)

**Datasets:**
- ImageNet-1K (1.28M images, ~140GB)
- COCO 2017 (118K train images, ~18GB)
- Synthetic dataset (100K generated images)

**Expected TurboLoader advantages:**
- Lock-free SPSC queues
- SIMD-accelerated decoding
- Zero-copy where possible

### Category 2: Transform Pipeline Performance

**What we measure:**
- Time to apply standard augmentation pipelines
- Per-transform breakdown

**Standard Pipeline:**
```python
# Typical ImageNet training pipeline
transforms = [
    Resize(256, 256),
    RandomCrop(224, 224),
    RandomHorizontalFlip(0.5),
    ColorJitter(0.4, 0.4, 0.4, 0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
]
```

**Expected TurboLoader advantages:**
- SIMD-accelerated transforms (AVX2/NEON)
- Efficient pipeline composition
- Reduced memory allocations

### Category 3: Memory Efficiency

**What we measure:**
- Peak memory usage during loading
- Memory usage over time
- Memory efficiency (useful data / total memory)

**Expected TurboLoader advantages:**
- Smart Batching reduces padding overhead by 15-25%
- Streaming architecture with constant memory
- Memory-mapped I/O for TBL format

### Category 4: End-to-End Training Throughput

**What we measure:**
- Samples processed per second during actual training
- GPU utilization during training
- Data loading bottleneck analysis

**Models to test:**
- ResNet-50 (standard benchmark)
- EfficientNet-B0 (efficient architecture)
- ViT-B/16 (transformer, different batch patterns)

### Category 5: Distributed Training Scalability

**What we measure:**
- Throughput scaling with multiple GPUs
- Throughput scaling with multiple nodes
- Sharding efficiency

**Configurations:**
- 1, 2, 4, 8 GPUs (single node)
- 2, 4, 8 nodes (multi-node)

### Category 6: Format-Specific Benchmarks

**TBL v2 Format:**
- Conversion throughput (TAR → TBL)
- Loading throughput (TBL vs TAR vs raw files)
- Space efficiency (compression ratios)

**Remote Storage:**
- HTTP/HTTPS throughput
- S3 throughput
- GCS throughput

---

## Benchmark Implementation

### Directory Structure
```
benchmarks/
├── BENCHMARK_PLAN.md
├── run_all_benchmarks.py      # Master script
├── results/                    # Output directory
│   ├── throughput/
│   ├── transforms/
│   ├── memory/
│   └── reports/
├── datasets/
│   ├── prepare_imagenet.py
│   ├── prepare_coco.py
│   └── generate_synthetic.py
├── throughput/
│   ├── bench_turboloader.py
│   ├── bench_pytorch.py
│   ├── bench_dali.py
│   ├── bench_ffcv.py
│   └── bench_webdataset.py
├── transforms/
│   ├── bench_turboloader_transforms.py
│   ├── bench_torchvision_transforms.py
│   ├── bench_albumentations.py
│   └── bench_dali_transforms.py
├── memory/
│   ├── bench_memory_usage.py
│   └── profile_memory.py
├── e2e_training/
│   ├── train_resnet50.py
│   ├── train_efficientnet.py
│   └── train_vit.py
└── visualization/
    ├── plot_results.py
    └── generate_report.py
```

### Benchmark Script Template

```python
#!/usr/bin/env python3
"""
Benchmark: [Name]
Compares: TurboLoader vs [Competitor]
Metric: [What we measure]
"""

import time
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict
import psutil
import gc

@dataclass
class BenchmarkResult:
    library: str
    metric: str
    value: float
    unit: str
    config: Dict
    timestamp: str

def warmup(loader, num_batches=10):
    """Warmup to ensure fair comparison"""
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

def benchmark_throughput(loader, num_batches=100):
    """Measure throughput in images/second"""
    gc.collect()

    total_images = 0
    start = time.perf_counter()

    for i, batch in enumerate(loader):
        total_images += len(batch)
        if i >= num_batches:
            break

    elapsed = time.perf_counter() - start
    throughput = total_images / elapsed

    return throughput

def benchmark_memory(loader, num_batches=50):
    """Measure peak memory usage"""
    gc.collect()

    process = psutil.Process()
    baseline_memory = process.memory_info().rss
    peak_memory = baseline_memory

    for i, batch in enumerate(loader):
        current_memory = process.memory_info().rss
        peak_memory = max(peak_memory, current_memory)
        if i >= num_batches:
            break

    return (peak_memory - baseline_memory) / (1024 * 1024)  # MB

def run_benchmark(config):
    """Run complete benchmark suite"""
    results = []

    # ... benchmark implementation ...

    return results

def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON"""
    with open(output_path, 'w') as f:
        json.dump([r.__dict__ for r in results], f, indent=2)

if __name__ == '__main__':
    config = {
        'dataset': 'imagenet',
        'batch_size': 64,
        'num_workers': 8,
        'num_batches': 100,
    }

    results = run_benchmark(config)
    save_results(results, 'results/benchmark_results.json')
```

---

## Expected Results

Based on previous benchmarks and architecture analysis:

### Throughput (ImageNet, batch_size=64, 8 workers)

| Library | Images/sec | vs PyTorch |
|---------|------------|------------|
| TurboLoader | ~21,000 | 12x faster |
| NVIDIA DALI | ~18,000 | 10x faster |
| FFCV | ~15,000 | 8x faster |
| WebDataset | ~8,000 | 4x faster |
| PyTorch DataLoader | ~1,800 | baseline |

### Transform Performance (per-image, 224x224 RGB)

| Transform | TurboLoader | torchvision | Speedup |
|-----------|-------------|-------------|---------|
| Resize | 0.15ms | 0.48ms | 3.2x |
| Normalize | 0.02ms | 0.08ms | 4.0x |
| HorizontalFlip | 0.01ms | 0.03ms | 3.0x |
| ColorJitter | 0.08ms | 0.25ms | 3.1x |
| GaussianBlur | 0.12ms | 0.35ms | 2.9x |

### Memory Efficiency

| Library | Peak Memory (MB) | Memory/Image |
|---------|------------------|--------------|
| TurboLoader | 2,100 | 32.8 KB |
| PyTorch | 4,500 | 70.3 KB |
| FFCV | 2,800 | 43.8 KB |

---

## Running the Benchmarks

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision
pip install nvidia-dali-cuda110  # For DALI
pip install ffcv                  # For FFCV
pip install webdataset           # For WebDataset
pip install albumentations       # For transform comparison
pip install psutil matplotlib pandas seaborn

# Prepare dataset
python benchmarks/datasets/prepare_imagenet.py --path /data/imagenet
# OR generate synthetic data
python benchmarks/datasets/generate_synthetic.py --num-images 10000
```

### Run All Benchmarks

```bash
# Quick benchmark (synthetic data, fewer iterations)
python benchmarks/run_all_benchmarks.py --quick

# Full benchmark suite (requires ImageNet)
python benchmarks/run_all_benchmarks.py --dataset /data/imagenet --full

# Specific benchmark
python benchmarks/throughput/bench_turboloader.py --batch-size 64 --workers 8
```

### Generate Report

```bash
python benchmarks/visualization/generate_report.py --results-dir results/
```

---

## Benchmark Configurations

### Hardware Configurations to Test

1. **Consumer Workstation**
   - CPU: AMD Ryzen 9 / Intel i9
   - GPU: RTX 3090 / RTX 4090
   - RAM: 64GB
   - Storage: NVMe SSD

2. **Apple Silicon**
   - M1/M2/M3/M4 Max
   - Unified memory: 32-128GB
   - Tests NEON optimizations

3. **Cloud Instance**
   - AWS p4d.24xlarge (8x A100)
   - GCP a2-highgpu-8g
   - Tests distributed training

### Software Configurations

| Config | Batch Size | Workers | Description |
|--------|------------|---------|-------------|
| Small | 32 | 4 | Typical laptop |
| Medium | 64 | 8 | Workstation |
| Large | 128 | 16 | Server |
| XLarge | 256 | 32 | Multi-GPU |

---

## Metrics to Report

### Primary Metrics
1. **Throughput**: Images processed per second
2. **Latency**: Time to first batch, per-batch latency
3. **Memory**: Peak RSS, GPU memory if applicable
4. **CPU Utilization**: During data loading

### Secondary Metrics
1. **Scaling Efficiency**: Throughput vs worker count
2. **GPU Utilization**: During end-to-end training
3. **I/O Bandwidth**: Disk read throughput
4. **Cache Efficiency**: L1/L2/L3 cache hit rates

---

## Visualization

### Charts to Generate

1. **Bar Chart**: Throughput comparison across libraries
2. **Line Chart**: Throughput vs batch size
3. **Line Chart**: Throughput vs worker count (scaling)
4. **Stacked Bar**: Transform pipeline breakdown
5. **Memory Timeline**: Memory usage over time
6. **Heatmap**: Performance across configurations

### Example Output

```
===============================================================================
                      TurboLoader Benchmark Report
===============================================================================

Test Configuration:
  Dataset: ImageNet-1K (1.28M images)
  Hardware: Apple M4 Max (48GB RAM)
  Date: 2024-XX-XX

-------------------------------------------------------------------------------
                         THROUGHPUT COMPARISON
-------------------------------------------------------------------------------

                    Images/sec    vs PyTorch    Memory (MB)
TurboLoader            21,035        12.0x          2,100
NVIDIA DALI            18,200        10.4x          3,200
FFCV                   15,100         8.6x          2,800
WebDataset              8,050         4.6x          1,900
PyTorch DataLoader      1,753         1.0x          4,500

-------------------------------------------------------------------------------
                         TRANSFORM PERFORMANCE
-------------------------------------------------------------------------------

Transform          TurboLoader    torchvision    Speedup
Resize               0.15 ms        0.48 ms       3.2x
Normalize            0.02 ms        0.08 ms       4.0x
HorizontalFlip       0.01 ms        0.03 ms       3.0x
ColorJitter          0.08 ms        0.25 ms       3.1x
Full Pipeline        0.35 ms        1.10 ms       3.1x

===============================================================================
```

---

## Next Steps

1. **Create benchmark scripts** for each category
2. **Prepare datasets** (ImageNet subset, COCO, synthetic)
3. **Run benchmarks** on target hardware
4. **Generate visualizations** and report
5. **Publish results** in documentation
