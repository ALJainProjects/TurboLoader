# Performance Tuning Guide


> **Note**: Performance claims in this documentation are based on preliminary benchmarks on synthetic datasets. 
> Actual performance will vary based on hardware, dataset characteristics, and workload. 
> We recommend running benchmarks on your specific use case.



Optimization tips for maximum TurboLoader performance.

---

## Worker Configuration

### Optimal Worker Count

**Rule of thumb**: `num_workers = num_cpu_cores`

```python
import multiprocessing
num_workers = multiprocessing.cpu_count()

pipeline = turboloader.Pipeline(
    tar_paths=['/data/train.tar'],
    num_workers=num_workers,
    decode_jpeg=True
)
```

### Finding Optimal Workers

Run the scaling benchmark:
```bash
python benchmarks/scaling_benchmark.py /tmp/benchmark_10k.tar
```

Expected results:
- 1 → 2 workers: ~significant speedup
- 1 → 4 workers: ~significant speedup
- 1 → 8 workers: ~significant speedup (diminishing returns)

---

## Batch Size Selection

Larger batches = higher throughput (up to a point)

| Batch Size | Throughput | Notes |
|------------|------------|-------|
| 8 | high throughput | Underutilized |
| 32 | high throughput | Good ✅ |
| 64 | high throughput | Optimal |
| 128 | high throughput | Marginal gains |

**Recommendation**: 32-64 for balanced throughput/latency

---

## Dataset Size Impact

| Dataset Size | TurboLoader Benefit | Why |
|--------------|---------------------|-----|
| < 1K images | Minimal (compute-bound) | Data fully cached |
| 1K-10K images | Moderate (1.2-1.5x) | Partial caching |
| 10K-100K images | High (1.5-2.0x) | Disk I/O matters |
| 100K+ images | Maximum (2.0-2.5x) | Data loading bottleneck |

---

## Common Bottlenecks

### 1. Too Few Workers

**Symptom**: Low CPU utilization, low throughput

**Fix**: Increase `num_workers`

### 2. Too Many Workers

**Symptom**: Context switching overhead, diminishing returns

**Fix**: Use scaling benchmark to find sweet spot

### 3. Small Batches

**Symptom**: High overhead per sample

**Fix**: Increase batch size to 32-64

### 4. Data Fully Cached

**Symptom**: TurboLoader only marginally faster

**Explanation**: On small datasets, compute is bottleneck, not data loading

---

## Memory Optimization

TurboLoader uses ~56 MB per worker (shared memory).

PyTorch uses ~300 MB per worker (duplicated memory).

**Memory savings**: 81% less memory at 8 workers

---

## See Also

- [Benchmarks](../benchmarks/README.md) - Performance comparisons
- [Architecture](ARCHITECTURE.md) - How it works
