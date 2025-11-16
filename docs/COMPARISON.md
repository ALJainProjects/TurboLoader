# Framework Comparison Guide


> **Note**: Performance claims in this documentation are based on preliminary benchmarks on synthetic datasets. 
> Actual performance will vary based on hardware, dataset characteristics, and workload. 
> We recommend running benchmarks on your specific use case.



When to use TurboLoader vs other ML data loading frameworks.

---

## Framework Comparison Matrix

| Feature | TurboLoader | PyTorch | TensorFlow | FFCV | DALI |
|---------|-------------|---------|------------|------|------|
| **Data Loading (CPU, img/s)** | 11,628 | 400 | 9,477 | 31,278* | ~12,000 |
| **Data Loading (GPU, img/s)** | **45,000** | ❌ | ❌ | ❌ | ~48,000* |
| **Speedup vs PyTorch** | **27.8x (CPU)** | 1.0x | 22.7x | 75x* | 30x |
| **GPU Speedup** | **8.5x** | N/A | N/A | N/A | 9.0x |
| **Memory Efficiency** | ✅ Shared | ❌ Duplicated | ✅ Shared | ✅ | ✅ |
| **Setup Complexity** | Low | Low | Low | High | High |
| **TAR Streaming** | ✅ Yes | ✅ Yes | ❌ Extract | ❌ No | ❌ No |
| **Cloud Storage (S3)** | ✅ Yes | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Multi-format** | ✅ JPEG/PNG/WebP | ✅ All | ✅ All | ❌ JPEG only | ✅ All |
| **GPU Decode** | ✅ nvJPEG | ❌ CPU | ❌ CPU | ❌ CPU | ✅ Yes |
| **Distributed Training** | ✅ NCCL/Gloo | ✅ DDP | ✅ MultiWorker | ❌ No | ❌ No |
| **Pre-processing** | ❌ Not yet | ✅ Torchvision | ✅ tf.image | ✅ Custom | ✅ Yes |

\* Published benchmarks from papers

---

## Use TurboLoader When

✅ **Large datasets** (100K+ images)
✅ **TAR archives** (WebDataset format)
✅ **Cloud storage streaming** (S3/GCS)
✅ **Data loading is bottleneck** (30%+ of training time)
✅ **Memory-constrained environments**

---

## Use FFCV When

✅ **Maximum data loading performance**
✅ **Willing to pre-process dataset** (.beton format)
✅ **JPEG-only datasets**
✅ **GPU training with high-end GPUs**

**Note**: FFCV requires dataset conversion, TurboLoader works on raw TAR files.

---

## Use PyTorch DataLoader When

✅ **Small datasets** (< 10K images, fully cached)
✅ **Rapid prototyping**
✅ **Complex data augmentation pipelines**

---

## Use TensorFlow tf.data When

✅ **TensorFlow ecosystem**
✅ **Production deployment with TF Serving**
✅ **TPU training**

---

## Use NVIDIA DALI When

✅ **NVIDIA GPUs available**
✅ **GPU-accelerated augmentation needed**
✅ **Video/multi-modal data**

---

## Performance Summary

**Data Loading Only**:
- TurboLoader: high throughput
- PyTorch: high throughput (27.8x slower)
- TensorFlow: high throughput (1.2x slower)
- FFCV: high throughput (significantly faster, but requires preprocessing)

**End-to-End Training**:
- TurboLoader provides 1.2-significant speedup on large datasets
- On small datasets (<10K images), minimal difference (compute-bound)

---

## See Also

- [Benchmarks](../benchmarks/README.md) - Detailed comparisons
- [Performance](PERFORMANCE.md) - Tuning guide
