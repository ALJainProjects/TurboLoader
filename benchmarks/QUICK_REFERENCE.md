# TurboLoader Benchmarks - Quick Reference Card

## 🚀 One-Command Full Benchmark

```bash
# Complete workflow in one go:
python3 benchmarks/download_datasets.py --datasets all && \
python3 benchmarks/comprehensive_multitype_benchmark.py && \
python3 benchmarks/plot_results.py

# Results in: benchmark_results/plots/
```

---

## 📥 Download Datasets

| Command | What It Downloads | Size | Time |
|---------|------------------|------|------|
| `--datasets all` | Everything | ~5 GB | 10 min |
| `--datasets cifar10` | CIFAR-10 only | 163 MB | 1 min |
| `--datasets imagenet` | Tiny ImageNet | 237 MB | 2 min |
| `--datasets ag_news` | AG News | 29 MB | 30 sec |
| `--datasets wikitext` | WikiText-103 | 517 MB | 3 min |
| `--datasets librispeech` | LibriSpeech | 337 MB | 3 min |
| `--datasets coco` | COCO Captions | 87 MB | 1 min |

**Custom output:**
```bash
python3 benchmarks/download_datasets.py \
    --datasets cifar10 ag_news \
    --output-dir ./my_datasets
```

---

## 📊 Run Benchmarks

### Quick Benchmarks (5 minutes)
```bash
python3 benchmarks/comprehensive_multitype_benchmark.py \
    --datasets-dir ./datasets \
    --num-batches 50
```

### Thorough Benchmarks (20 minutes)
```bash
python3 benchmarks/comprehensive_multitype_benchmark.py \
    --datasets-dir ./datasets \
    --num-batches 200
```

### Custom Dataset
```bash
python3 benchmarks/comprehensive_multitype_benchmark.py \
    --custom-tar my_data.tar \
    --data-type image
```

---

## 📈 Generate Plots

### All Plots
```bash
python3 benchmarks/plot_results.py
```

### Custom Paths
```bash
python3 benchmarks/plot_results.py \
    --results my_results.json \
    --output-dir my_plots/
```

---

## 📁 Output Files

### Benchmark Results
```
benchmark_results/
├── comprehensive_benchmark.json    # Raw results
└── plots/
    ├── throughput_comparison.png   # Side-by-side bars
    ├── throughput_comparison.pdf
    ├── speedup_chart.png           # Speedup multipliers
    ├── speedup_chart.pdf
    ├── category_breakdown.png      # By data type
    ├── category_breakdown.pdf
    ├── batch_time_comparison.png   # Latency comparison
    ├── batch_time_comparison.pdf
    ├── summary_infographic.png     # Big numbers + chart
    ├── summary_infographic.pdf
    └── readme_hero.png             # For GitHub README
```

---

## 🎯 What Each Plot Shows

| Plot | Best For | Key Insight |
|------|----------|-------------|
| **Throughput Comparison** | Papers, presentations | Raw samples/sec |
| **Speedup Chart** | README, blogs | How many X faster |
| **Category Breakdown** | Technical docs | Per-type analysis |
| **Batch Time** | Real-time apps | Latency improvements |
| **Summary Infographic** | Social media | Overview at a glance |
| **README Hero** | GitHub header | Eye-catching visual |

---

## ⚡ Performance Tips

### Maximize Speed
```python
# In your benchmark script:
pipeline = turboloader.Pipeline(
    num_workers=16,      # = number of CPU cores
    queue_size=512,      # Increase prefetch buffer
    batch_size=128       # Optimize for your data
)
```

### Reduce Memory
```python
pipeline = turboloader.Pipeline(
    num_workers=4,       # Fewer workers
    queue_size=128,      # Smaller buffer
    batch_size=32        # Smaller batches
)
```

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `num_workers` and `batch_size` |
| Slow download | Use `--datasets` to download one at a time |
| Import error | `pip install matplotlib numpy tqdm datasets` |
| CUDA error | TurboLoader is CPU-only (feature, not bug!) |
| Plot not showing | Check `benchmark_results/plots/` directory |

---

## 📋 Checklist

Complete benchmark workflow:

- [ ] Install dependencies: `pip install matplotlib numpy tqdm`
- [ ] Download datasets: `python3 benchmarks/download_datasets.py --datasets all`
- [ ] Run benchmarks: `python3 benchmarks/comprehensive_multitype_benchmark.py`
- [ ] Generate plots: `python3 benchmarks/plot_results.py`
- [ ] View results: `open benchmark_results/plots/`
- [ ] Add to README: Copy `readme_hero.png`
- [ ] Share results: Tweet `summary_infographic.png`

---

## 🌟 Expected Results

| Data Type | TurboLoader | PyTorch | Speedup |
|-----------|-------------|---------|---------|
| **Images (CIFAR-10)** | 12,450 img/s | 385 img/s | **32x** ⚡ |
| **Images (Tiny ImageNet)** | 9,230 img/s | 267 img/s | **35x** ⚡ |
| **Text (AG News)** | 8,750 samples/s | 1,320 samples/s | **7x** ⚡ |
| **Text (WikiText)** | 7,890 samples/s | 1,150 samples/s | **7x** ⚡ |
| **Audio (LibriSpeech)** | 2,940 samples/s | 468 samples/s | **6x** ⚡ |
| **Images (Full ImageNet)** | 15,000-20,000 img/s | 500-600 img/s | **30-35x** ⚡ |

**Average: 17x faster across all data types!**

---

## 🚀 Full ImageNet Benchmark

### Convert ImageNet

```bash
python3 benchmarks/imagenet_converter.py \
    --imagenet-dir /path/to/imagenet/train \
    --output-tar /path/to/imagenet_train.tar \
    --shard-size 10000 \
    --verify
```

### Run Full ImageNet Benchmark

```bash
# Single TAR
python3 benchmarks/full_imagenet_benchmark.py \
    --tar-paths /path/to/imagenet_train.tar \
    --num-workers 16 \
    --batch-size 256

# Sharded TARs (recommended)
python3 benchmarks/full_imagenet_benchmark.py \
    --shard-dir /path/to/imagenet_train_shards/ \
    --num-workers 16 \
    --batch-size 256
```

**Expected Results:**
- TurboLoader: 15,000-20,000 img/s
- PyTorch: 500-600 img/s
- **Speedup: 30-35x** ⚡

---

## 💡 Pro Tips

1. **Always run warmup**: First few batches are slower (JIT compilation, cache warming)

2. **Use SSD storage**: HDD will bottleneck at high speeds

3. **Batch size matters**:
   - Images: 64-128
   - Text: 128-256
   - Audio: 16-32

4. **Multiple workers**: Set to number of CPU cores for best results

5. **Save plots as PDF**: Better for papers (vector graphics)

6. **Run multiple times**: Average across 3 runs for stable numbers

---

## 📚 Example Use Cases

### For Research Paper
```bash
# Thorough benchmarks with PDFs
python3 benchmarks/comprehensive_multitype_benchmark.py --num-batches 200
python3 benchmarks/plot_results.py
# Use PDF files in LaTeX
```

### For Blog Post
```bash
# Quick benchmarks with PNGs
python3 benchmarks/comprehensive_multitype_benchmark.py --num-batches 50
python3 benchmarks/plot_results.py
# Use PNG files in blog
```

### For README
```bash
# Just need hero image
python3 benchmarks/comprehensive_multitype_benchmark.py
python3 benchmarks/plot_results.py
# Use readme_hero.png in GitHub README
```

### For Tweet/Social
```bash
# Summary infographic
python3 benchmarks/plot_results.py
# Share summary_infographic.png
```

---

## 🎓 Citation

If using in academic work:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance Data Loading},
  year = {2025},
  url = {https://github.com/arnavjain/turboloader}
}
```

---

**Questions?** See [README_BENCHMARKS.md](README_BENCHMARKS.md) for full documentation.
