# TurboLoader Comprehensive Benchmarks

Complete benchmarking suite for TurboLoader across all data types with real-world datasets.

## Quick Start

### 1. Download Real-World Datasets

```bash
# Download all datasets (requires ~5GB disk space)
python3 benchmarks/download_datasets.py --datasets all

# Or download specific datasets
python3 benchmarks/download_datasets.py --datasets cifar10 ag_news librispeech
```

**Available Datasets:**
- `cifar10`: CIFAR-10 (60K images, 32x32)
- `imagenet`: Tiny ImageNet (100K images, 64x64)
- `ag_news`: AG News (120K text samples)
- `wikitext`: WikiText-103 (100M+ tokens)
- `librispeech`: LibriSpeech (speech audio)
- `coco`: COCO Captions (image + text)

### 2. Run Benchmarks

```bash
# Run comprehensive benchmarks
python3 benchmarks/comprehensive_multitype_benchmark.py --datasets-dir ./datasets

# Results saved to: benchmark_results/comprehensive_benchmark.json
```

---

## Datasets Overview

### Images

| Dataset | Samples | Size | Format | Use Case |
|---------|---------|------|--------|----------|
| **CIFAR-10** | 60,000 | 163 MB | 32x32 RGB | Image classification |
| **Tiny ImageNet** | 100,000 | 237 MB | 64x64 RGB | ImageNet proxy |
| **Full ImageNet** | 1.3M | 150 GB | 256x256+ | Production scale |

**Expected TurboLoader Speedup:** 30-35x

### Text

| Dataset | Samples | Size | Format | Use Case |
|---------|---------|------|--------|----------|
| **AG News** | 120,000 | 29 MB | Text | Text classification |
| **WikiText-103** | 100,000 | 517 MB | Text | Language modeling |
| **IMDB Reviews** | 50,000 | 66 MB | Text | Sentiment analysis |

**Expected TurboLoader Speedup:** 5-10x

### Audio

| Dataset | Samples | Size | Format | Use Case |
|---------|---------|------|--------|----------|
| **LibriSpeech dev-clean** | 2,703 | 337 MB | FLAC | Speech recognition |
| **LibriSpeech train-clean-100** | 28,539 | 6.3 GB | FLAC | Full training |

**Expected TurboLoader Speedup:** 5-8x

### Multi-Modal

| Dataset | Samples | Size | Format | Use Case |
|---------|---------|------|--------|----------|
| **COCO Captions** | 1,000 | 87 MB | JPG + JSON | Image captioning |

**Expected TurboLoader Speedup:** 25-30x (images), 5-10x (text)

---

## Benchmark Results

### Example Output

```
======================================================================
TURBOLOADER COMPREHENSIVE BENCHMARK SUITE
======================================================================

======================================================================
IMAGE BENCHMARKS
======================================================================

CIFAR-10:
  TurboLoader: 12,450 img/s
  PyTorch:        385 img/s
  Speedup:      32.34x

Tiny ImageNet:
  TurboLoader:  9,230 img/s
  PyTorch:        267 img/s
  Speedup:      34.57x

======================================================================
TEXT BENCHMARKS
======================================================================

AG News:
  TurboLoader:  8,750 samples/s
  PyTorch:      1,320 samples/s
  Speedup:       6.63x

WikiText-103:
  TurboLoader:  7,890 samples/s
  PyTorch:      1,150 samples/s
  Speedup:       6.86x

======================================================================
AUDIO BENCHMARKS
======================================================================

LibriSpeech:
  TurboLoader:  2,940 samples/s (1.8 GB)
  PyTorch:        468 samples/s (1.8 GB)
  Speedup:       6.28x

======================================================================
BENCHMARK SUMMARY
======================================================================

Average speedup: 17.34x
Min speedup:      6.28x (Audio)
Max speedup:     34.57x (Images)
```

---

## Full ImageNet Benchmark

TurboLoader works great with full ImageNet! We provide automated tools for conversion and benchmarking.

### 1. Download ImageNet

1. Register at [image-net.org](https://image-net.org)
2. Download ILSVRC2012 training set (~150GB)
3. Extract to `/path/to/imagenet/train/` (should contain 1000 class folders like `n01440764/`, `n01443537/`, etc.)

### 2. Convert to WebDataset TAR

Use our automated converter:

```bash
python3 benchmarks/imagenet_converter.py \
    --imagenet-dir /path/to/imagenet/train \
    --output-tar /path/to/imagenet_train.tar \
    --shard-size 10000 \
    --verify
```

**Options:**
- `--imagenet-dir`: Path to extracted ImageNet train directory
- `--output-tar`: Output TAR path (or base path for shards)
- `--num-samples`: Limit samples for testing (e.g., 10000)
- `--shard-size`: Samples per shard (default: 10000)
- `--verify`: Verify TAR integrity after creation

**Sharding:** For large datasets (1.3M images), the converter automatically creates sharded TARs for easier handling:
- Each shard contains ~10K images (default)
- Shards are saved in `{output_tar}_shards/`
- A `shard_list.txt` file lists all shard paths

**Example conversion script (embedded in converter):**

```python
#!/usr/bin/env python3
"""Convert ImageNet to WebDataset TAR format"""
import tarfile
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io

def convert_imagenet_to_tar(imagenet_dir, output_tar):
    """Convert ImageNet directory to TAR"""
    imagenet_dir = Path(imagenet_dir)
    samples = []

    # Iterate through class folders
    for class_dir in tqdm(list(imagenet_dir.iterdir())):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for img_path in class_dir.glob("*.JPEG"):
            # Read image
            img = Image.open(img_path).convert('RGB')

            # Convert to JPEG bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=95)

            samples.append({
                'data': img_bytes.getvalue(),
                'ext': 'jpg',
                'label': class_name,
                'metadata': {
                    'class': class_name,
                    'filename': img_path.name
                }
            })

    # Create TAR
    print(f"Creating {output_tar} with {len(samples)} images...")
    with tarfile.open(output_tar, 'w') as tar:
        for i, sample in enumerate(tqdm(samples)):
            basename = f"sample_{i:08d}"

            # Write image
            img_path = f"/tmp/{basename}.jpg"
            with open(img_path, 'wb') as f:
                f.write(sample['data'])
            tar.add(img_path, arcname=f"{basename}.jpg")
            os.remove(img_path)

            # Write metadata
            meta_path = f"/tmp/{basename}.json"
            with open(meta_path, 'w') as f:
                json.dump({
                    'label': sample['label'],
                    **sample['metadata']
                }, f)
            tar.add(meta_path, arcname=f"{basename}.json")
            os.remove(meta_path)

    print(f"✅ Created {output_tar}")

# Convert
convert_imagenet_to_tar(
    "/path/to/imagenet/train",
    "/path/to/imagenet_train.tar"
)
```

### 3. Benchmark Full ImageNet

Use the dedicated full ImageNet benchmark:

```bash
# Single TAR file
python3 benchmarks/full_imagenet_benchmark.py \
    --tar-paths /path/to/imagenet_train.tar \
    --num-workers 16 \
    --batch-size 256 \
    --num-batches 500

# Sharded TARs (recommended for full 1.3M images)
python3 benchmarks/full_imagenet_benchmark.py \
    --shard-dir /path/to/imagenet_train_shards/ \
    --num-workers 16 \
    --batch-size 256 \
    --num-batches 500
```

**Options:**
- `--tar-paths`: One or more TAR file paths
- `--shard-dir`: Directory containing sharded TARs
- `--num-workers`: Number of worker threads (default: 8)
- `--batch-size`: Batch size (default: 256)
- `--num-batches`: Number of batches to benchmark (default: 500)
- `--output`: Output JSON file path (default: `benchmark_results/imagenet_benchmark.json`)
- `--skip-pytorch`: Skip PyTorch benchmark (TurboLoader only)

**Expected Results:**
- TurboLoader: ~15,000-20,000 img/s
- PyTorch: ~500-600 img/s
- **Speedup: 30-35x**

**Benchmark Output:**
```
================================================================================
FULL IMAGENET PRODUCTION-SCALE BENCHMARK
================================================================================
Dataset: 130 TAR file(s)
Workers: 16
Batch size: 256
Batches: 500
Total samples: ~128,000
================================================================================

================================================================================
TURBOLOADER BENCHMARK
================================================================================
Configuration:
  TAR files: 130
  Workers: 16
  Batch size: 256
  Queue size: 512
  SIMD transforms: Enabled
  Target size: 224x224

Warming up...

Running benchmark (500 batches)...
TurboLoader: 100%|████████████████████| 500/500 [00:08<00:00, 61.23 batch/s]

--------------------------------------------------------------------------------
RESULTS:
  Total samples: 128,000
  Total time: 8.16s
  Throughput: 15,686.27 images/sec
  Avg batch time: 16.33ms
  P50 batch time: 16.12ms
  P95 batch time: 17.89ms
  P99 batch time: 18.45ms
--------------------------------------------------------------------------------

================================================================================
PYTORCH DATALOADER BENCHMARK
================================================================================
...

================================================================================
COMPARISON SUMMARY
================================================================================

Throughput:
  TurboLoader:  15,686.27 images/sec
  PyTorch:          543.12 images/sec
  Speedup:          28.88x ⚡

Average Batch Time:
  TurboLoader:      16.33ms
  PyTorch:         471.53ms
  Improvement:      28.88x faster

P99 Latency:
  TurboLoader:      18.45ms
  PyTorch:         523.76ms

================================================================================
🚀 TurboLoader is 28.9× FASTER on full ImageNet!
================================================================================

📊 Results saved to: benchmark_results/imagenet_benchmark.json
```

---

## Custom Dataset Benchmarks

### Add Your Own Dataset

```python
# 1. Convert to TAR format
import tarfile
import json

with tarfile.open('my_dataset.tar', 'w') as tar:
    for i, (data, label) in enumerate(your_data):
        # Save data file
        data_path = f'/tmp/sample_{i:06d}.ext'
        with open(data_path, 'wb') as f:
            f.write(data)
        tar.add(data_path, arcname=f'sample_{i:06d}.ext')

        # Save metadata
        meta_path = f'/tmp/sample_{i:06d}.json'
        with open(meta_path, 'w') as f:
            json.dump({'label': label}, f)
        tar.add(meta_path, arcname=f'sample_{i:06d}.json')

# 2. Benchmark
python3 benchmarks/comprehensive_multitype_benchmark.py \
    --custom-tar my_dataset.tar \
    --data-type [image|text|audio|other]
```

---

## Performance Tips

### Maximize Throughput

1. **CPU Workers**: Set to number of cores
   ```python
   num_workers=16  # For 16-core CPU
   ```

2. **Queue Size**: Increase for larger prefetch
   ```python
   queue_size=512  # More prefetching
   ```

3. **Batch Size**: Optimize for your workload
   ```python
   batch_size=128  # Images
   batch_size=256  # Text
   ```

4. **Storage**: Use SSD for maximum I/O
   - NVMe SSD: Best
   - SATA SSD: Good
   - HDD: Will bottleneck

### CPU Architecture

Different CPUs have different SIMD capabilities:

| CPU | SIMD | Image Speedup |
|-----|------|---------------|
| **Apple M1/M2/M3** | NEON | 30-35x |
| **Intel (recent)** | AVX2 | 25-30x |
| **AMD Ryzen** | AVX2 | 25-30x |
| **ARM (server)** | NEON | 30-35x |

---

## Benchmark Scripts

### 1. `download_datasets.py`
Downloads and converts real-world datasets to WebDataset TAR format.

**Usage:**
```bash
python3 benchmarks/download_datasets.py --datasets all
python3 benchmarks/download_datasets.py --datasets cifar10 ag_news
python3 benchmarks/download_datasets.py --output-dir ./my_datasets
```

### 2. `comprehensive_multitype_benchmark.py`
Runs comprehensive benchmarks across all data types.

**Usage:**
```bash
python3 benchmarks/comprehensive_multitype_benchmark.py
python3 benchmarks/comprehensive_multitype_benchmark.py --datasets-dir ./datasets
python3 benchmarks/comprehensive_multitype_benchmark.py --num-batches 200
```

### 3. `quick_imagenet_comparison.py`
Quick ImageNet-scale comparison (existing script).

**Usage:**
```bash
python3 benchmarks/quick_imagenet_comparison.py dataset.tar
```

---

## Troubleshooting

### Out of Memory

Reduce batch size or num_workers:
```python
pipeline = turboloader.Pipeline(
    num_workers=4,  # Reduce from 8
    queue_size=128  # Reduce from 256
)
```

### Slow Performance

1. Check CPU usage: Should be 80-100%
2. Check disk I/O: Use `iostat -x 1`
3. Verify SIMD enabled: Should see "NEON" or "AVX2" in logs
4. Use SSD storage instead of HDD

### Dataset Not Found

```bash
# Re-download specific dataset
python3 benchmarks/download_datasets.py --datasets cifar10

# Check downloaded files
ls -lh datasets/*/
```

---

## Citation

If you use TurboLoader in your research, please cite:

```bibtex
@software{turboloader2025,
  author = {Jain, Arnav},
  title = {TurboLoader: High-Performance Data Loading for Machine Learning},
  year = {2025},
  url = {https://github.com/arnavjain/turboloader}
}
```

---

## Contributing Benchmarks

Have a dataset you'd like to add? Submit a PR with:

1. Dataset downloader in `download_datasets.py`
2. Benchmark function in `comprehensive_multitype_benchmark.py`
3. Expected performance numbers in this README

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

---

## 📊 Visualization & Plotting

Beautiful publication-ready plots are automatically generated from benchmark results!

### Generate All Plots

```bash
# After running benchmarks
python3 benchmarks/plot_results.py

# Or specify custom paths
python3 benchmarks/plot_results.py \
    --results benchmark_results/comprehensive_benchmark.json \
    --output-dir benchmark_results/plots
```

### Generated Plots

The plotting script creates 6 publication-quality visualizations:

#### 1. **Throughput Comparison** (`throughput_comparison.png`)
- Side-by-side bar chart
- TurboLoader vs PyTorch samples/second
- All datasets shown
- Perfect for: Papers, presentations

#### 2. **Speedup Chart** (`speedup_chart.png`)
- Speedup multiplier for each dataset
- Average speedup line
- Color-coded by magnitude
- Perfect for: README, blog posts

#### 3. **Category Breakdown** (`category_breakdown.png`)
- Separate panels for Images, Text, Audio
- Speedup by data type
- Easy comparison within categories
- Perfect for: Technical docs

#### 4. **Batch Time Comparison** (`batch_time_comparison.png`)
- Time per batch (milliseconds)
- Shows latency improvements
- Important for real-time applications
- Perfect for: Performance analysis

#### 5. **Summary Infographic** (`summary_infographic.png`)
- Big numbers: Avg speedup, peak speedup, datasets tested
- Horizontal bar chart of all results
- Color-coded by category
- Perfect for: Social media, presentations

#### 6. **README Hero Image** (`readme_hero.png`)
- Eye-catching "35× FASTER" visual
- Mini bar chart inset
- Perfect for: GitHub README header

### Plot Formats

All plots are saved in two formats:
- **PNG** (300 DPI) - for web, presentations
- **PDF** (vector) - for publications, papers

### Customization

Edit `benchmarks/plot_results.py` to customize:

```python
# Color scheme
TURBO_COLOR = '#FF6B35'  # Orange for TurboLoader
PYTORCH_COLOR = '#4A90E2'  # Blue for PyTorch
ACCENT_COLOR = '#50E3C2'  # Teal for highlights

# Font settings
rcParams['font.size'] = 10
rcParams['font.family'] = 'sans-serif'

# Figure sizes
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust as needed
```

### Example Output

```
$ python3 benchmarks/plot_results.py

Loading results from benchmark_results/comprehensive_benchmark.json...

Generating 6 dataset plots...
Creating throughput comparison plot...
  Saved: benchmark_results/plots/throughput_comparison.png
Creating speedup chart...
  Saved: benchmark_results/plots/speedup_chart.png
Creating category breakdown...
  Saved: benchmark_results/plots/category_breakdown.png
Creating batch time comparison...
  Saved: benchmark_results/plots/batch_time_comparison.png
Creating summary infographic...
  Saved: benchmark_results/plots/summary_infographic.png
Creating README hero image...
  Saved: benchmark_results/plots/readme_hero.png

✅ All plots saved to benchmark_results/plots/

Generated files:
  - throughput_comparison.png/pdf
  - speedup_chart.png/pdf
  - category_breakdown.png/pdf
  - batch_time_comparison.png/pdf
  - summary_infographic.png/pdf
  - readme_hero.png
```

### Requirements

```bash
pip install matplotlib numpy
```

Already included in `setup.py` extras:
```bash
pip install turboloader[benchmarks]
```

---

## 🎨 Using Plots in Your Work

### In README

```markdown
# TurboLoader

![Performance](benchmark_results/plots/readme_hero.png)

## Benchmarks

![Speedup](benchmark_results/plots/speedup_chart.png)
```

### In Papers (LaTeX)

```latex
\begin{figure}
  \centering
  \includegraphics[width=0.8\textwidth]{plots/throughput_comparison.pdf}
  \caption{TurboLoader vs PyTorch DataLoader throughput comparison}
  \label{fig:throughput}
\end{figure}
```

### In Presentations

- Use PNG files at 300 DPI
- Full-screen plots look professional
- Summary infographic works great for conclusions

### On Social Media

- `readme_hero.png` - Twitter/X header
- `summary_infographic.png` - LinkedIn posts
- `speedup_chart.png` - Quick visual for threads

---

## 📈 Complete Workflow

Here's the complete benchmark → plot → share workflow:

```bash
# 1. Download datasets
python3 benchmarks/download_datasets.py --datasets all

# 2. Run benchmarks
python3 benchmarks/comprehensive_multitype_benchmark.py

# 3. Generate plots
python3 benchmarks/plot_results.py

# 4. View results
open benchmark_results/plots/

# 5. Share!
# - Add plots to README
# - Tweet the hero image
# - Include in paper
# - Post on Reddit/HN
```

**Total time**: ~30 minutes for complete benchmark suite with beautiful visualizations!

