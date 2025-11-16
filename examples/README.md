# TurboLoader Examples

This directory contains practical examples demonstrating how to use TurboLoader in real-world scenarios.

## Examples Overview

### 1. Simple ImageNet Loading (`simple_imagenet.py`)

**Purpose:** Minimal example showing basic TurboLoader usage.

**What it demonstrates:**
- Basic configuration and setup
- Loading images from TAR files
- Converting to PyTorch tensors
- Measuring throughput

**Usage:**
```bash
python3 examples/simple_imagenet.py
```

**Requirements:**
- ImageNet TAR file (modify `tar_path` in the script)
- Built TurboLoader library

---

### 2. ResNet-50 Training (`resnet50_training.py`)

**Purpose:** Complete training pipeline for ResNet-50 on ImageNet.

**What it demonstrates:**
- Full training loop with TurboLoader
- Mixed precision training (AMP)
- Learning rate scheduling
- Checkpointing and resuming
- Accuracy metrics (Top-1, Top-5)
- Performance monitoring

**Usage:**
```bash
# Basic training
python3 examples/resnet50_training.py \
    --train-tar /data/imagenet_train.tar \
    --batch-size 256 \
    --workers 16 \
    --epochs 90

# Resume from checkpoint
python3 examples/resnet50_training.py \
    --train-tar /data/imagenet_train.tar \
    --resume checkpoints/checkpoint_epoch_30.pth.tar

# Quick test run (few batches)
python3 examples/resnet50_training.py \
    --train-tar /data/imagenet_train.tar \
    --batch-size 128 \
    --batches-per-epoch 100 \
    --epochs 1
```

**Key features:**
- Automatic checkpointing every 10 epochs
- Mixed precision training for speed
- Progress logging during training
- Customizable hyperparameters

**Requirements:**
- ImageNet TAR file
- PyTorch with CUDA support (recommended)
- 32GB+ RAM for batch size 256

---

### 3. DataLoader Comparison (`compare_dataloaders.py`)

**Purpose:** Side-by-side comparison of TurboLoader vs PyTorch DataLoader.

**What it demonstrates:**
- Direct performance comparison
- Same preprocessing pipeline
- Speedup calculation
- Migration guide (shows code for both)

**Usage:**
```bash
# Full comparison (both TurboLoader and PyTorch)
python3 examples/compare_dataloaders.py \
    /data/imagenet_train.tar \
    --workers 8 \
    --batch-size 256 \
    --num-batches 100

# TurboLoader only (skip PyTorch)
python3 examples/compare_dataloaders.py \
    /data/imagenet_train.tar \
    --workers 16 \
    --batch-size 256 \
    --skip-pytorch
```

**Expected output:**
```
================================================================================
COMPARISON SUMMARY
================================================================================

Throughput:
  PyTorch:         587.34 images/sec
  TurboLoader:  18,456.73 images/sec
  Speedup:          31.42x faster

Average Batch Time:
  PyTorch:         435.89ms
  TurboLoader:      13.87ms
  Improvement:      31.42x faster

================================================================================
TurboLoader is 31.4x FASTER!
================================================================================
```

**Requirements:**
- ImageNet TAR file (or any TAR with images)
- Both PyTorch and TurboLoader installed

---

## Common Setup

### Prerequisites

1. **Build TurboLoader:**
```bash
cd /Users/arnavjain/turboloader
mkdir -p build && cd build
cmake ..
make -j8
cd ..
```

2. **Install Python dependencies:**
```bash
pip install torch torchvision Pillow numpy
```

3. **Prepare data:**
- For full ImageNet: See [IMAGENET_GUIDE.md](../benchmarks/IMAGENET_GUIDE.md)
- For quick testing: Use any TAR file with JPEG images

### Creating a Test Dataset

If you don't have ImageNet, create a small test dataset:

```bash
# Create test images
python3 << 'EOF'
from PIL import Image
import tarfile
import random

# Generate 100 random images
for i in range(100):
    img = Image.new('RGB', (256, 256))
    pixels = img.load()
    for x in range(256):
        for y in range(256):
            pixels[x, y] = (random.randint(0, 255),
                           random.randint(0, 255),
                           random.randint(0, 255))
    img.save(f'/tmp/test_img_{i:03d}.jpg')

# Create TAR file
with tarfile.open('/tmp/test_dataset.tar', 'w') as tar:
    for i in range(100):
        tar.add(f'/tmp/test_img_{i:03d}.jpg', arcname=f'images/img_{i:03d}.jpg')

print("Created /tmp/test_dataset.tar with 100 test images")
EOF

# Now use /tmp/test_dataset.tar in the examples
```

---

## Example Modifications

### Adjust Worker Count

Based on your CPU cores:
```python
config.num_workers = 16  # Set to your CPU core count
```

Check CPU cores:
```bash
# macOS
sysctl -n hw.ncpu

# Linux
nproc
```

### Custom Transforms

Modify transform configuration:
```python
transform_config = turboloader.TransformConfig()
transform_config.target_width = 384  # Larger images
transform_config.target_height = 384
transform_config.resize_mode = "bicubic"  # Better quality
transform_config.normalize = True
transform_config.mean = [0.5, 0.5, 0.5]  # Custom normalization
transform_config.std = [0.5, 0.5, 0.5]
```

### Different Batch Sizes

For memory-constrained environments:
```python
batch_size = 64  # Smaller batches
config.queue_size = 256  # Smaller queue
```

For high-throughput benchmarking:
```python
batch_size = 512  # Larger batches
config.queue_size = 1024  # Larger queue
```

---

## Troubleshooting

### Out of Memory

**Symptoms:** Process killed, "Out of memory" error

**Solutions:**
```python
# Reduce batch size
batch_size = 64

# Reduce workers
config.num_workers = 4

# Reduce queue size
config.queue_size = 128
```

### Low Throughput

**Check CPU usage:** Should be 80-100%
```bash
top  # or htop
```

**Increase workers:**
```python
config.num_workers = 16  # Match your CPU cores
```

**Verify SIMD is enabled:**
```python
config.enable_simd_transforms = True
```

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'turboloader'`

**Solution:**
```bash
# Ensure build/python is in path or install package
export PYTHONPATH=/Users/arnavjain/turboloader/build/python:$PYTHONPATH

# Or install package
pip install -e .
```

---

## Performance Tips

1. **Use local SSD storage** - Network storage may bottleneck
2. **Match workers to CPU cores** - Usually optimal
3. **Enable SIMD transforms** - 4-8x speedup
4. **Use larger batches** - Better throughput, more memory
5. **Pin memory** - Faster GPU transfers (if using GPU)

---

## Next Steps

After running the examples:

1. **Integrate into your project:**
   - Replace PyTorch DataLoader with TurboLoader
   - Adjust configuration for your use case
   - Benchmark on your data

2. **Optimize performance:**
   - Run `benchmarks/detailed_profiling.py` for metrics
   - Tune worker count and batch size
   - Monitor CPU/memory usage

3. **Contribute:**
   - Share your results in [BENCHMARKS.md](../BENCHMARKS.md)
   - Report issues on GitHub
   - Submit improvements

---

## Questions?

- See [README.md](../README.md) for general documentation
- See [ARCHITECTURE.md](../ARCHITECTURE.md) for technical details
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines
- Open an issue on GitHub for support
