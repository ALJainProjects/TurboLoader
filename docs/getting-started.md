# Getting Started with TurboLoader

This guide will help you get up and running with TurboLoader in minutes.

## Installation

### From PyPI (Recommended)

```bash
pip install turboloader
```

### From Source

```bash
git clone https://github.com/ALJainProjects/TurboLoader.git
cd TurboLoader
pip install -e .
```

## System Requirements

- **Python:** 3.8 or later
- **Compiler:** C++20 compatible (GCC 10+, Clang 12+, MSVC 19.29+)
- **OS:** macOS, Linux, or Windows

### Optional Dependencies

For best performance, install these libraries:

**macOS (Homebrew):**
```bash
brew install jpeg-turbo libpng libwebp
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libjpeg-turbo8-dev libpng-dev libwebp-dev
```

## Your First Data Loader

### Step 1: Prepare Your Data

TurboLoader works with TAR archives (WebDataset format):

```bash
# Create a TAR archive from images
tar -cf dataset.tar images/*.jpg
```

Or use WebDataset format:

```
dataset.tar/
  000000.jpg
  000000.json
  000001.jpg
  000001.json
  ...
```

### Step 2: Create a DataLoader

```python
import turboloader

# Create loader (similar to PyTorch)
loader = turboloader.DataLoader(
    data_path='dataset.tar',
    batch_size=32,
    num_workers=8
)

# Iterate over batches
for batch in loader:
    for sample in batch:
        print(f"Image shape: {sample['image'].shape}")
        print(f"Filename: {sample['filename']}")
```

### Step 3: Add Transforms

```python
import turboloader

# Create transforms
resize = turboloader.Resize(224, 224)
normalize = turboloader.ImageNetNormalize(to_float=True)
flip = turboloader.RandomHorizontalFlip(p=0.5)

# Apply in your training loop
for batch in loader:
    for sample in batch:
        img = sample['image']

        # Apply transforms
        img = resize.apply(img)
        img = flip.apply(img)
        img = normalize.apply(img)

        # img is now ready for training
```

## Basic Examples

### Example 1: ImageNet Training

```python
import turboloader
import torch
import torch.nn as nn

# Setup data loader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=128,
    num_workers=16
)

# Setup transforms
resize = turboloader.Resize(256, 256)
crop = turboloader.RandomCrop(224, 224)
flip = turboloader.RandomHorizontalFlip(p=0.5)
color_jitter = turboloader.ColorJitter(
    brightness=0.4,
    contrast=0.4,
    saturation=0.4,
    hue=0.1
)
to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW,
    normalize=True
)
normalize = turboloader.ImageNetNormalize(to_float=True)

# Model and optimizer
model = torch.hub.load('pytorch/vision', 'resnet50')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(100):
    for batch in loader:
        images = []
        labels = []

        for sample in batch:
            # Apply transforms
            img = sample['image']
            img = resize.apply(img)
            img = crop.apply(img)
            img = flip.apply(img)
            img = color_jitter.apply(img)
            img = to_tensor.apply(img)
            img = normalize.apply(img)

            images.append(torch.from_numpy(img))
            labels.append(sample['label'])

        # Stack into batch
        batch_images = torch.stack(images)
        batch_labels = torch.tensor(labels)

        # Forward pass
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### Example 2: AutoAugment

```python
import turboloader

# Create loader
loader = turboloader.DataLoader(
    'cifar10.tar',
    batch_size=64,
    num_workers=8
)

# Use AutoAugment with CIFAR10 policy
autoaugment = turboloader.AutoAugment(
    policy=turboloader.AutoAugmentPolicy.CIFAR10
)
to_tensor = turboloader.ToTensor()

# Training loop
for batch in loader:
    for sample in batch:
        img = autoaugment.apply(sample['image'])
        img = to_tensor.apply(img)
        # Train...
```

### Example 3: TensorFlow Integration

```python
import turboloader
import tensorflow as tf

# Create loader
loader = turboloader.DataLoader(
    'dataset.tar',
    batch_size=32,
    num_workers=8
)

# TensorFlow-format tensors
to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.TENSORFLOW_HWC,
    normalize=True
)
normalize = turboloader.ImageNetNormalize(to_float=True)

# Training loop
for batch in loader:
    images = []
    labels = []

    for sample in batch:
        img = to_tensor.apply(sample['image'])
        img = normalize.apply(img)
        images.append(tf.convert_to_tensor(img))
        labels.append(sample['label'])

    batch_images = tf.stack(images)
    batch_labels = tf.convert_to_tensor(labels)

    # Train with TensorFlow...
```

## Performance Tips

### 1. Choose Optimal Worker Count

```python
import os

# Rule of thumb: num_workers = 2 * num_CPU_cores
num_workers = 2 * os.cpu_count()

loader = turboloader.DataLoader(
    'data.tar',
    num_workers=num_workers,
    batch_size=32
)
```

### 2. Use SIMD-Optimized Interpolation

```python
# Bilinear is fastest for most cases
resize = turboloader.Resize(
    224, 224,
    interpolation=turboloader.InterpolationMode.BILINEAR  # 3.2x faster
)

# Lanczos for highest quality (slower)
resize_hq = turboloader.Resize(
    224, 224,
    interpolation=turboloader.InterpolationMode.LANCZOS
)
```

### 3. Enable Memory-Mapped I/O

TurboLoader automatically uses `mmap()` for TAR archives, but ensure your files are on fast storage (SSD recommended).

### 4. Batch Size Tuning

```python
# Larger batches = better throughput (up to a point)
# Test different sizes: 32, 64, 128, 256

loader_32 = turboloader.DataLoader('data.tar', batch_size=32, num_workers=8)
loader_64 = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)
loader_128 = turboloader.DataLoader('data.tar', batch_size=128, num_workers=8)
```

## Common Patterns

### Context Manager Usage

```python
with turboloader.DataLoader('data.tar', batch_size=32, num_workers=8) as loader:
    for batch in loader:
        # Process batch
        pass
# Loader automatically stopped and cleaned up
```

### Validation Pipeline

```python
# Validation uses deterministic transforms (no randomness)
val_loader = turboloader.DataLoader(
    'imagenet_val.tar',
    batch_size=64,
    num_workers=8
)

resize = turboloader.Resize(256, 256)
crop = turboloader.CenterCrop(224, 224)  # Center crop (not random)
to_tensor = turboloader.ToTensor()
normalize = turboloader.ImageNetNormalize(to_float=True)

# Validation loop
for batch in val_loader:
    for sample in batch:
        img = resize.apply(sample['image'])
        img = crop.apply(img)
        img = to_tensor.apply(img)
        img = normalize.apply(img)
        # Evaluate...
```

## Troubleshooting

### Issue: "Could not find libjpeg-turbo"

**Solution:** Install dependencies:

```bash
# macOS
brew install jpeg-turbo

# Ubuntu
sudo apt-get install libjpeg-turbo8-dev
```

### Issue: Low throughput

**Checklist:**
1. Increase `num_workers` (try 8-16)
2. Use SSD storage for TAR files
3. Ensure transforms are using SIMD (check with `turboloader.features()`)
4. Profile your training code (data loading might not be the bottleneck)

### Issue: Out of memory

**Solution:**
1. Reduce `batch_size`
2. Reduce `num_workers`
3. Check for memory leaks in your training code

## Next Steps

- [API Reference](api/index.md) - Complete API documentation
- [Advanced Usage](guides/advanced-usage.md) - Complex pipelines and patterns
- [PyTorch Integration](guides/pytorch-integration.md) - PyTorch-specific guide
- [Benchmarks](benchmarks/index.md) - Performance analysis

## Questions?

- [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- [Documentation](https://github.com/ALJainProjects/TurboLoader/tree/main/docs)
