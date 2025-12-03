# Quick Start Guide

Get up and running with TurboLoader in 5 minutes.

## Installation

```bash
pip install turboloader
```

Verify installation:

```python
import turboloader
print(turboloader.__version__)  # Should print "2.7.0" or later
```

---

## Basic Usage

### Load Data from TAR Archive

```python
import turboloader

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet.tar',  # Path to TAR archive
    batch_size=32,    # Samples per batch
    num_workers=4,    # Parallel worker threads
    shuffle=True      # Shuffle data
)

# Iterate over batches
for batch in loader:
    print(f"Batch size: {len(batch)}")

    # Access first sample
    sample = batch[0]
    image = sample['image']  # NumPy array (H, W, C)
    label = sample.get('label', 0)

    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    break  # Just show first batch
```

---

## Data Augmentation

Apply transforms for data augmentation:

```python
import turboloader

# Create transform pipeline
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),              # Resize to 256x256
    turboloader.RandomCrop(224, 224),          # Random crop 224x224
    turboloader.RandomHorizontalFlip(0.5),     # Flip with 50% probability
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1), # Color augmentation
    turboloader.ImageNetNormalize()            # Normalize to ImageNet stats
])

# Create DataLoader
loader = turboloader.DataLoader('imagenet.tar', batch_size=32, num_workers=4)

# Apply transforms
for batch in loader:
    for sample in batch:
        # Apply all transforms in pipeline
        image = transforms.apply(sample['image'])
        print(f"Transformed image shape: {image.shape}")
    break
```

---

## PyTorch Integration

Use with PyTorch for training:

```python
import turboloader
import torch
import torch.nn as nn

# Create model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10)
)

# Create transforms
transforms = turboloader.Compose([
    turboloader.Resize(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()  # Convert to tensor format (C, H, W)
])

# Create DataLoader
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

# Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch in loader:
        # Convert batch to tensors
        images = []
        labels = []

        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(torch.from_numpy(img).float())
            labels.append(sample.get('label', 0))

        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        break  # Just show one batch
    break
```

---

## TensorFlow Integration

Use with TensorFlow/Keras:

```python
import turboloader
import tensorflow as tf
from tensorflow import keras

# Create model
model = keras.Sequential([
    keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create transforms
transforms = turboloader.Compose([
    turboloader.Resize(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor(turboloader.TensorFormat.TENSORFLOW_HWC)
])

# Create DataLoader
loader = turboloader.DataLoader('data.tar', batch_size=64, num_workers=8)

# Generator for TensorFlow
def data_generator():
    for batch in loader:
        images = []
        labels = []
        for sample in batch:
            img = transforms.apply(sample['image'])
            images.append(tf.convert_to_tensor(img, dtype=tf.float32))
            labels.append(sample.get('label', 0))

        yield tf.stack(images), tf.constant(labels)

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Train
model.fit(dataset, epochs=10, steps_per_epoch=100)
```

---

## Available Transforms

TurboLoader provides 19 SIMD-accelerated transforms:

### Geometric Transforms
- `Resize(width, height)` - Resize image
- `CenterCrop(width, height)` - Center crop
- `RandomCrop(width, height)` - Random crop
- `RandomHorizontalFlip(p)` - Horizontal flip
- `RandomVerticalFlip(p)` - Vertical flip
- `RandomRotation(degrees)` - Random rotation
- `RandomAffine(...)` - Affine transformation
- `RandomPerspective(...)` - Perspective transformation
- `Pad(...)` - Padding

### Color Transforms
- `ColorJitter(brightness, contrast, saturation, hue)` - Color augmentation
- `Normalize(mean, std)` - Normalization
- `ImageNetNormalize()` - ImageNet normalization
- `Grayscale()` - Convert to grayscale

### Augmentation Transforms
- `RandomErasing(...)` - Random erasing (Cutout)
- `GaussianBlur(...)` - Gaussian blur
- `RandomPosterize(...)` - Posterization
- `RandomSolarize(...)` - Solarization
- `AutoAugment(policy)` - Learned augmentation

### Tensor Conversion
- `ToTensor(format)` - Convert to tensor format (PyTorch/TensorFlow)

See [Transforms API](api/transforms.md) for complete documentation.

---

## Performance Tips

### 1. Optimal Worker Count

Benchmark different worker counts:

```python
import time

for num_workers in [1, 2, 4, 8, 16]:
    loader = turboloader.DataLoader(
        'data.tar',
        batch_size=32,
        num_workers=num_workers
    )

    start = time.time()
    count = 0
    for batch in loader:
        count += len(batch)
        if count >= 1000:
            break

    elapsed = time.time() - start
    throughput = count / elapsed
    print(f"Workers: {num_workers}, Throughput: {throughput:.1f} img/s")
```

### 2. Use TBL Format

Convert TAR to TBL for faster loading:

```python
# Convert TAR to TBL
writer = turboloader.TblWriterV2('output.tbl', compression=True)

reader = turboloader.DataLoader('input.tar', batch_size=1, num_workers=1)

for batch in reader:
    for sample in batch:
        writer.add_sample(
            data=sample['image'],
            format=turboloader.SampleFormat.JPEG,
            metadata={'label': sample.get('label', 0)}
        )

writer.finalize()

# Load from TBL (40-60% faster)
loader = turboloader.DataLoader('output.tbl', batch_size=64, num_workers=8)
```

### 3. Use FastDataLoader with Caching (v2.7.0)

For multi-epoch training, use FastDataLoader with `cache_decoded=True` for 2.6x faster total training:

```python
import turboloader

# FastDataLoader with decoded tensor caching
loader = turboloader.FastDataLoader(
    'imagenet.tar',
    batch_size=64,
    num_workers=8,
    cache_decoded=True  # Cache decoded arrays in memory
)

for epoch in range(10):
    for images, metadata in loader:
        # First epoch: ~25K img/s (decode from TAR)
        # Subsequent epochs: 100K+ img/s (cache hit)
        train_step(images)

    if loader.cache_populated:
        print(f"Cache size: {loader.cache_size_mb:.1f} MB")

# Clear cache when done
loader.clear_cache()
```

**Performance:**
- First epoch: Standard decode throughput
- Cached epochs: Memory iteration speed (100K+ img/s)
- Total time for 5 epochs: 2.6x faster than TensorFlow `.cache()`

### 4. Use Larger Batch Sizes

Larger batches = better throughput (if GPU memory allows):

```python
# Good for large GPUs
loader = turboloader.DataLoader('data.tar', batch_size=256, num_workers=8)
```

---

## Next Steps

Now that you have the basics:

1. **Explore Examples**: Check [examples/](../examples/) for complete scripts
   - [ImageNet ResNet50 Training](../examples/imagenet_resnet50.py)
   - [PyTorch DDP](../examples/distributed_ddp.py)
   - [PyTorch Lightning](../examples/pytorch_lightning_example.py)

2. **Read Integration Guides**:
   - [PyTorch Integration](guides/pytorch-integration.md)
   - [TensorFlow Integration](guides/tensorflow-integration.md)

3. **API Reference**: [API Documentation](api/index.md)

4. **Benchmarks**: See [Benchmarks](benchmarks/index.md) for performance analysis

---

## Getting Help

- **Troubleshooting**: [Troubleshooting Guide](TROUBLESHOOTING.md)
- **Issues**: [GitHub Issues](https://github.com/ALJainProjects/TurboLoader/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ALJainProjects/TurboLoader/discussions)
- **Verify Installation**: Run `python scripts/verify_installation.py`
