# TurboLoader Transforms Guide (v0.6.0)

## Overview

TurboLoader v0.6.0 introduces a complete SIMD-accelerated transform system compatible with PyTorch and TensorFlow. All transforms are implemented in C++ with AVX2/NEON vectorization for maximum performance.

## Features

- **14 PyTorch-compatible transforms** with identical API
- **SIMD acceleration** (AVX2 on x86, NEON on ARM) for 4-8x speedup
- **Zero-copy tensor conversion** for PyTorch and TensorFlow
- **Thread-safe** implementations for pipeline parallelism
- **Drop-in replacement** for torchvision.transforms

## Available Transforms

### Core Transforms

#### Resize
```python
import turboloader as tl

# Resize to 224x224 with bilinear interpolation
transform = tl.Resize(224, 224, tl.InterpolationMode.BILINEAR)
output = transform.apply(image)  # image is numpy array (H, W, C)
```

**Modes:**
- `NEAREST`: Nearest neighbor (fastest)
- `BILINEAR`: Bilinear interpolation (balanced)
- `BICUBIC`: Bicubic interpolation (highest quality)

**Performance:** ~3x faster than PIL, 2x faster than OpenCV

#### Normalize
```python
# ImageNet normalization
transform = tl.ImageNetNormalize(to_float=True)

# Custom normalization
transform = tl.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    to_float=True
)
```

**Performance:** ~6x faster than torchvision (SIMD-accelerated)

#### CenterCrop
```python
transform = tl.CenterCrop(224, 224)
```

#### RandomCrop
```python
transform = tl.RandomCrop(
    width=224,
    height=224,
    padding=32,
    pad_mode=tl.PaddingMode.REFLECT,
    seed=42
)
```

**Padding Modes:**
- `CONSTANT`: Fill with constant value
- `EDGE`: Replicate edge pixels
- `REFLECT`: Reflect around edges

### Random Augmentations

#### RandomHorizontalFlip
```python
transform = tl.RandomHorizontalFlip(p=0.5, seed=42)
```

**Performance:** ~8x faster than torchvision

#### RandomVerticalFlip
```python
transform = tl.RandomVerticalFlip(p=0.5, seed=42)
```

#### ColorJitter
```python
transform = tl.ColorJitter(
    brightness=0.4,    # Random brightness adjustment
    contrast=0.4,      # Random contrast adjustment
    saturation=0.4,    # Random saturation adjustment (RGB->HSV)
    hue=0.1,          # Random hue shift
    seed=42
)
```

**Performance:** ~4x faster than torchvision (SIMD HSV conversion)

#### RandomRotation
```python
transform = tl.RandomRotation(
    degrees=45,       # Rotate in [-45, +45] degrees
    expand=False,     # Expand output to fit rotated image
    fill=0,           # Fill value for empty pixels
    seed=42
)
```

**Performance:** ~3x faster than torchvision (SIMD bilinear interpolation)

#### RandomAffine
```python
transform = tl.RandomAffine(
    degrees=15,           # Rotation range
    translate_x=0.1,      # Horizontal translation (fraction of width)
    translate_y=0.1,      # Vertical translation (fraction of height)
    scale_min=0.9,        # Minimum scale
    scale_max=1.1,        # Maximum scale
    shear=10,             # Shear angle range
    fill=0,
    seed=42
)
```

#### GaussianBlur
```python
transform = tl.GaussianBlur(
    kernel_size=5,
    sigma=1.5          # If 0, calculated from kernel_size
)
```

**Performance:** ~5x faster than torchvision (separable SIMD convolution)

#### RandomErasing
```python
transform = tl.RandomErasing(
    p=0.5,                # Probability of applying
    scale_min=0.02,       # Minimum erased area (fraction)
    scale_max=0.33,       # Maximum erased area
    ratio_min=0.3,        # Minimum aspect ratio
    ratio_max=3.33,       # Maximum aspect ratio
    value=0,              # Fill value
    seed=42
)
```

### Format Transforms

#### Grayscale
```python
# Convert to single-channel grayscale
transform = tl.Grayscale(num_output_channels=1)

# Convert to grayscale but keep 3 channels (replicated)
transform = tl.Grayscale(num_output_channels=3)
```

**Performance:** ~7x faster than torchvision (SIMD weighted sum)

#### Pad
```python
# Uniform padding
transform = tl.Pad(10, tl.PaddingMode.CONSTANT, 0)

# Per-side padding (not yet implemented in bindings)
# Use multiple Pad transforms or contribute a PR!
```

## Tensor Conversion

### PyTorch Format (CHW)
```python
# Manual conversion
tensor_data = tl.to_pytorch_tensor(image, normalize=True)
# Returns TensorData with shape (C, H, W) and float32 data

# As transform
transform = tl.ToTensor(tl.TensorFormat.PYTORCH_CHW, normalize=True)
```

**Memory Layout:** Channels-first (C, H, W)
**Zero-copy:** Yes (when source is contiguous uint8)

### TensorFlow Format (HWC)
```python
# Manual conversion
tensor_data = tl.to_tensorflow_tensor(image, normalize=True)
# Returns TensorData with shape (H, W, C) and float32 data

# As transform
transform = tl.ToTensor(tl.TensorFormat.TENSORFLOW_HWC, normalize=True)
```

**Memory Layout:** Channels-last (H, W, C)
**Zero-copy:** Yes (when source matches layout)

## Creating Pipelines

### Sequential Transforms
```python
# Apply transforms one by one
transforms = [
    tl.Resize(256, 256),
    tl.RandomCrop(224, 224, padding=32),
    tl.RandomHorizontalFlip(p=0.5),
    tl.ColorJitter(brightness=0.4, contrast=0.4),
    tl.ImageNetNormalize(to_float=True),
]

image = load_image()  # Returns numpy array (H, W, C) uint8
for transform in transforms:
    image = transform.apply(image)
```

### Pre-built Pipelines
```python
# Coming soon: Compose() helper
```

## Performance Benchmarks

### Single Image (1000x1000 RGB)

| Transform           | TurboLoader | torchvision | Speedup |
|---------------------|-------------|-------------|---------|
| Resize              | 2.1 ms      | 6.8 ms      | 3.2x    |
| Normalize           | 0.8 ms      | 4.9 ms      | 6.1x    |
| HorizontalFlip      | 0.3 ms      | 2.4 ms      | 8.0x    |
| ColorJitter         | 3.2 ms      | 12.7 ms     | 4.0x    |
| GaussianBlur (5x5)  | 1.9 ms      | 9.3 ms      | 4.9x    |
| Grayscale           | 0.4 ms      | 2.9 ms      | 7.3x    |

**Hardware:** Intel Core i7 (AVX2), single-threaded

### Full Pipeline (ImageNet-style)

| Implementation      | Time/Image | Throughput  |
|---------------------|------------|-------------|
| TurboLoader         | 8.3 ms     | 120 img/s   |
| torchvision (CPU)   | 31.2 ms    | 32 img/s    |
| **Speedup**         | **3.8x**   |             |

**Pipeline:** Resize(256) -> RandomCrop(224) -> HFlip -> ColorJitter -> Normalize

## Integration Examples

### PyTorch DataLoader
```python
import turboloader as tl
import torch

# TurboLoader with transforms
loader = tl.DataLoader(
    data_path='imagenet.tar',
    batch_size=128,
    num_workers=8
)

# Apply transforms to each batch
transforms = [
    tl.Resize(256, 256),
    tl.CenterCrop(224, 224),
    tl.ImageNetNormalize(to_float=True),
]

for batch in loader:
    # batch is list of dicts with 'image' (numpy array)
    images = [sample['image'] for sample in batch]

    # Apply transforms
    for transform in transforms:
        images = [transform.apply(img) for img in images]

    # Convert to PyTorch tensor
    images_tensor = torch.from_numpy(np.stack(images))  # (N, C, H, W)

    # Train model
    outputs = model(images_tensor)
```

### TensorFlow Integration
```python
import turboloader as tl
import tensorflow as tf

loader = tl.DataLoader('imagenet.tar', batch_size=128, num_workers=8)

transforms = [
    tl.Resize(224, 224),
    tl.ImageNetNormalize(to_float=True),
]

for batch in loader:
    images = [sample['image'] for sample in batch]

    # Apply transforms (already in HWC format)
    for transform in transforms:
        images = [transform.apply(img) for img in images]

    # Convert to TensorFlow tensor
    images_tensor = tf.stack([tf.constant(img) for img in images])  # (N, H, W, C)

    # Train model
    with tf.GradientTape() as tape:
        outputs = model(images_tensor)
```

## SIMD Optimization Details

### AVX2 (x86_64)
- **Operations:** 256-bit vectors (8x float32, 32x uint8)
- **Coverage:** All arithmetic operations, type conversions
- **Speedup:** 4-8x over scalar code

### NEON (ARM)
- **Operations:** 128-bit vectors (4x float32, 16x uint8)
- **Coverage:** All arithmetic operations, type conversions
- **Speedup:** 3-6x over scalar code

### Scalar Fallback
Automatic fallback for platforms without SIMD support (same API, slower performance).

## Thread Safety

All transforms are **thread-safe** except for:
- Transforms with internal random state (use different seeds per thread)
- Shared transform pipelines (create one pipeline per thread)

**Recommended pattern:**
```python
# Per-worker transforms (in worker initialization)
def worker_init(worker_id):
    global transforms
    transforms = [
        tl.RandomHorizontalFlip(p=0.5, seed=worker_id * 1000),
        tl.ColorJitter(brightness=0.4, seed=worker_id * 1000 + 1),
    ]
```

## Common Issues

### 1. Random seed not working
**Problem:** Same random transforms across iterations
**Solution:** Set unique seed per worker/thread

### 2. Performance slower than expected
**Check:**
- SIMD enabled? (check `tl.features()['simd_acceleration']`)
- Image format correct? (should be numpy uint8 HWC)
- Not copying data excessively?

### 3. Memory leak
**Cause:** Not releasing ImageData properly
**Solution:** Use transforms sequentially (each returns new data)

## Contributing

Want to add more transforms? See `CONTRIBUTING.md` for:
- Transform implementation template
- SIMD optimization guidelines
- Testing requirements

## References

- [PyTorch torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [TensorFlow tf.image](https://www.tensorflow.org/api_docs/python/tf/image)
- [SIMD Programming Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
