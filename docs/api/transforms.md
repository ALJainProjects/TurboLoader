# Transforms API Reference

Complete API reference for all TurboLoader transforms with SIMD acceleration (AVX2/AVX-512/NEON).

## Quick Links

- [Geometric Transforms](#geometric-transforms) - Resize, Crop, Rotate, Flip
- [Color Transforms](#color-transforms) - ColorJitter, Normalize, Grayscale
- [Augmentation](#augmentation-transforms) - RandomErasing, Blur, AutoAugment
- [Tensor Conversion](#tensor-conversion) - ToTensor for PyTorch/TensorFlow
- [Transform Composition](#transform-composition) - Compose multiple transforms

---

## Transform Basics

All transforms share a common interface:

```python
import turboloader
import numpy as np

# Create transform
transform = turboloader.Resize(224, 224)

# Apply to image (H, W, C NumPy array)
image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
output = transform.apply(image)
```

### Transform Composition

Chain multiple transforms together:

```python
transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
    turboloader.ImageNetNormalize()
])

# Apply all transforms
output = transforms.apply(image)
```

---

## Geometric Transforms

### `Resize`

Resize image to specified dimensions with SIMD-accelerated interpolation.

**Constructor:**
```python
turboloader.Resize(width, height, interpolation=turboloader.InterpolationMode.BILINEAR)
```

**Parameters:**
- `width` (int): Target width in pixels
- `height` (int): Target height in pixels
- `interpolation` (InterpolationMode, optional): Interpolation method
  - `NEAREST`: Nearest neighbor (fastest)
  - `BILINEAR`: Bilinear interpolation (default, good quality/speed)
  - `BICUBIC`: Bicubic interpolation (highest quality, slower)
  - `LANCZOS`: Lanczos resampling (best for downscaling)

**Returns:** NumPy array of shape `(height, width, channels)`

**Example:**
```python
import turboloader

# Resize to 224x224 with bilinear interpolation
resize = turboloader.Resize(224, 224)
output = resize.apply(image)  # (224, 224, 3)

# High-quality resize with Lanczos
resize_hq = turboloader.Resize(
    224, 224,
    turboloader.InterpolationMode.LANCZOS
)
output_hq = resize_hq.apply(image)
```

**Performance:** ~336,000 images/sec with AVX2 SIMD

---

### `CenterCrop`

Extract center region of specified size.

**Constructor:**
```python
turboloader.CenterCrop(width, height)
```

**Parameters:**
- `width` (int): Crop width in pixels
- `height` (int): Crop height in pixels

**Returns:** NumPy array of shape `(height, width, channels)`

**Example:**
```python
# Extract 224x224 center crop
crop = turboloader.CenterCrop(224, 224)
output = crop.apply(image)  # (224, 224, 3)
```

**Note:** If requested size exceeds image dimensions, image is returned unchanged.

---

### `RandomCrop`

Extract random region of specified size (for training).

**Constructor:**
```python
turboloader.RandomCrop(width, height)
```

**Parameters:**
- `width` (int): Crop width in pixels
- `height` (int): Crop height in pixels

**Returns:** NumPy array of shape `(height, width, channels)`

**Example:**
```python
# Random 224x224 crop for data augmentation
crop = turboloader.RandomCrop(224, 224)
output = crop.apply(image)  # (224, 224, 3)
```

**Note:** Crop location is randomized on each call. Use `CenterCrop` for deterministic validation.

---

### `RandomHorizontalFlip`

Randomly flip image horizontally with given probability.

**Constructor:**
```python
turboloader.RandomHorizontalFlip(probability)
```

**Parameters:**
- `probability` (float): Probability of flipping (0.0 to 1.0)

**Returns:** NumPy array (flipped or original)

**Example:**
```python
# Flip 50% of images
flip = turboloader.RandomHorizontalFlip(0.5)
output = flip.apply(image)
```

**Performance:** ~310,000 images/sec with SIMD

---

### `RandomVerticalFlip`

Randomly flip image vertically with given probability.

**Constructor:**
```python
turboloader.RandomVerticalFlip(probability)
```

**Parameters:**
- `probability` (float): Probability of flipping (0.0 to 1.0)

**Returns:** NumPy array (flipped or original)

**Example:**
```python
# Flip 25% of images vertically
flip = turboloader.RandomVerticalFlip(0.25)
output = flip.apply(image)
```

---

### `RandomRotation`

Rotate image by random angle within range.

**Constructor:**
```python
turboloader.RandomRotation(degrees)
```

**Parameters:**
- `degrees` (float): Maximum rotation angle in degrees (±degrees)

**Returns:** NumPy array (rotated image)

**Example:**
```python
# Rotate by ±15 degrees
rotate = turboloader.RandomRotation(15.0)
output = rotate.apply(image)
```

**Note:** Uses bilinear interpolation. Rotated regions filled with black pixels.

---

### `RandomAffine`

Apply random affine transformation (rotation, translation, scale, shear).

**Constructor:**
```python
turboloader.RandomAffine(
    degrees,
    translate_x=0.0,
    translate_y=0.0,
    scale_min=1.0,
    scale_max=1.0,
    shear=0.0
)
```

**Parameters:**
- `degrees` (float): Maximum rotation in degrees (±degrees)
- `translate_x` (float, optional): Max horizontal translation (fraction of width)
- `translate_y` (float, optional): Max vertical translation (fraction of height)
- `scale_min` (float, optional): Minimum scale factor
- `scale_max` (float, optional): Maximum scale factor
- `shear` (float, optional): Shear angle in degrees

**Returns:** NumPy array (transformed image)

**Example:**
```python
# Complex affine augmentation
affine = turboloader.RandomAffine(
    degrees=30.0,
    translate_x=0.1,
    translate_y=0.1,
    scale_min=0.8,
    scale_max=1.2,
    shear=10.0
)
output = affine.apply(image)
```

---

### `RandomPerspective`

Apply random perspective transformation.

**Constructor:**
```python
turboloader.RandomPerspective(distortion_scale, probability)
```

**Parameters:**
- `distortion_scale` (float): Distortion magnitude (0.0 to 1.0)
- `probability` (float): Probability of applying (0.0 to 1.0)

**Returns:** NumPy array (perspective-transformed image)

**Example:**
```python
# Apply perspective with 50% probability
perspective = turboloader.RandomPerspective(0.3, 0.5)
output = perspective.apply(image)
```

---

### `Pad`

Pad image with specified mode and value.

**Constructor:**
```python
turboloader.Pad(
    pad_left, pad_top, pad_right, pad_bottom,
    mode=turboloader.PaddingMode.CONSTANT,
    value=0
)
```

**Parameters:**
- `pad_left` (int): Left padding in pixels
- `pad_top` (int): Top padding in pixels
- `pad_right` (int): Right padding in pixels
- `pad_bottom` (int): Bottom padding in pixels
- `mode` (PaddingMode, optional): Padding mode
  - `CONSTANT`: Fill with constant value (default)
  - `EDGE`: Replicate edge pixels
  - `REFLECT`: Reflect pixels at border
- `value` (int, optional): Fill value for CONSTANT mode (0-255)

**Returns:** NumPy array (padded image)

**Example:**
```python
# Pad 10 pixels on all sides with black
pad = turboloader.Pad(10, 10, 10, 10)
output = pad.apply(image)

# Pad with edge replication
pad_edge = turboloader.Pad(
    10, 10, 10, 10,
    mode=turboloader.PaddingMode.EDGE
)
output = pad_edge.apply(image)
```

---

## Color Transforms

### `ColorJitter`

Randomly adjust brightness, contrast, saturation, and hue.

**Constructor:**
```python
turboloader.ColorJitter(brightness, contrast, saturation, hue)
```

**Parameters:**
- `brightness` (float): Brightness jitter factor (0.0 to 1.0)
- `contrast` (float): Contrast jitter factor (0.0 to 1.0)
- `saturation` (float): Saturation jitter factor (0.0 to 1.0)
- `hue` (float): Hue jitter factor (0.0 to 0.5)

**Returns:** NumPy array (color-adjusted image)

**Example:**
```python
# Standard ImageNet augmentation
jitter = turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1)
output = jitter.apply(image)
```

**Implementation:** Uses HSV color space conversion with SIMD optimization.

---

### `Normalize`

Normalize image with custom mean and std per channel.

**Constructor:**
```python
turboloader.Normalize(mean, std)
```

**Parameters:**
- `mean` (list[float]): Mean values per channel [R, G, B]
- `std` (list[float]): Standard deviation per channel [R, G, B]

**Returns:** NumPy array of dtype float32

**Example:**
```python
# Custom normalization
normalize = turboloader.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
output = normalize.apply(image)  # float32 array
```

**Formula:** `output[c] = (input[c] / 255.0 - mean[c]) / std[c]`

---

### `ImageNetNormalize`

Convenience normalize with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

**Constructor:**
```python
turboloader.ImageNetNormalize()
```

**Parameters:** None

**Returns:** NumPy array of dtype float32

**Example:**
```python
# Quick ImageNet normalization
normalize = turboloader.ImageNetNormalize()
output = normalize.apply(image)  # float32 normalized
```

**Performance:** ~250,000 images/sec with AVX2 SIMD

---

### `Grayscale`

Convert RGB image to grayscale.

**Constructor:**
```python
turboloader.Grayscale()
```

**Parameters:** None

**Returns:** NumPy array of shape `(height, width, 1)`

**Example:**
```python
# Convert to grayscale
gray = turboloader.Grayscale()
output = gray.apply(image)  # (H, W, 1)
```

**Formula:** `Y = 0.299*R + 0.587*G + 0.114*B` (ITU-R BT.601 standard)

---

## Augmentation Transforms

### `RandomErasing`

Randomly erase rectangular region (Cutout augmentation).

**Constructor:**
```python
turboloader.RandomErasing(
    probability,
    area_fraction_min,
    area_fraction_max,
    aspect_ratio
)
```

**Parameters:**
- `probability` (float): Probability of erasing (0.0 to 1.0)
- `area_fraction_min` (float): Minimum erased area fraction (0.0 to 1.0)
- `area_fraction_max` (float): Maximum erased area fraction (0.0 to 1.0)
- `aspect_ratio` (float): Max aspect ratio of erased region

**Returns:** NumPy array (with random region erased)

**Example:**
```python
# Erase 2-40% of image with 50% probability
erase = turboloader.RandomErasing(0.5, 0.02, 0.4, 0.3)
output = erase.apply(image)
```

**Note:** Erased region filled with random pixel values.

---

### `GaussianBlur`

Apply Gaussian blur with random kernel size.

**Constructor:**
```python
turboloader.GaussianBlur(kernel_size, sigma_min, sigma_max)
```

**Parameters:**
- `kernel_size` (int): Blur kernel size (must be odd, e.g., 3, 5, 7)
- `sigma_min` (float): Minimum Gaussian sigma
- `sigma_max` (float): Maximum Gaussian sigma

**Returns:** NumPy array (blurred image)

**Example:**
```python
# Random blur for augmentation
blur = turboloader.GaussianBlur(5, 0.1, 2.0)
output = blur.apply(image)
```

**Performance:** SIMD-accelerated convolution

---

### `RandomPosterize`

Reduce image to random number of bits per channel.

**Constructor:**
```python
turboloader.RandomPosterize(bits_min, bits_max)
```

**Parameters:**
- `bits_min` (int): Minimum bits per channel (1-8)
- `bits_max` (int): Maximum bits per channel (1-8)

**Returns:** NumPy array (posterized image)

**Example:**
```python
# Posterize to 4-6 bits per channel
posterize = turboloader.RandomPosterize(4, 6)
output = posterize.apply(image)
```

**Performance:** ~336,000 images/sec (fastest transform)

---

### `RandomSolarize`

Invert pixels above random threshold.

**Constructor:**
```python
turboloader.RandomSolarize(threshold_min, threshold_max)
```

**Parameters:**
- `threshold_min` (int): Minimum threshold (0-255)
- `threshold_max` (int): Maximum threshold (0-255)

**Returns:** NumPy array (solarized image)

**Example:**
```python
# Solarize with threshold 128-200
solarize = turboloader.RandomSolarize(128, 200)
output = solarize.apply(image)
```

**Formula:** `output[i] = input[i] if input[i] < threshold else (255 - input[i])`

---

### `AutoAugment`

Apply learned AutoAugment policy (ImageNet, CIFAR10, or SVHN).

**Constructor:**
```python
turboloader.AutoAugment(policy=turboloader.AutoAugmentPolicy.IMAGENET)
```

**Parameters:**
- `policy` (AutoAugmentPolicy, optional): Augmentation policy
  - `IMAGENET`: ImageNet policy (default)
  - `CIFAR10`: CIFAR-10 policy
  - `SVHN`: SVHN policy

**Returns:** NumPy array (augmented image)

**Example:**
```python
# Apply ImageNet AutoAugment
autoaugment = turboloader.AutoAugment()
output = autoaugment.apply(image)

# CIFAR-10 policy
autoaugment_cifar = turboloader.AutoAugment(
    turboloader.AutoAugmentPolicy.CIFAR10
)
output_cifar = autoaugment_cifar.apply(image)
```

**Note:** Applies randomly selected sub-policy from learned set.

---

## Tensor Conversion

### `ToTensor`

Convert NumPy image array to tensor format (PyTorch CHW or TensorFlow HWC).

**Constructor:**
```python
turboloader.ToTensor(format=turboloader.TensorFormat.PYTORCH_CHW)
```

**Parameters:**
- `format` (TensorFormat, optional): Output tensor format
  - `PYTORCH_CHW`: PyTorch format (C, H, W) - default
  - `TENSORFLOW_HWC`: TensorFlow format (H, W, C)
  - `NONE`: No conversion (H, W, C)

**Returns:** NumPy array in specified format

**Example:**
```python
# Convert to PyTorch format (C, H, W)
to_tensor = turboloader.ToTensor()
output = to_tensor.apply(image)  # (3, H, W)

# Convert to TensorFlow format (H, W, C)
to_tensor_tf = turboloader.ToTensor(
    turboloader.TensorFormat.TENSORFLOW_HWC
)
output_tf = to_tensor_tf.apply(image)  # (H, W, 3)
```

**Note:** This only reshapes the array. Use with `torch.from_numpy()` or `tf.convert_to_tensor()` for framework tensors.

---

## Performance Notes

All transforms use SIMD acceleration when available:
- **x86_64**: AVX2 or AVX-512
- **ARM64**: NEON

### Throughput (Single-threaded on M4 Max)

| Transform | Throughput (img/s) | SIMD |
|-----------|-------------------|------|
| RandomPosterize | 336,053 | ✓ |
| RandomHorizontalFlip | 310,477 | ✓ |
| Resize (256→224) | 285,000 | ✓ |
| ImageNetNormalize | 250,000 | ✓ |
| ColorJitter | 180,000 | ✓ |
| GaussianBlur | 120,000 | ✓ |
| RandomAffine | 95,000 | ✓ |

*Benchmarked on 256x256 RGB images*

---

## Complete Example

Typical ImageNet training pipeline:

```python
import turboloader
import torch

# Training transforms
train_transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.RandomCrop(224, 224),
    turboloader.RandomHorizontalFlip(0.5),
    turboloader.ColorJitter(0.2, 0.2, 0.2, 0.1),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()
])

# Validation transforms (deterministic)
val_transforms = turboloader.Compose([
    turboloader.Resize(256, 256),
    turboloader.CenterCrop(224, 224),
    turboloader.ImageNetNormalize(),
    turboloader.ToTensor()
])

# Create DataLoader
loader = turboloader.DataLoader(
    'imagenet_train.tar',
    batch_size=256,
    num_workers=8,
    shuffle=True
)

# Training loop
for batch in loader:
    images = []
    labels = []

    for sample in batch:
        # Apply transforms
        img = train_transforms.apply(sample['image'])

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img).float()

        images.append(img_tensor)
        labels.append(sample['label'])

    # Stack batch
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Train model
    # outputs = model(images)
    # loss = criterion(outputs, labels)
    # ...
```

---

## See Also

- [Pipeline API](pipeline.md) - DataLoader configuration
- [Getting Started Guide](../getting-started.md) - Usage examples
- [PyTorch Integration](../guides/pytorch-integration.md) - Framework-specific patterns
- [Benchmarks](../benchmarks/index.md) - Performance analysis
