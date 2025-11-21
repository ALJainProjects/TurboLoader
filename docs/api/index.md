# API Reference

Complete API documentation for TurboLoader v1.7.7.

## Quick Links

- [Pipeline API](pipeline.md) - DataLoader and configuration
- [Transforms API](transforms.md) - All 19 transforms
- [Tensor Conversion](tensor-conversion.md) - PyTorch/TensorFlow integration

## Module Overview

```python
import turboloader

# Core functionality
loader = turboloader.DataLoader(...)     # Main data loader
pipeline = turboloader.TransformPipeline()  # Transform composition

# Module functions
turboloader.version()          # Get version string
turboloader.features()         # Get feature flags
turboloader.list_transforms()  # List all transforms

# Enums
turboloader.InterpolationMode  # NEAREST, BILINEAR, BICUBIC, LANCZOS
turboloader.PaddingMode        # CONSTANT, EDGE, REFLECT
turboloader.TensorFormat       # NONE, PYTORCH_CHW, TENSORFLOW_HWC
turboloader.AutoAugmentPolicy  # IMAGENET, CIFAR10, SVHN
```

## Module Functions

### `turboloader.version()`

Get TurboLoader version string.

**Returns:**
- `str`: Version string (e.g., "1.7.7")

**Example:**
```python
import turboloader
print(turboloader.__version__)  # "1.7.7"
```

### `turboloader.features()`

Get TurboLoader feature flags and capabilities.

**Returns:**
- `dict`: Feature flags

**Keys:**
- `version` (str): Version string
- `tar_support` (bool): TAR archive support
- `remote_tar` (bool): HTTP/S3/GCS support
- `http_support` (bool): HTTP protocol
- `s3_support` (bool): S3 protocol
- `gcs_support` (bool): GCS protocol
- `jpeg_decode` (bool): JPEG decoding
- `png_decode` (bool): PNG decoding
- `webp_decode` (bool): WebP decoding
- `simd_acceleration` (bool): SIMD optimizations
- `lock_free_queues` (bool): Lock-free queues
- `num_transforms` (int): Number of transforms (19)
- `autoaugment` (bool): AutoAugment support
- `pytorch_tensors` (bool): PyTorch tensor conversion
- `tensorflow_tensors` (bool): TensorFlow tensor conversion
- `lanczos_interpolation` (bool): Lanczos resampling

**Example:**
```python
import turboloader

features = turboloader.features()
print(f"Version: {features['version']}")
print(f"Transforms: {features['num_transforms']}")
print(f"SIMD: {features['simd_acceleration']}")
```

### `turboloader.list_transforms()`

List all available transform names.

**Returns:**
- `list[str]`: Transform names

**Example:**
```python
import turboloader

transforms = turboloader.list_transforms()
print(f"Available transforms: {len(transforms)}")
for name in transforms:
    print(f"  - {name}")
```

## Enumerations

### `InterpolationMode`

Interpolation modes for image resizing.

**Values:**
- `NEAREST`: Nearest-neighbor (fastest, lowest quality)
- `BILINEAR`: Bilinear interpolation (good balance)
- `BICUBIC`: Bicubic interpolation (higher quality)
- `LANCZOS`: Lanczos resampling (highest quality, best for downsampling)

**Example:**
```python
import turboloader

resize = turboloader.Resize(
    224, 224,
    interpolation=turboloader.InterpolationMode.BILINEAR
)
```

### `PaddingMode`

Padding modes for image operations.

**Values:**
- `CONSTANT`: Pad with constant value
- `EDGE`: Pad with edge pixel values
- `REFLECT`: Reflect pixels at border

**Example:**
```python
import turboloader

pad = turboloader.Pad(
    padding=32,
    mode=turboloader.PaddingMode.REFLECT
)
```

### `TensorFormat`

Tensor format for framework compatibility.

**Values:**
- `NONE`: Keep as HWC uint8 (NumPy default)
- `PYTORCH_CHW`: Convert to CHW float32 (PyTorch)
- `TENSORFLOW_HWC`: Convert to HWC float32 (TensorFlow)

**Example:**
```python
import turboloader

to_tensor = turboloader.ToTensor(
    format=turboloader.TensorFormat.PYTORCH_CHW,
    normalize=True
)
```

### `AutoAugmentPolicy`

AutoAugment policy presets.

**Values:**
- `IMAGENET`: Optimized for ImageNet classification
- `CIFAR10`: Optimized for CIFAR-10
- `SVHN`: Optimized for Street View House Numbers

**Example:**
```python
import turboloader

autoaugment = turboloader.AutoAugment(
    policy=turboloader.AutoAugmentPolicy.IMAGENET
)
```

## Base Classes

### `Transform`

Base class for all image transforms.

**Methods:**

#### `apply(image: np.ndarray) -> np.ndarray`

Apply transform to image.

**Parameters:**
- `image` (np.ndarray): Input image (H, W, C) uint8

**Returns:**
- `np.ndarray`: Transformed image (H, W, C) uint8

**Example:**
```python
import turboloader
import numpy as np

transform = turboloader.Resize(224, 224)
image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
output = transform.apply(image)
print(output.shape)  # (224, 224, 3)
```

#### `name() -> str`

Get transform name.

**Returns:**
- `str`: Name of the transform

**Example:**
```python
import turboloader

transform = turboloader.Resize(224, 224)
print(transform.name())  # "Resize"
```

#### `is_deterministic() -> bool`

Check if transform is deterministic.

**Returns:**
- `bool`: True if transform produces same output for same input

**Example:**
```python
import turboloader

resize = turboloader.Resize(224, 224)
print(resize.is_deterministic())  # True

flip = turboloader.RandomHorizontalFlip(p=0.5)
print(flip.is_deterministic())  # False
```

### `TransformPipeline`

Compose multiple transforms into a pipeline.

**Methods:**

#### `__init__()`

Create empty transform pipeline.

**Example:**
```python
import turboloader

pipeline = turboloader.TransformPipeline()
```

#### `apply(image: np.ndarray) -> np.ndarray`

Apply all transforms in sequence.

**Parameters:**
- `image` (np.ndarray): Input image (H, W, C) uint8

**Returns:**
- `np.ndarray`: Transformed image

**Example:**
```python
import turboloader
import numpy as np

pipeline = turboloader.TransformPipeline()
# Note: Add transforms via C++ or use individual transform.apply()

image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
output = pipeline.apply(image)
```

## Transform List

### Geometric Transforms

- [Resize](transforms.md#resize) - SIMD-accelerated resizing
- [CenterCrop](transforms.md#centercrop) - Center crop
- [RandomCrop](transforms.md#randomcrop) - Random crop with padding
- [RandomRotation](transforms.md#randomrotation) - Random rotation
- [RandomAffine](transforms.md#randomaffine) - Affine transformations
- [RandomPerspective](transforms.md#randomperspective) - Perspective warp

### Color Transforms

- [Normalize](transforms.md#normalize) - Normalization
- [ImageNetNormalize](transforms.md#imagenetnormalize) - ImageNet preset
- [ColorJitter](transforms.md#colorjitter) - Color adjustments
- [Grayscale](transforms.md#grayscale) - Convert to grayscale
- [RandomPosterize](transforms.md#randomposterize) - Bit depth reduction
- [RandomSolarize](transforms.md#randomsolarize) - Threshold inversion

### Spatial Transforms

- [RandomHorizontalFlip](transforms.md#randomhorizontalflip) - Horizontal flip
- [RandomVerticalFlip](transforms.md#randomverticalflip) - Vertical flip
- [Pad](transforms.md#pad) - Image padding

### Effect Transforms

- [GaussianBlur](transforms.md#gaussianblur) - Gaussian blur
- [RandomErasing](transforms.md#randomerasing) - Random erasing (Cutout)

### Advanced Transforms

- [AutoAugment](transforms.md#autoaugment) - Learned augmentation policies

### Tensor Conversion

- [ToTensor](tensor-conversion.md#totensor) - Convert to tensor format

## See Also

- [Pipeline API](pipeline.md) - DataLoader configuration
- [Transforms API](transforms.md) - Detailed transform reference
- [Getting Started Guide](../getting-started.md) - Usage examples
