#!/usr/bin/env python3
"""
TurboLoader Transform Examples
Demonstrates all available transforms and their usage
"""

import numpy as np
import time

try:
    import turboloader as tl
except ImportError:
    print("TurboLoader not installed. Run: pip install turboloader")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("PIL not available. Install with: pip install pillow")
    Image = None


def create_sample_image(width=100, height=100):
    """Create a sample RGB image."""
    # Create a gradient image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x] = [
                int((x / width) * 255),      # Red gradient
                int((y / height) * 255),     # Green gradient
                128                           # Constant blue
            ]
    return image


def save_image(image, filename):
    """Save image using PIL if available."""
    if Image:
        img = Image.fromarray(image)
        img.save(filename)
        print(f"Saved: {filename}")
    else:
        print(f"Would save: {filename} (PIL not available)")


def demo_basic_transforms():
    """Demonstrate basic transforms."""
    print("\n=== BASIC TRANSFORMS ===\n")

    image = create_sample_image(200, 200)

    # Resize
    print("1. Resize Transform")
    resize = tl.Resize(100, 100, tl.InterpolationMode.BILINEAR)
    resized = resize.apply(image)
    print(f"   Original: {image.shape} -> Resized: {resized.shape}")
    save_image(resized, "output_resize.png")

    # Center Crop
    print("\n2. Center Crop")
    crop = tl.CenterCrop(150, 150)
    cropped = crop.apply(image)
    print(f"   Original: {image.shape} -> Cropped: {cropped.shape}")
    save_image(cropped, "output_centercrop.png")

    # Horizontal Flip
    print("\n3. Horizontal Flip")
    flip = tl.RandomHorizontalFlip(p=1.0, seed=42)  # Always flip
    flipped = flip.apply(image)
    print(f"   Original: {image.shape} -> Flipped: {flipped.shape}")
    save_image(flipped, "output_hflip.png")

    # Grayscale
    print("\n4. Grayscale")
    gray = tl.Grayscale(num_output_channels=3)  # Keep 3 channels
    grayed = gray.apply(image)
    print(f"   Original: {image.shape} -> Grayscale: {grayed.shape}")
    save_image(grayed, "output_grayscale.png")


def demo_augmentation_transforms():
    """Demonstrate augmentation transforms."""
    print("\n=== AUGMENTATION TRANSFORMS ===\n")

    image = create_sample_image(150, 150)

    # Color Jitter
    print("1. Color Jitter")
    jitter = tl.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1,
        seed=42
    )
    jittered = jitter.apply(image)
    print(f"   Applied color jitter")
    save_image(jittered, "output_colorjitter.png")

    # Rotation
    print("\n2. Random Rotation")
    rotate = tl.RandomRotation(degrees=45, expand=False, seed=42)
    rotated = rotate.apply(image)
    print(f"   Rotated by random angle")
    save_image(rotated, "output_rotation.png")

    # Gaussian Blur
    print("\n3. Gaussian Blur")
    blur = tl.GaussianBlur(kernel_size=11, sigma=2.0)
    blurred = blur.apply(image)
    print(f"   Applied Gaussian blur")
    save_image(blurred, "output_blur.png")

    # Random Erasing
    print("\n4. Random Erasing")
    erase = tl.RandomErasing(p=1.0, seed=42)  # Always erase
    erased = erase.apply(image)
    print(f"   Applied random erasing")
    save_image(erased, "output_erasing.png")

    # Padding
    print("\n5. Padding")
    pad = tl.Pad(20, tl.PaddingMode.REFLECT, 0)
    padded = pad.apply(image)
    print(f"   Original: {image.shape} -> Padded: {padded.shape}")
    save_image(padded, "output_pad.png")


def demo_pipeline():
    """Demonstrate transform pipeline."""
    print("\n=== TRANSFORM PIPELINE ===\n")

    image = create_sample_image(256, 256)

    # Create ImageNet-style training pipeline
    transforms = [
        tl.Resize(256, 256),
        tl.RandomCrop(224, 224, padding=32, seed=42),
        tl.RandomHorizontalFlip(p=0.5, seed=43),
        tl.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, seed=44),
        tl.GaussianBlur(kernel_size=5, sigma=1.5),
        tl.ImageNetNormalize(to_float=False),
    ]

    print("Pipeline:")
    for i, transform in enumerate(transforms, 1):
        print(f"  {i}. {transform.name()}")

    # Apply pipeline
    result = image.copy()
    for transform in transforms:
        result = transform.apply(result)

    print(f"\nFinal output shape: {result.shape}")
    save_image(result, "output_pipeline.png")


def benchmark_transforms():
    """Benchmark transform performance."""
    print("\n=== PERFORMANCE BENCHMARKS ===\n")

    # Large image for benchmarking
    image = create_sample_image(1000, 1000)

    benchmarks = [
        ("Resize (1000x1000 -> 224x224)", tl.Resize(224, 224)),
        ("RandomHorizontalFlip", tl.RandomHorizontalFlip(p=1.0, seed=42)),
        ("ColorJitter", tl.ColorJitter(0.4, 0.4, 0.4, 0.1, seed=42)),
        ("GaussianBlur (5x5)", tl.GaussianBlur(5, 1.5)),
        ("Grayscale", tl.Grayscale(1)),
        ("Normalize", tl.ImageNetNormalize(to_float=False)),
    ]

    iterations = 10

    for name, transform in benchmarks:
        start = time.time()
        for _ in range(iterations):
            _ = transform.apply(image)
        elapsed = time.time() - start

        avg_time = elapsed / iterations * 1000  # ms
        throughput = iterations / elapsed  # images/sec

        print(f"{name:30} {avg_time:6.2f} ms/image ({throughput:6.1f} img/s)")


def demo_tensor_conversion():
    """Demonstrate tensor conversion."""
    print("\n=== TENSOR CONVERSION ===\n")

    image = create_sample_image(100, 100)

    # Check features
    features = tl.features()
    print(f"PyTorch tensor support: {features.get('pytorch_tensors', False)}")
    print(f"TensorFlow tensor support: {features.get('tensorflow_tensors', False)}")
    print(f"SIMD acceleration: {features.get('simd_acceleration', False)}")

    # Show tensor format options
    print(f"\nAvailable tensor formats:")
    print(f"  - NONE: {tl.TensorFormat.NONE}")
    print(f"  - PYTORCH_CHW: {tl.TensorFormat.PYTORCH_CHW}")
    print(f"  - TENSORFLOW_HWC: {tl.TensorFormat.TENSORFLOW_HWC}")


def demo_interpolation_modes():
    """Compare different interpolation modes."""
    print("\n=== INTERPOLATION MODES ===\n")

    # Small image to upscale
    small_image = create_sample_image(10, 10)

    modes = [
        ("Nearest", tl.InterpolationMode.NEAREST),
        ("Bilinear", tl.InterpolationMode.BILINEAR),
        ("Bicubic", tl.InterpolationMode.BICUBIC),
    ]

    for name, mode in modes:
        resize = tl.Resize(100, 100, mode)
        result = resize.apply(small_image)
        print(f"{name:10} -> {result.shape}")
        save_image(result, f"output_interp_{name.lower()}.png")


def demo_padding_modes():
    """Compare different padding modes."""
    print("\n=== PADDING MODES ===\n")

    image = create_sample_image(50, 50)

    modes = [
        ("Constant", tl.PaddingMode.CONSTANT),
        ("Edge", tl.PaddingMode.EDGE),
        ("Reflect", tl.PaddingMode.REFLECT),
    ]

    for name, mode in modes:
        pad = tl.Pad(20, mode, 0)
        result = pad.apply(image)
        print(f"{name:10} -> {result.shape}")
        save_image(result, f"output_pad_{name.lower()}.png")


def main():
    """Run all examples."""
    print("=" * 60)
    print("TurboLoader Transform Examples")
    print("=" * 60)

    # Check version
    print(f"\nTurboLoader version: {tl.version()}")
    features = tl.features()
    print(f"Features: {', '.join(k for k, v in features.items() if v and k != 'version')}")

    # Run demos
    demo_basic_transforms()
    demo_augmentation_transforms()
    demo_interpolation_modes()
    demo_padding_modes()
    demo_pipeline()
    demo_tensor_conversion()
    benchmark_transforms()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
