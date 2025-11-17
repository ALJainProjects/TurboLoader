#!/usr/bin/env python3
"""
Create test datasets for TurboLoader unit tests.

Generates small TAR files with test images for testing the prefetch pipeline.
"""

from PIL import Image
import tarfile
import os
import random
from pathlib import Path


def create_test_images(output_dir, num_images, size=(64, 64)):
    """Create test images in a directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = Image.new('RGB', size)
        pixels = img.load()

        # Fill with random colors
        for x in range(size[0]):
            for y in range(size[1]):
                pixels[x, y] = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255)
                )

        img.save(f"{output_dir}/{i:04d}.jpg", quality=85)


def create_tar_from_dir(tar_path, image_dir):
    """Create TAR file from directory of images."""
    with tarfile.open(tar_path, "w") as tar:
        for file in sorted(Path(image_dir).glob("*.jpg")):
            tar.add(file, arcname=file.name)


def main():
    print("=" * 80)
    print("Creating TurboLoader Test Datasets")
    print("=" * 80)

    # Dataset 1: Small dataset for basic tests (100 images)
    print("\n[1/3] Creating small test dataset (100 images)...")
    small_dir = "/tmp/test_images_small"
    create_test_images(small_dir, num_images=100, size=(64, 64))
    create_tar_from_dir("/tmp/test_prefetch_small.tar", small_dir)
    size_mb = os.path.getsize("/tmp/test_prefetch_small.tar") / 1024 / 1024
    print(f"  ✓ Created /tmp/test_prefetch_small.tar ({size_mb:.2f} MB)")

    # Dataset 2: Medium dataset for prefetch tests (500 images)
    print("\n[2/3] Creating medium test dataset (500 images)...")
    medium_dir = "/tmp/test_images_medium"
    create_test_images(medium_dir, num_images=500, size=(128, 128))
    create_tar_from_dir("/tmp/test_prefetch.tar", medium_dir)
    size_mb = os.path.getsize("/tmp/test_prefetch.tar") / 1024 / 1024
    print(f"  ✓ Created /tmp/test_prefetch.tar ({size_mb:.2f} MB)")

    # Dataset 3: Large dataset for benchmarks (1000 images)
    print("\n[3/3] Creating large test dataset (1000 images)...")
    large_dir = "/tmp/test_images_large"
    create_test_images(large_dir, num_images=1000, size=(256, 256))
    create_tar_from_dir("/tmp/test_prefetch_large.tar", large_dir)
    size_mb = os.path.getsize("/tmp/test_prefetch_large.tar") / 1024 / 1024
    print(f"  ✓ Created /tmp/test_prefetch_large.tar ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("Test Datasets Created Successfully")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - /tmp/test_prefetch_small.tar (100 images, for basic tests)")
    print("  - /tmp/test_prefetch.tar (500 images, for prefetch tests)")
    print("  - /tmp/test_prefetch_large.tar (1000 images, for benchmarks)")
    print("\nYou can now run: ./build/tests/test_prefetch_pipeline")
    print("=" * 80)


if __name__ == "__main__":
    main()
