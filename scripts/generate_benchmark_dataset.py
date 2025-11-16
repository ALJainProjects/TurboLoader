#!/usr/bin/env python3
"""
Generate benchmark datasets for TurboLoader performance testing.

Creates realistic synthetic image datasets in multiple formats:
- Individual JPEG files (for PIL baseline)
- TAR archive (for TurboLoader)
- With labels for classification tasks
"""

import os
import sys
import time
import tarfile
import json
import random
import argparse
from pathlib import Path
from typing import Tuple

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy are required")
    print("Install with: pip install Pillow numpy")
    sys.exit(1)


def generate_realistic_image(size: Tuple[int, int] = (256, 256),
                            complexity: str = 'medium') -> np.ndarray:
    """
    Generate a realistic-looking synthetic image with patterns.

    Args:
        size: Image dimensions (height, width)
        complexity: 'simple', 'medium', or 'complex' - affects image entropy

    Returns:
        NumPy array (H, W, 3) with RGB image data
    """
    h, w = size

    if complexity == 'simple':
        # Solid colors with gradients
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for c in range(3):
            gradient = np.linspace(0, 255, h).astype(np.uint8)
            img[:, :, c] = gradient[:, np.newaxis]
        return img

    elif complexity == 'medium':
        # Random noise - fast and realistic compression
        return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    else:  # complex
        # Procedural patterns (checkerboard + noise)
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cell_size = 16
        for i in range(0, h, cell_size):
            for j in range(0, w, cell_size):
                color = random.randint(100, 200) if (i//cell_size + j//cell_size) % 2 == 0 else random.randint(50, 150)
                img[i:i+cell_size, j:j+cell_size] = color

        # Add noise
        noise = np.random.randint(-30, 30, (h, w, 3), dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


def generate_dataset(output_dir: str,
                    num_images: int = 2000,
                    image_size: Tuple[int, int] = (256, 256),
                    num_classes: int = 10,
                    create_tar: bool = True,
                    create_files: bool = True,
                    quality: int = 90):
    """
    Generate a complete benchmark dataset.

    Args:
        output_dir: Directory to save dataset
        num_images: Number of images to generate
        image_size: Image dimensions (height, width)
        num_classes: Number of classification classes
        create_tar: Create TAR archive
        create_files: Create individual files
        quality: JPEG quality (1-100)
    """

    print("="*80)
    print(f"BENCHMARK DATASET GENERATOR")
    print("="*80)
    print(f"Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of images: {num_images}")
    print(f"  Image size: {image_size[0]}x{image_size[1]}")
    print(f"  Number of classes: {num_classes}")
    print(f"  JPEG quality: {quality}")
    print(f"  Create TAR: {create_tar}")
    print(f"  Create files: {create_files}")
    print("="*80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    images_dir = output_path / "images"
    if create_files:
        images_dir.mkdir(exist_ok=True)

    # Generate metadata
    metadata = {
        'num_images': num_images,
        'image_size': image_size,
        'num_classes': num_classes,
        'quality': quality,
        'labels': {}
    }

    # Generate images
    print(f"\nGenerating {num_images} images...")
    start_time = time.time()

    temp_files = []

    for i in range(num_images):
        # Assign class label
        label = i % num_classes
        metadata['labels'][f'{i:06d}.jpg'] = label

        # Generate image
        img_array = generate_realistic_image(image_size, complexity='medium')
        img = Image.fromarray(img_array, mode='RGB')

        # Save to file if requested
        if create_files:
            img_path = images_dir / f'{i:06d}.jpg'
            img.save(img_path, 'JPEG', quality=quality)

        # Save to temp for TAR
        if create_tar:
            temp_path = f'/tmp/benchmark_temp_{i:06d}.jpg'
            img.save(temp_path, 'JPEG', quality=quality)
            temp_files.append(temp_path)

        # Progress update
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Progress: {i + 1}/{num_images} ({rate:.1f} img/s)")

    generation_time = time.time() - start_time
    print(f"\nImage generation completed in {generation_time:.1f}s")
    print(f"Average: {num_images/generation_time:.1f} images/second")

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {metadata_path}")

    # Create TAR archive
    if create_tar:
        print(f"\nCreating TAR archive...")
        tar_start = time.time()
        tar_path = output_path / "dataset.tar"

        with tarfile.open(tar_path, 'w') as tar:
            # Add images
            for i, temp_file in enumerate(temp_files):
                arcname = f'{i:06d}.jpg'
                tar.add(temp_file, arcname=arcname)

                if (i + 1) % 500 == 0:
                    print(f"  Archived: {i + 1}/{num_images}")

            # Add metadata
            tar.add(str(metadata_path), arcname='metadata.json')

        tar_time = time.time() - tar_start
        tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)

        print(f"\nTAR creation completed in {tar_time:.1f}s")
        print(f"TAR size: {tar_size_mb:.2f} MB")

        # Clean up temp files
        for temp_file in temp_files:
            os.remove(temp_file)

    # Summary
    print(f"\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)

    if create_files:
        files_size_mb = sum(f.stat().st_size for f in images_dir.glob('*.jpg')) / (1024 * 1024)
        print(f"Individual files: {images_dir}")
        print(f"Files size: {files_size_mb:.2f} MB")

    if create_tar:
        print(f"TAR archive: {tar_path}")
        print(f"TAR size: {tar_size_mb:.2f} MB")

    print(f"Metadata: {metadata_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Generate benchmark dataset for TurboLoader')
    parser.add_argument('--output', '-o', type=str, default='/private/tmp/benchmark_datasets/bench_2k',
                       help='Output directory (default: /private/tmp/benchmark_datasets/bench_2k)')
    parser.add_argument('--num-images', '-n', type=int, default=2000,
                       help='Number of images to generate (default: 2000)')
    parser.add_argument('--size', '-s', type=int, default=256,
                       help='Image size in pixels (square) (default: 256)')
    parser.add_argument('--classes', '-c', type=int, default=10,
                       help='Number of classes (default: 10)')
    parser.add_argument('--quality', '-q', type=int, default=90,
                       help='JPEG quality 1-100 (default: 90)')
    parser.add_argument('--no-tar', action='store_true',
                       help='Skip TAR archive creation')
    parser.add_argument('--no-files', action='store_true',
                       help='Skip individual file creation')

    args = parser.parse_args()

    # Validate arguments
    if args.no_tar and args.no_files:
        print("Error: Cannot disable both TAR and files output")
        sys.exit(1)

    if args.quality < 1 or args.quality > 100:
        print("Error: Quality must be between 1 and 100")
        sys.exit(1)

    # Generate dataset
    generate_dataset(
        output_dir=args.output,
        num_images=args.num_images,
        image_size=(args.size, args.size),
        num_classes=args.classes,
        create_tar=not args.no_tar,
        create_files=not args.no_files,
        quality=args.quality
    )


if __name__ == '__main__':
    main()
