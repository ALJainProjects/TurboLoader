#!/usr/bin/env python3
"""
Generate synthetic TAR dataset for benchmarking.

Creates WebDataset-style TAR files with dummy images and labels.
"""

import os
import io
import tarfile
import numpy as np
from PIL import Image
import json
import argparse


def generate_image(width=224, height=224):
    """Generate random RGB image."""
    # Random noise image
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data)


def generate_tar_shard(output_path, num_samples, image_size=(224, 224)):
    """Generate a single TAR shard with samples."""

    print(f"Generating {output_path} with {num_samples} samples...")

    with tarfile.open(output_path, 'w') as tar:
        for i in range(num_samples):
            sample_id = f"{i:06d}"

            # Generate image
            img = generate_image(*image_size)
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()

            # Create image tar entry
            img_info = tarfile.TarInfo(name=f"{sample_id}.jpg")
            img_info.size = len(img_bytes)
            tar.addfile(img_info, io.BytesIO(img_bytes))

            # Generate label
            label = {"class": i % 1000, "sample_id": i}
            label_str = json.dumps(label)
            label_bytes = label_str.encode('utf-8')

            # Create label tar entry
            label_info = tarfile.TarInfo(name=f"{sample_id}.json")
            label_info.size = len(label_bytes)
            tar.addfile(label_info, io.BytesIO(label_bytes))

            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Created {output_path} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic TAR dataset')
    parser.add_argument('--output-dir', default='/tmp/turboloader_test_data',
                       help='Output directory for TAR files')
    parser.add_argument('--num-shards', type=int, default=4,
                       help='Number of TAR shards to create')
    parser.add_argument('--samples-per-shard', type=int, default=1000,
                       help='Number of samples per shard')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (square)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating test dataset:")
    print(f"  Output: {args.output_dir}")
    print(f"  Shards: {args.num_shards}")
    print(f"  Samples per shard: {args.samples_per_shard}")
    print(f"  Total samples: {args.num_shards * args.samples_per_shard}")
    print(f"  Image size: {args.image_size}x{args.image_size}")
    print()

    # Generate shards
    for shard_idx in range(args.num_shards):
        output_path = os.path.join(args.output_dir, f"shard_{shard_idx:04d}.tar")
        generate_tar_shard(
            output_path,
            args.samples_per_shard,
            (args.image_size, args.image_size)
        )

    print()
    print("Dataset generation complete!")
    print(f"Total size: {sum(os.path.getsize(os.path.join(args.output_dir, f)) for f in os.listdir(args.output_dir)) / (1024**2):.1f} MB")


if __name__ == '__main__':
    main()
