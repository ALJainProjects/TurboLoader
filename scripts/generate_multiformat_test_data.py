#!/usr/bin/env python3
"""Generate test images in multiple formats for benchmarking"""

import os
import numpy as np
from PIL import Image
import tarfile

def generate_test_image(size=(224, 224)):
    """Generate a random RGB image"""
    return np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

def save_multiformat_dataset(output_dir, num_samples=100):
    """Generate dataset with JPEG, PNG, and WebP images"""

    os.makedirs(output_dir, exist_ok=True)

    formats = ['JPEG', 'PNG', 'WEBP']

    for fmt in formats:
        print(f"\nGenerating {num_samples} {fmt} images...")

        tar_path = os.path.join(output_dir, f'test_{fmt.lower()}.tar')

        with tarfile.open(tar_path, 'w') as tar:
            for i in range(num_samples):
                # Generate image
                img_array = generate_test_image()
                img = Image.fromarray(img_array, 'RGB')

                # Save to temporary file
                ext = 'jpg' if fmt == 'JPEG' else fmt.lower()
                temp_path = f'/tmp/sample_{i:05d}.{ext}'

                if fmt == 'JPEG':
                    img.save(temp_path, 'JPEG', quality=90)
                elif fmt == 'PNG':
                    img.save(temp_path, 'PNG', compress_level=6)
                elif fmt == 'WEBP':
                    img.save(temp_path, 'WEBP', quality=90)

                # Add to TAR
                tar.add(temp_path, arcname=f'sample_{i:05d}.{ext}')
                os.remove(temp_path)

                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{num_samples} images...")

        # Get file size
        size_mb = os.path.getsize(tar_path) / (1024 * 1024)
        print(f"  Created: {tar_path} ({size_mb:.2f} MB)")

def generate_mixed_format_tar(output_dir, num_samples=100):
    """Generate TAR with mixed formats (JPEG, PNG, WebP)"""

    print(f"\nGenerating {num_samples} mixed format images...")

    tar_path = os.path.join(output_dir, 'test_mixed.tar')

    with tarfile.open(tar_path, 'w') as tar:
        for i in range(num_samples):
            # Generate image
            img_array = generate_test_image()
            img = Image.fromarray(img_array, 'RGB')

            # Cycle through formats
            if i % 3 == 0:
                fmt, ext = 'JPEG', 'jpg'
                temp_path = f'/tmp/sample_{i:05d}.{ext}'
                img.save(temp_path, 'JPEG', quality=90)
            elif i % 3 == 1:
                fmt, ext = 'PNG', 'png'
                temp_path = f'/tmp/sample_{i:05d}.{ext}'
                img.save(temp_path, 'PNG', compress_level=6)
            else:
                fmt, ext = 'WEBP', 'webp'
                temp_path = f'/tmp/sample_{i:05d}.{ext}'
                img.save(temp_path, 'WEBP', quality=90)

            # Add to TAR
            tar.add(temp_path, arcname=f'sample_{i:05d}.{ext}')
            os.remove(temp_path)

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_samples} images...")

    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"  Created: {tar_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    output_dir = '/tmp/turboloader_multiformat'

    print("="*60)
    print("Generating Multi-Format Test Dataset")
    print("="*60)

    # Generate separate format datasets
    save_multiformat_dataset(output_dir, num_samples=100)

    # Generate mixed format dataset
    generate_mixed_format_tar(output_dir, num_samples=100)

    print("\n" + "="*60)
    print("Dataset generation complete!")
    print(f"Location: {output_dir}")
    print("="*60)
    print("\nFiles:")
    for f in sorted(os.listdir(output_dir)):
        path = os.path.join(output_dir, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  {f}: {size_mb:.2f} MB")
