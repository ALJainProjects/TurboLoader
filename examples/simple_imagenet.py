#!/usr/bin/env python3
"""
Simple ImageNet Loading Example

Demonstrates basic usage of TurboLoader for ImageNet data loading.
This is the minimal example showing how to replace PyTorch DataLoader.
"""

import sys
import time
sys.path.insert(0, 'build/python')
import turboloader
import torch


def main():
    # Path to your ImageNet TAR file
    # Replace with your actual path
    tar_path = "/data/imagenet_train.tar"

    print("=" * 80)
    print("SIMPLE IMAGENET LOADING EXAMPLE")
    print("=" * 80)

    # Configure transforms (standard ImageNet preprocessing)
    transform_config = turboloader.TransformConfig()
    transform_config.target_width = 224
    transform_config.target_height = 224
    transform_config.resize_mode = "bilinear"
    transform_config.normalize = True
    transform_config.mean = [0.485, 0.456, 0.406]  # ImageNet mean
    transform_config.std = [0.229, 0.224, 0.225]   # ImageNet std
    transform_config.to_chw = True  # Convert to CHW format for PyTorch

    # Configure TurboLoader
    config = turboloader.Config()
    config.num_workers = 16  # Adjust based on your CPU cores
    config.queue_size = 512
    config.decode_jpeg = True
    config.enable_simd_transforms = True
    config.transform_config = transform_config

    # Create pipeline
    print("\nCreating TurboLoader pipeline...")
    pipeline = turboloader.Pipeline([tar_path], config)

    # Start loading
    print("Starting pipeline...")
    pipeline.start()

    # Fetch batches
    batch_size = 256
    num_batches = 100

    print(f"\nFetching {num_batches} batches of {batch_size} images each...")

    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx in range(num_batches):
        # Fetch next batch
        batch = pipeline.next_batch(batch_size)

        # Convert to PyTorch tensors
        images = []
        for sample in batch:
            # Get transformed image data (already normalized)
            img_data = sample.get_transformed_data()  # Shape: (3, 224, 224)
            images.append(torch.from_numpy(img_data))

        # Stack into batch tensor
        batch_tensor = torch.stack(images)  # Shape: (batch_size, 3, 224, 224)

        total_samples += len(batch)

        # Print progress
        if (batch_idx + 1) % 20 == 0:
            elapsed = time.perf_counter() - start_time
            throughput = total_samples / elapsed
            print(f"  Processed {batch_idx + 1}/{num_batches} batches "
                  f"({total_samples:,} images) @ {throughput:,.0f} img/s")

    # Final statistics
    total_time = time.perf_counter() - start_time
    throughput = total_samples / total_time

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total images:  {total_samples:,}")
    print(f"Total time:    {total_time:.2f} seconds")
    print(f"Throughput:    {throughput:,.2f} images/second")
    print(f"Avg batch time: {(total_time / num_batches) * 1000:.2f} ms")
    print("=" * 80)

    # Stop pipeline
    pipeline.stop()

    print("\nDone! TurboLoader successfully loaded ImageNet data.")


if __name__ == "__main__":
    main()
