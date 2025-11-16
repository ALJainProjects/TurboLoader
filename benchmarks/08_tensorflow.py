#!/usr/bin/env python3
"""
TensorFlow tf.data Benchmark

Measures the performance of TensorFlow's tf.data API, which provides
an optimized data loading pipeline for TensorFlow models.

TensorFlow tf.data Features:
- Pipeline parallelization with prefetch and parallel map
- Deterministic and non-deterministic execution modes
- Built-in transformations and augmentations
- Integration with TensorFlow ecosystem
- Auto-tuning capabilities

Configuration tested:
- AUTOTUNE for parallel operations
- Prefetching for pipeline overlap
- Parallel image decoding
- Standard ImageNet preprocessing

Installation:
    pip install tensorflow

Reference: https://www.tensorflow.org/guide/data
"""

import os
import sys
import time
import json
import argparse
import psutil
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Any

try:
    import numpy as np
except ImportError:
    print("Error: numpy required")
    print("Install with: pip install numpy")
    sys.exit(1)

try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow not installed")
    print("Install with: pip install tensorflow")
    sys.exit(1)


def extract_tar_to_images(tar_path: str, output_dir: str) -> str:
    """
    Extract TAR to individual image files for TensorFlow.

    Args:
        tar_path: Path to TAR file
        output_dir: Directory to extract to

    Returns:
        Path to directory with extracted images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting TAR to: {output_dir}")
    extract_start = time.time()

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(output_path)

    extract_time = time.time() - extract_start
    image_files = list(output_path.glob('*.jpg'))
    print(f"Extracted {len(image_files)} images in {extract_time:.2f}s")

    return str(output_path)


def preprocess_image(image_path):
    """
    Preprocess image using TensorFlow ops (ImageNet-style).

    Args:
        image_path: Path to image file

    Returns:
        Preprocessed image tensor
    """
    # Read image file
    image = tf.io.read_file(image_path)

    # Decode JPEG
    image = tf.image.decode_jpeg(image, channels=3)

    # Resize to 256 (shorter side)
    image = tf.image.resize(image, [256, 256])

    # Center crop to 224x224
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    # Convert to float32 and normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # ImageNet normalization
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std

    return image


def create_dataset(image_dir: str, batch_size: int = 32, num_parallel_calls: int = tf.data.AUTOTUNE):
    """
    Create TensorFlow dataset with optimized pipeline.

    Args:
        image_dir: Directory containing images
        batch_size: Batch size
        num_parallel_calls: Number of parallel calls for map operations

    Returns:
        tf.data.Dataset
    """
    # Get list of image files
    image_files = sorted(Path(image_dir).glob('*.jpg'))
    image_paths = [str(f) for f in image_files]

    print(f"\nCreating TensorFlow dataset:")
    print(f"  Images: {len(image_paths)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Parallel calls: {num_parallel_calls}")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)

    # Parallel image loading and preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=num_parallel_calls)

    # Batch
    dataset = dataset.batch(batch_size)

    # Prefetch for pipeline overlap
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, len(image_paths)


def run_benchmark(tar_path: str,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  num_epochs: int = 3,
                  work_dir: str = "/tmp/tensorflow_benchmark") -> Dict[str, Any]:
    """
    Run TensorFlow tf.data benchmark.

    Args:
        tar_path: Path to TAR file
        batch_size: Batch size
        num_workers: Number of parallel workers (mapped to num_parallel_calls)
        num_epochs: Number of epochs
        work_dir: Working directory

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("TENSORFLOW tf.data BENCHMARK")
    print("="*80)
    print(f"TAR file: {tar_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Features: AUTOTUNE, prefetching, parallel map")
    print("="*80)

    # Setup working directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    image_dir = work_path / "images"

    # Extract TAR
    extract_start = time.time()
    extract_tar_to_images(tar_path, str(image_dir))
    extraction_time = time.time() - extract_start

    # Create dataset
    dataset, dataset_size = create_dataset(str(image_dir), batch_size, num_workers)

    print(f"\nDataset created:")
    print(f"  Total images: {dataset_size}")
    print(f"  Batches per epoch: {dataset_size // batch_size}")

    # Track metrics
    epoch_times = []
    batch_times = []
    memory_usage = []
    process = psutil.Process()

    # Run benchmark
    total_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        batch_count = 0
        sample_count = 0

        for batch_images in dataset:
            batch_start = time.time()

            # Simulate model forward pass
            _ = tf.reduce_mean(batch_images)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            batch_count += 1
            sample_count += len(batch_images)

            # Track memory every 10 batches
            if batch_count % 10 == 0:
                mem_info = process.memory_info()
                memory_usage.append(mem_info.rss / (1024 * 1024))  # MB

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        throughput = sample_count / epoch_time if epoch_time > 0 else 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Batches: {batch_count}")
        print(f"  Samples: {sample_count}")
        print(f"  Throughput: {throughput:.1f} images/sec")

    total_time = time.time() - total_start

    # Clean up
    print(f"\nCleaning up: {work_dir}")
    shutil.rmtree(work_dir, ignore_errors=True)

    # Calculate statistics
    results = {
        'framework': 'TensorFlow tf.data',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'backend': 'tf.data with AUTOTUNE and prefetching',
        'extraction_time': extraction_time,
        'total_time': total_time,
        'total_time_with_extraction': total_time + extraction_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'std_epoch_time': np.std(epoch_times) if epoch_times else 0,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'std_batch_time': np.std(batch_times) if batch_times else 0,
        'throughput': dataset_size * num_epochs / total_time if total_time > 0 else 0,
        'throughput_with_extraction': dataset_size * num_epochs / (total_time + extraction_time) if (total_time + extraction_time) > 0 else 0,
        'peak_memory_mb': max(memory_usage) if memory_usage else 0,
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
    }

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Extraction time: {extraction_time:.2f}s")
    print(f"Training time: {total_time:.2f}s")
    print(f"Total time (with extraction): {total_time + extraction_time:.2f}s")
    print(f"Average epoch time: {results['avg_epoch_time']:.2f}s ± {results['std_epoch_time']:.2f}s")
    print(f"Average batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"Throughput (training only): {results['throughput']:.1f} images/sec")
    print(f"Throughput (with extraction): {results['throughput_with_extraction']:.1f} images/sec")
    print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"Average memory: {results['avg_memory_mb']:.1f} MB")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='TensorFlow tf.data benchmark')
    parser.add_argument('tar_path', type=str,
                       help='Path to TAR file containing images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--work-dir', type=str, default='/tmp/tensorflow_benchmark',
                       help='Working directory (default: /tmp/tensorflow_benchmark)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Validate TAR file exists
    if not os.path.exists(args.tar_path):
        print(f"Error: TAR file not found: {args.tar_path}")
        sys.exit(1)

    # Run benchmark
    results = run_benchmark(
        tar_path=args.tar_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        work_dir=args.work_dir
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
