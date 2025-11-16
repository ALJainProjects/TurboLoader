#!/usr/bin/env python3
"""
NVIDIA DALI Benchmark

Measures the performance of NVIDIA DALI (Data Loading Library), a GPU-accelerated
data loading and preprocessing library optimized for deep learning.

NVIDIA DALI Features:
- GPU-accelerated image decoding (nvJPEG)
- GPU-based data augmentation
- Optimized for NVIDIA GPUs
- Pipeline-based architecture
- Direct integration with PyTorch, TensorFlow, MXNet

Note: DALI provides maximum performance on systems with NVIDIA GPUs.
On CPU-only systems, it falls back to CPU-based operations.

Installation:
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110

Reference: https://docs.nvidia.com/deeplearning/dali/
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
    import torch
except ImportError:
    print("Error: numpy and torch required")
    print("Install with: pip install numpy torch")
    sys.exit(1)

try:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except ImportError:
    print("Error: NVIDIA DALI not installed")
    print("Install with: pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110")
    print("Note: DALI requires CUDA for GPU acceleration")
    sys.exit(1)


def extract_tar_to_images(tar_path: str, output_dir: str) -> str:
    """
    Extract TAR to individual image files for DALI.

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


@pipeline_def
def create_dali_pipeline(data_dir, device='cpu', shard_id=0, num_shards=1):
    """
    Create DALI pipeline for image loading and preprocessing.

    Args:
        data_dir: Directory containing images
        device: Device for processing ('cpu' or 'gpu')
        shard_id: Shard ID for distributed training
        num_shards: Total number of shards

    Returns:
        DALI pipeline
    """
    # Read images from directory
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        name='Reader'
    )

    # Decode JPEG (can use GPU if device='mixed')
    if device == 'gpu' or device == 'mixed':
        images = fn.decoders.image(images, device='mixed', output_type=types.RGB)
    else:
        images = fn.decoders.image(images, device='cpu', output_type=types.RGB)

    # Resize and crop
    images = fn.resize(images, resize_shorter=256, device=device)
    images = fn.crop(images, crop=(224, 224), device=device)

    # Normalize (ImageNet stats)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    images = fn.normalize(images, mean=mean, stddev=std, device=device)

    # Convert to HWC -> CHW format for PyTorch
    images = fn.transpose(images, perm=[2, 0, 1], device=device)

    return images, labels


def run_benchmark(tar_path: str,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  num_epochs: int = 3,
                  device: str = 'cpu',
                  work_dir: str = "/tmp/dali_benchmark") -> Dict[str, Any]:
    """
    Run NVIDIA DALI benchmark.

    Args:
        tar_path: Path to TAR file
        batch_size: Batch size
        num_workers: Number of worker threads
        num_epochs: Number of epochs
        device: Processing device ('cpu', 'gpu', or 'mixed')
        work_dir: Working directory

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("NVIDIA DALI BENCHMARK")
    print("="*80)
    print(f"TAR file: {tar_path}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print(f"Features: GPU-accelerated decoding (if available), pipeline architecture")
    print("="*80)

    # Setup working directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)

    image_dir = work_path / "images"

    # Extract TAR
    extract_start = time.time()
    extract_tar_to_images(tar_path, str(image_dir))
    extraction_time = time.time() - extract_start

    # Count images for dataset size
    image_files = sorted(image_dir.glob('*.jpg'))
    dataset_size = len(image_files)

    print(f"\nCreating DALI pipeline...")
    print(f"  Images: {dataset_size}")
    print(f"  Device: {device}")

    # Create DALI pipeline
    pipe = create_dali_pipeline(
        data_dir=str(image_dir),
        batch_size=batch_size,
        num_threads=num_workers,
        device_id=0,
        device=device
    )
    pipe.build()

    # Create PyTorch iterator
    dali_iter = DALIGenericIterator(
        pipe,
        ['images', 'labels'],
        reader_name='Reader',
        last_batch_policy=LastBatchPolicy.PARTIAL,
        auto_reset=True
    )

    print(f"DALI pipeline created")
    print(f"  Batches per epoch: {len(dali_iter)}")

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

        dali_iter.reset()

        for batch in dali_iter:
            batch_start = time.time()

            # Get images from DALI batch
            batch_images = batch[0]['images']

            # Simulate model forward pass
            _ = batch_images.mean()

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
        'framework': 'NVIDIA DALI',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'device': device,
        'backend': 'GPU-accelerated pipeline (if GPU available)',
        'extraction_time': extraction_time,
        'total_time': total_time,
        'total_time_with_extraction': total_time + extraction_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
        'std_epoch_time': np.std(epoch_times) if epoch_times else 0,
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'std_batch_time': np.std(batch_times) if batch_times else 0,
        'throughput': sample_count * num_epochs / total_time if total_time > 0 else 0,
        'throughput_with_extraction': sample_count * num_epochs / (total_time + extraction_time) if (total_time + extraction_time) > 0 else 0,
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
    parser = argparse.ArgumentParser(description='NVIDIA DALI benchmark')
    parser.add_argument('tar_path', type=str,
                       help='Path to TAR file containing images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of worker threads (default: 8)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--device', '-d', type=str, default='cpu',
                       choices=['cpu', 'gpu', 'mixed'],
                       help='Processing device (default: cpu)')
    parser.add_argument('--work-dir', type=str, default='/tmp/dali_benchmark',
                       help='Working directory (default: /tmp/dali_benchmark)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Validate TAR file exists
    if not os.path.exists(args.tar_path):
        print(f"Error: TAR file not found: {args.tar_path}")
        sys.exit(1)

    # Check GPU availability if requested
    if args.device in ['gpu', 'mixed']:
        if not torch.cuda.is_available():
            print("Warning: GPU requested but CUDA not available, falling back to CPU")
            args.device = 'cpu'

    # Run benchmark
    results = run_benchmark(
        tar_path=args.tar_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs,
        device=args.device,
        work_dir=args.work_dir
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
