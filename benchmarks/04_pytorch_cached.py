#!/usr/bin/env python3
"""
PyTorch DataLoader with Local File Caching Benchmark

Measures the performance of PyTorch's DataLoader with a local file caching strategy.
This approach pre-extracts TAR archives to local disk and loads from cached files.

Optimization Strategy:
- Extract TAR to local directory first (one-time cost)
- Load from cached files during training
- Leverage OS filesystem cache
- num_workers=8 for parallelism
- pin_memory=True for GPU transfer

This represents a common optimization where datasets are extracted once and
reused across multiple training runs.
"""

import os
import sys
import time
import json
import argparse
import psutil
import tarfile
import shutil
from pathlib import Path
from typing import Dict, Any

try:
    import torch
    import torch.utils.data as data
    from torchvision import transforms
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PyTorch, torchvision, PIL, and numpy required")
    print("Install with: pip install torch torchvision Pillow numpy")
    sys.exit(1)


class CachedImageDataset(data.Dataset):
    """PyTorch Dataset that loads from pre-cached local files"""

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        self.transform = transform

        if len(self.image_files) == 0:
            raise ValueError(f"No JPEG images found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        label = 0
        return image, label


def get_transforms():
    """Get optimized ImageNet-style transforms"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def extract_tar_to_cache(tar_path: str, cache_dir: str) -> str:
    """
    Extract TAR archive to cache directory.

    Args:
        tar_path: Path to TAR file
        cache_dir: Directory to extract to

    Returns:
        Path to cache directory with extracted files
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting TAR to cache: {cache_dir}")
    extract_start = time.time()

    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(cache_path)

    extract_time = time.time() - extract_start

    # Count extracted files
    image_files = list(cache_path.glob('*.jpg'))
    print(f"Extracted {len(image_files)} images in {extract_time:.2f}s")
    print(f"Extraction speed: {len(image_files) / extract_time:.1f} files/sec")

    return str(cache_path)


def run_benchmark(tar_path: str,
                  batch_size: int = 32,
                  num_workers: int = 8,
                  num_epochs: int = 3,
                  cache_dir: str = "/tmp/pytorch_cache") -> Dict[str, Any]:
    """
    Run PyTorch DataLoader benchmark with local file caching.

    Args:
        tar_path: Path to TAR file
        batch_size: Batch size
        num_workers: Number of worker processes
        num_epochs: Number of epochs
        cache_dir: Directory for cached files

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("PYTORCH DATALOADER WITH LOCAL FILE CACHING BENCHMARK")
    print("="*80)
    print(f"TAR file: {tar_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print(f"Optimizations: pin_memory=True, persistent_workers=True, local cache")
    print("="*80)

    # Extract TAR to cache (one-time cost)
    extract_start = time.time()
    image_dir = extract_tar_to_cache(tar_path, cache_dir)
    extraction_time = time.time() - extract_start

    # Create dataset
    dataset = CachedImageDataset(
        image_dir=image_dir,
        transform=get_transforms()
    )

    print(f"\nDataset initialized:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataset) // batch_size}")

    # Create optimized dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

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

        for batch_images, batch_labels in dataloader:
            batch_start = time.time()

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

        throughput = sample_count / epoch_time
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Batches: {batch_count}")
        print(f"  Samples: {sample_count}")
        print(f"  Throughput: {throughput:.1f} images/sec")

    total_time = time.time() - total_start

    # Clean up cache
    print(f"\nCleaning up cache: {cache_dir}")
    shutil.rmtree(cache_dir, ignore_errors=True)

    # Calculate statistics
    results = {
        'framework': 'PyTorch Cached DataLoader',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'persistent_workers': True,
        'pin_memory': True,
        'prefetch_factor': 4,
        'cache_strategy': 'local_file_extraction',
        'extraction_time': extraction_time,
        'total_time': total_time,
        'total_time_with_extraction': total_time + extraction_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'std_epoch_time': np.std(epoch_times),
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'throughput': len(dataset) * num_epochs / total_time,
        'throughput_with_extraction': len(dataset) * num_epochs / (total_time + extraction_time),
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
    parser = argparse.ArgumentParser(description='PyTorch DataLoader with caching benchmark')
    parser.add_argument('tar_path', type=str,
                       help='Path to TAR file containing images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=8,
                       help='Number of worker processes (default: 8)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--cache-dir', '-c', type=str, default='/tmp/pytorch_cache',
                       help='Cache directory (default: /tmp/pytorch_cache)')
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
        cache_dir=args.cache_dir
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
