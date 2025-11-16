#!/usr/bin/env python3
"""
Naive PyTorch DataLoader Benchmark

Measures the performance of PyTorch's standard DataLoader with basic configuration.
This represents a common starting point for many ML practitioners - using PyTorch
DataLoader with default settings and basic PIL-based transforms.

Configuration:
- Standard torch.utils.data.Dataset
- PIL Image loading
- torchvision transforms
- num_workers=4 (moderate parallelism)
- No prefetching or pin_memory optimizations

This provides a realistic baseline for comparison with optimized dataloaders.
"""

import os
import sys
import time
import json
import argparse
import psutil
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


class ImageDataset(data.Dataset):
    """
    Basic PyTorch Dataset for image loading.

    Uses PIL for image loading and torchvision transforms for preprocessing.
    This is a typical pattern used in PyTorch tutorials and examples.
    """

    def __init__(self, image_dir: str, transform=None):
        """
        Initialize dataset.

        Args:
            image_dir: Directory containing JPEG images
            transform: Optional torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        self.transform = transform

        if len(self.image_files) == 0:
            raise ValueError(f"No JPEG images found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and transform image"""
        img_path = self.image_files[idx]

        # Load image with PIL
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        # Dummy label (for classification tasks)
        label = 0

        return image, label


def get_transforms():
    """
    Get standard ImageNet-style transforms.

    Returns:
        torchvision.transforms.Compose
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def run_benchmark(image_dir: str,
                  batch_size: int = 32,
                  num_workers: int = 4,
                  num_epochs: int = 3) -> Dict[str, Any]:
    """
    Run naive PyTorch DataLoader benchmark.

    Args:
        image_dir: Directory containing images
        batch_size: Batch size
        num_workers: Number of worker processes
        num_epochs: Number of epochs

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("NAIVE PYTORCH DATALOADER BENCHMARK")
    print("="*80)
    print(f"Dataset: {image_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Num workers: {num_workers}")
    print(f"Epochs: {num_epochs}")
    print("="*80)

    # Create dataset
    dataset = ImageDataset(
        image_dir=image_dir,
        transform=get_transforms()
    )

    print(f"\nDataset initialized:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Batches per epoch: {len(dataset) // batch_size}")

    # Create dataloader (naive configuration)
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,  # Naive: no pin_memory
        prefetch_factor=2,  # Default PyTorch prefetch
        persistent_workers=False  # Naive: restart workers each epoch
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

    # Calculate statistics
    results = {
        'framework': 'PyTorch Naive DataLoader',
        'batch_size': batch_size,
        'num_workers': num_workers,
        'num_epochs': num_epochs,
        'persistent_workers': False,
        'pin_memory': False,
        'total_time': total_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'std_epoch_time': np.std(epoch_times),
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'throughput': len(dataset) * num_epochs / total_time,
        'peak_memory_mb': max(memory_usage) if memory_usage else 0,
        'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0,
    }

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Total time: {total_time:.2f}s")
    print(f"Average epoch time: {results['avg_epoch_time']:.2f}s ± {results['std_epoch_time']:.2f}s")
    print(f"Average batch time: {results['avg_batch_time']*1000:.2f}ms ± {results['std_batch_time']*1000:.2f}ms")
    print(f"Throughput: {results['throughput']:.1f} images/sec")
    print(f"Peak memory: {results['peak_memory_mb']:.1f} MB")
    print(f"Average memory: {results['avg_memory_mb']:.1f} MB")
    print("="*80)

    return results


def main():
    parser = argparse.ArgumentParser(description='Naive PyTorch DataLoader benchmark')
    parser.add_argument('image_dir', type=str,
                       help='Directory containing JPEG images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num-workers', '-w', type=int, default=4,
                       help='Number of worker processes (default: 4)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_epochs=args.epochs
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
