#!/usr/bin/env python3
"""
PIL Baseline Benchmark

Measures the baseline performance of loading and processing images using
pure Python PIL (Pillow) without any optimizations. This provides the
baseline reference for comparison with TurboLoader and other dataloaders.

Performance metrics:
- Throughput (images/sec)
- Memory usage (peak RSS)
- CPU utilization
- Per-epoch timing

This benchmark simulates a typical training loop by loading images,
applying basic transformations, and creating batches.
"""

import os
import sys
import time
import json
import argparse
import psutil
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy required")
    print("Install with: pip install Pillow numpy")
    sys.exit(1)


class PILDataLoader:
    """
    Simple PIL-based data loader.

    Loads images sequentially from a directory, applies basic transformations,
    and batches them for training.
    """

    def __init__(self,
                 image_dir: str,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 num_epochs: int = 1,
                 transform: bool = True):
        """
        Initialize PIL DataLoader.

        Args:
            image_dir: Directory containing JPEG images
            batch_size: Number of images per batch
            shuffle: Shuffle images (simple random shuffle)
            num_epochs: Number of epochs to iterate
            transform: Apply basic image transforms (resize, normalize)
        """
        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_epochs = num_epochs
        self.transform = transform

        # Get all image files
        self.image_files = sorted(self.image_dir.glob('*.jpg'))
        self.num_images = len(self.image_files)

        if self.num_images == 0:
            raise ValueError(f"No JPEG images found in {image_dir}")

        print(f"PILDataLoader initialized:")
        print(f"  Images: {self.num_images}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {self.num_images // batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Shuffle: {shuffle}")
        print(f"  Transform: {transform}")

    def _load_and_transform(self, img_path: Path) -> np.ndarray:
        """
        Load image and apply transformations.

        Args:
            img_path: Path to image file

        Returns:
            NumPy array (H, W, 3) with transformed image
        """
        # Load image
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            # Resize to 224x224 (standard ImageNet size)
            img = img.resize((224, 224), Image.BILINEAR)

        # Convert to numpy
        img_array = np.array(img, dtype=np.float32)

        if self.transform:
            # Normalize to [0, 1]
            img_array /= 255.0

            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_array = (img_array - mean) / std

        return img_array

    def __iter__(self):
        """Iterate over batches"""
        for epoch in range(self.num_epochs):
            # Shuffle if requested
            if self.shuffle:
                import random
                image_files = list(self.image_files)
                random.shuffle(image_files)
            else:
                image_files = self.image_files

            # Create batches
            for i in range(0, len(image_files), self.batch_size):
                batch_files = image_files[i:i + self.batch_size]

                # Load and transform images
                batch_images = []
                for img_path in batch_files:
                    img_array = self._load_and_transform(img_path)
                    batch_images.append(img_array)

                # Stack into batch
                if batch_images:
                    batch = np.stack(batch_images, axis=0)
                    yield batch


def run_benchmark(image_dir: str,
                  batch_size: int = 32,
                  num_epochs: int = 3,
                  shuffle: bool = False) -> Dict[str, Any]:
    """
    Run PIL baseline benchmark.

    Args:
        image_dir: Directory containing images
        batch_size: Batch size
        num_epochs: Number of epochs
        shuffle: Shuffle data

    Returns:
        Dictionary with benchmark results
    """
    print("="*80)
    print("PIL BASELINE BENCHMARK")
    print("="*80)
    print(f"Dataset: {image_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Shuffle: {shuffle}")
    print("="*80)

    # Create dataloader
    loader = PILDataLoader(
        image_dir=image_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        num_epochs=num_epochs,
        transform=True
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

        for batch in loader:
            batch_start = time.time()

            # Simulate model forward pass (just access the data)
            _ = batch.mean()

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            batch_count += 1
            sample_count += len(batch)

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
        'framework': 'PIL Baseline',
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'shuffle': shuffle,
        'total_time': total_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'std_epoch_time': np.std(epoch_times),
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'throughput': loader.num_images * num_epochs / total_time,
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
    parser = argparse.ArgumentParser(description='PIL baseline benchmark')
    parser.add_argument('image_dir', type=str,
                       help='Directory containing JPEG images')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--epochs', '-e', type=int, default=3,
                       help='Number of epochs (default: 3)')
    parser.add_argument('--shuffle', '-s', action='store_true',
                       help='Shuffle data')
    parser.add_argument('--output', '-o', type=str,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Run benchmark
    results = run_benchmark(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        shuffle=args.shuffle
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
